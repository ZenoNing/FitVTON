import igl
import json
import pickle
import numpy as np
import yaml

import warp as wp

import warp.sim.render
from warp.sim.utils import implicit_laplacian_smoothing
import warp.collision.panel_assignment as assign
from warp.sim.collide import count_self_intersections, count_body_cloth_intersections
from warp.sim.integrator_xpbd import (
    apply_limb_end_stop_constraint_kernel,
    apply_lower_body_clearance_constraint_kernel,
    apply_untucked_waistband_fit_constraint_kernel,
    apply_waistband_side_constraint_kernel,
    replace_mesh_points,
    update_body_pose,
)

# Custom
from pygarment.meshgen.sim_config import PathCofig, SimConfig
from pygarment.pattern.core import BasicPattern


class Cloth:

    def __init__(self, 
                 name, config: SimConfig, paths: PathCofig, 
                 caching=False,
                 body_sequence=None, body_faces=None):

        self.caching = caching   # Saves intermediate frames, extra logs, etc.
        self.paths = paths
        self.name = name
        self.config = config

        self.sim_fps = config.sim_fps
        self.sim_substeps = config.sim_substeps
        self.zero_gravity_steps = config.zero_gravity_steps
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.usd_frame_time = 0.0 
        self.sim_use_graph = wp.get_device().is_cuda
        self.device = wp.get_device() if wp.get_device().is_cuda else 'cpu' 
        self.frame = -1

        self.c_scale = 1.0
        self.b_scale = 100.0
        if hasattr(paths, 'in_body_obj'):
            self.body_path = paths.in_body_obj
        # build_stage scales and shifts body_sequence into GarmentCode units.
        # Keep caller-owned cached sequences immutable across batch tasks.
        self.body_sequence = np.array(body_sequence, copy=True) if body_sequence is not None else None
        self.body_faces = body_faces

        # collision resolution options
        self.enable_body_smoothing = config.enable_body_smoothing
        self.enable_cloth_reference_drag = config.enable_cloth_reference_drag
        self.static_detected_frame = None  # Frame when static garment was detected
        self.dynamic_body_simulation = False  # Flag to enable dynamic body simulation after static garment simulation
        self.face_filter_reset = False  # Flag to track if face filter has been reset
        self.static_body_simulation = False  # Flag to enable static body simulation
        self.original_face_filters = None
        self.part_v_sequence = {}
        self.part_inds_dict = {}  # Dictionary to store indices of each part in the body sequence
        self.mesh_dict = {}

        # Build the stage -- model object, colliders, etc.
        self.build_stage(config)

        # -------- Final model settings ----------
        # NOTE: global_viscous_damping: (damping_factor, min_vel_damp, max_vel) 
        # apply damping when vel > min_vel_damp, and clamp vel below max_vel after damping
        # TODO Remove after refactoring Euler integrator
        self.model.global_viscous_damping = wp.vec3(
            (config.global_damping_factor, config.global_damping_effective_velocity, config.global_max_velocity))
        self.model.particle_max_velocity = config.global_max_velocity
        
        self.model.ground = config.ground  

        self.model.global_collision_filter = config.enable_global_collision_filter
        self.model.cloth_reference_drag = self.enable_cloth_reference_drag
        self.model.cloth_reference_margin = config.cloth_reference_margin
        self.model.cloth_reference_k = config.cloth_reference_k
        self.model.cloth_reference_watertight_whole_shape_index = 0
        self.model.enable_particle_particle_collisions = config.enable_particle_particle_collisions
        self.model.enable_triangle_particle_collisions = config.enable_triangle_particle_collisions
        self.model.enable_edge_edge_collisions = config.enable_edge_edge_collisions
        self.model.attachment_constraint = config.enable_attachment_constraint

        self.model.soft_contact_margin = config.soft_contact_margin
        self.model.soft_contact_ke = config.soft_contact_ke
        self.model.soft_contact_kd = config.soft_contact_kd
        self.model.soft_contact_kf = config.soft_contact_kf
        self.model.soft_contact_mu = config.soft_contact_mu

        self.model.particle_ke = config.particle_ke
        self.model.particle_kd = config.particle_kd
        self.model.particle_kf = config.particle_kf
        self.model.particle_mu = config.particle_mu
        self.model.particle_cohesion = config.particle_cohesion
        self.model.particle_adhesion = config.particle_adhesion

        #self.integrator = wp.sim.SemiImplicitIntegrator() #intialize semi-implicit time-integrator
        self.integrator = wp.sim.XPBDIntegrator() #intialize semi-implicit time-integrator
        self.state_0 = self.model.state() #returns state object for model (holds all *time-varying* data for a model)
        self.state_1 = self.model.state() #i.e. body/particle positions and velocities
        if self.caching:
            self.renderer = wp.sim.render.SimRenderer(self.model, str(paths.usd), scaling=1.0)

        if self.sim_use_graph:
            self.create_graph()

        self.last_verts = None
        self.current_verts = wp.array.numpy(self.state_0.particle_q)

    def build_stage(self, config):

        builder = wp.sim.ModelBuilder(gravity=0.0)
        # --------------- Load body info -----------------
        if self.body_sequence.any() and self.body_faces is not None:
            self.body_sequence *= self.b_scale
            self.shift_y = self.get_shift_param(self.body_sequence[-1]) 
            if self.shift_y:
                self.body_sequence[:, :, 1] += self.shift_y
            # self.shift_y_sequence = [self.get_shift_param(frame) for frame in self.body_sequence]
            # for i, shift in enumerate(self.shift_y_sequence):
            #     if shift:
            #         self.body_sequence[i, :, 1] += shift
            # self.shift_y = self.shift_y_sequence[0] if self.shift_y_sequence is not None else 0.0
            # Use body sequence and faces from the sample
            body_vertices = self.body_sequence[0]   # Use the first frame of the body sequence
            body_faces = self.body_faces
            body_indices = body_faces.flatten()
        else:
            body_vertices, body_indices, body_faces = self.load_obj(self.paths.in_body_obj)
            body_vertices = body_vertices * self.b_scale
            self.shift_y = self.get_shift_param(body_vertices)

            if self.shift_y:
                body_vertices[:, 1] = body_vertices[:, 1] + self.shift_y            
            
        body_seg = self.read_json(self.paths.body_seg) 

        # v_body should reflect the *current* body used in the stage (starts from frame 0)
        # and will be updated during simulation when the body is animated.
        self.v_body = body_vertices
        self.f_body = body_faces
        self.body_indices = body_indices

        # -------------- Load cloth ------------
        cloth_vertices, cloth_indices, cloth_faces = self.load_obj(self.paths.g_box_mesh)
        cloth_seg_dict = assign.read_segmentation(self.paths.g_mesh_segmentation)
        self.cloth_seg_dict = cloth_seg_dict
        stitching_vertices = cloth_seg_dict["stitch"] if 'stitch' in cloth_seg_dict.keys() else []

        cloth_vertices = cloth_vertices * self.c_scale
        if self.shift_y:
            cloth_vertices[:, 1] = cloth_vertices[:, 1] + self.shift_y
                
        self.v_cloth_init = cloth_vertices
        self.f_cloth = cloth_faces

        #Load ground truth stitching lengths
        if not self.paths.g_orig_edge_len.exists():
            orig_lens_dict = None
            print("no original length dict found")
        else:
            with open(self.paths.g_orig_edge_len, 'rb') as file:
                orig_lens_dict = pickle.load(file)

        cloth_pos = (0.0, 0.0, 0.0)
        cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0)) #no rotation, but orientation of cloth in world space

        builder.add_cloth_mesh_sewing_spring(
            pos=cloth_pos,
            rot=cloth_rot,
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=cloth_vertices,
            indices=cloth_indices,
            resolution_scale=config.resolution_scale,
            orig_lens=orig_lens_dict,
            stitching_vertices=stitching_vertices,
            density=config.garment_density,
            edge_ke=config.garment_edge_ke,
            edge_kd=config.garment_edge_kd,
            tri_ke=config.garment_tri_ke,
            tri_ka=config.garment_tri_ka,
            tri_kd=config.garment_tri_kd,
            tri_drag=config.garment_tri_drag,
            tri_lift=config.garment_tri_lift,
            radius=config.garment_radius,
            add_springs=True,
            spring_ke=config.spring_ke,
            spring_kd=config.spring_kd,
        )

        # ------------ Add a body -----------      
        if self.enable_body_smoothing:
            # Starts sim from smoothed-out body and slowly restores original details
            smoothing_total_smoothing_factor = config.smoothing_total_smoothing_factor
            smoothing_num_steps = config.smoothing_num_steps
            smoothing_recover_start_frame = config.smoothing_recover_start_frame
            smoothing_frame_gap_between_steps = config.smoothing_frame_gap_between_steps
            smoothing_step_size = smoothing_total_smoothing_factor / smoothing_num_steps
            self.body_smoothing_frames = [smoothing_recover_start_frame + smoothing_frame_gap_between_steps*i for i in range(smoothing_num_steps + 1)]
            self.body_smoothing_vertices_list = []
            self.body_smoothing_vertices_list = implicit_laplacian_smoothing(body_vertices, body_indices.reshape(-1, 3), 
                                                                             step_size=smoothing_step_size, 
                                                                             iters=smoothing_num_steps)
            body_vertices = self.body_smoothing_vertices_list.pop()
            self.body_smoothing_frames.pop()
            self.body_indices = body_indices
            self.body_vertices_device_buffer = wp.array(body_vertices, dtype=wp.vec3, device=self.device)
            self.v_body = body_vertices
        
        self.body_mesh = wp.sim.Mesh(body_vertices, body_indices)
        
        body_pos = wp.vec3(0.0, 0, 0.0)
        body_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0))


        # Cloth-body segemntation
        cloth_reference_labels, body_parts = assign.panel_assignment(
                        cloth_seg_dict, cloth_vertices, cloth_indices, wp.transform(cloth_pos, cloth_rot), 
                        body_seg, body_vertices, body_indices, wp.transform(body_pos, body_rot), 
                        device=self.device,
                        panel_init_labels=self._load_panel_labels(),
                        strategy='closest', 
                        merge_two_legs=True,
                        smpl_body=self.paths.use_smpl_seg
                        )  
        
        face_filters, particle_filter = [], []
        if config.enable_body_collision_filters:
            v_connectivity = self._build_vert_connectivity(cloth_vertices, cloth_indices)

            # Filter 0 – main-body exclusion for sleeves:
            #   Sleeve cloth (arm-labeled) skips collision with main-body faces
            #   so that sleeves can settle on the arms without snagging the torso.
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, ['main_body'], smpl_body=self.paths.use_smpl_seg))
            particle_filter = assign.assign_face_filter_points(
                cloth_reference_labels, 
                ['left_arm', 'right_arm', 'arms'],
                filter_id=0,
                vert_connectivity=v_connectivity,
            )

            # Filter 1 – arm exclusion for skirts:
            #   Skirt cloth (leg-labeled) skips collision with arm body faces
            #   so that the arms don't push the skirt during settling.
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, ['left_arm', 'right_arm', 'arms'], smpl_body=self.paths.use_smpl_seg))
            particle_filter = assign.assign_face_filter_points(
                cloth_reference_labels, 
                ['left_leg', 'right_leg', 'legs'],
                filter_id=1,
                vert_connectivity=v_connectivity,
                current_vertex_filter=particle_filter
            )


        self.body_shape_index = 0   # Body is the first collider object to be added
        builder.add_shape_mesh(
            body=-1,
            mesh=self.body_mesh,
            pos=body_pos,
            rot=body_rot,
            scale=wp.vec3(1.0,1.0,1.0), #performed body scaling above
            thickness=config.body_thickness,  
            mu=config.body_friction,
            face_filters=face_filters if face_filters else [[]],
            model_particle_filter_ids = particle_filter,
        )
        
        # ----- Attachment constraint -------

        if config.enable_attachment_constraint:
            self._add_attachment_labels(builder, config, body_seg)

        # ----- Global collision resolution error ---- 
        # for part in body_parts:
        #     part_v, part_inds = assign.extract_submesh(body_vertices, body_indices, body_parts[part])
        #     builder.add_cloth_reference_shape_mesh(
        #         mesh = wp.sim.Mesh(part_v, part_inds),
        #         name = part,
        #         pos = body_pos,
        #         rot = body_rot,
        #         scale = (1.0,1.0,1.0) #performed body scaling above
        #     )
        for part in body_parts:
            v_sub_indices = body_parts[part]
            part_v, part_inds = assign.extract_submesh(self.body_sequence[0], body_indices, v_sub_indices)
            self.part_inds_dict[part] = part_inds
            idx = np.array(v_sub_indices, dtype=np.int32)
            self.part_v_sequence[part] = self.body_sequence[:, idx, :] if self.body_sequence.any() else body_vertices[idx, :]

            self.mesh_dict[part] = wp.sim.Mesh(part_v, part_inds)
            builder.add_cloth_reference_shape_mesh(
                mesh = self.mesh_dict[part],
                name = part,
                pos = body_pos,
                rot = body_rot,
                scale = (1.0,1.0,1.0) #performed body scaling above
            )

        # NOTE: has a side-effect of filling up model.particle_reference_label array 
        self.body_parts_names2index = builder.add_cloth_reference_labels(
            cloth_reference_labels, 
            [   # NOTE: Not adding drag between legs and the body as it's useless and contradicts attachment
                ['left_arm', 'body'], 
                ['right_arm', 'body'], 
                ['left_leg', 'right_leg'],
                ['left_arm', 'left_leg'], 
                ['right_arm', 'left_leg'], 
                ['left_arm', 'right_leg'], 
                ['right_arm', 'right_leg'], 
                ['left_arm', 'legs'], 
                ['right_arm', 'legs'], 
            ]
        )  

        # ------- Finalize --------------
        self.model: wp.sim.Model = builder.finalize(device = self.device) #data is transferred to warp tensors, object used in simulation

    def _add_attachment_labels(self, builder, config, body_seg):
        # with open(self.paths.in_body_mes, 'r') as file:
        #     body_dict = yaml.load(file, Loader=yaml.SafeLoader)['body']

        with open(self.paths.g_vert_labels, 'r') as f:
            vertex_labels = yaml.load(f, Loader=yaml.SafeLoader)
        
        lables_present = False
        for i, attach_label in enumerate(config.attachment_labels):
            if attach_label in vertex_labels.keys() and len(vertex_labels[attach_label]) > 0:
                constaint_verts = vertex_labels[attach_label]
                # 保存每个区域的 constraint_verts 到 self 属性
                if attach_label == 'lower_interface':
                    self.waist_constraint_verts = constaint_verts
                    lables_present = True
                    waist_indices =  body_seg['spine']
                    self.waist_point_list = []
                    self.waist_normal_list = []
                    for body_vertices in self.body_sequence:
                        waist_points = body_vertices[waist_indices]
                        center = waist_points.mean(axis=0)
                        center[1] = center[1] - 5  # Slightly lower than the actual waist center
                        self.waist_point_list.append(center)
                        self.waist_normal_list.append(np.array([0, 1, 0]))  # Vertical normal
                    self.waist_point = wp.vec3(*self.waist_point_list[0])
                    self.waist_normal = wp.vec3(*self.waist_normal_list[0])
                    builder.add_attachment(
                        constaint_verts,
                        self.waist_point,
                        self.waist_normal,
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                elif attach_label == 'right_collar':
                    self.right_collar_constraint_verts = constaint_verts
                    lables_present = True
                    # 重新设计：取rightShoulder区域中离neck_points中心最近的点为attachment_point，方向为指向center的归一化向量
                    right_collar_indices = body_seg.get('rightShoulder', None)
                    neck_indices = body_seg.get('spine2', None) 
                    if right_collar_indices is not None and neck_indices is not None:
                        self.right_collar_point_list = []
                        self.right_collar_normal_list = []
                        scale = 0.5
                        for body_vertices in self.body_sequence:
                            collar_points = body_vertices[right_collar_indices]
                            neck_points = body_vertices[neck_indices]
                            center = neck_points.mean(axis=0)
                            dists = np.linalg.norm(collar_points - center, axis=1)
                            min_idx = np.argmin(dists)
                            attach_point = collar_points[min_idx]
                            interp_point = center + scale * (attach_point - center)
                            attach_norm = center - attach_point
                            norm = np.linalg.norm(attach_norm)
                            if norm > 1e-8:
                                attach_norm = attach_norm / norm
                            else:
                                attach_norm = np.array([1, 0, 0]) # fallback
                            self.right_collar_point_list.append(interp_point)
                            self.right_collar_normal_list.append(attach_norm)
                        self.right_collar_point = wp.vec3(*self.right_collar_point_list[0])
                        self.right_collar_normal = wp.vec3(*self.right_collar_normal_list[0])
                        builder.add_attachment(
                            constaint_verts, 
                            self.right_collar_point,
                            self.right_collar_normal,
                            stiffness = config.attachment_stiffness[i],
                            damping = config.attachment_damping[i]
                        )
                    else:
                        print(f"{self.name}::WARNING::right_collar or neck indices not found in body_seg. Skipped.")
                elif attach_label == 'left_collar':
                    self.left_collar_constraint_verts = constaint_verts
                    lables_present = True
                    # 参考 right_collar 的 attachment 逻辑
                    left_collar_indices = body_seg.get('leftShoulder', None)
                    neck_indices = body_seg.get('spine2', None)
                    if left_collar_indices is not None and neck_indices is not None:
                        self.left_collar_point_list = []
                        self.left_collar_normal_list = []
                        scale = 0.5
                        for body_vertices in self.body_sequence:
                            collar_points = body_vertices[left_collar_indices]
                            neck_points = body_vertices[neck_indices]
                            center = neck_points.mean(axis=0)
                            dists = np.linalg.norm(collar_points - center, axis=1)
                            min_idx = np.argmin(dists)
                            attach_point = collar_points[min_idx]
                            interp_point = center + scale * (attach_point - center)
                            attach_norm = center - attach_point
                            norm = np.linalg.norm(attach_norm)
                            if norm > 1e-8:
                                attach_norm = attach_norm / norm
                            else:
                                attach_norm = np.array([-1, 0, 0]) # fallback，左侧默认朝-x
                            self.left_collar_point_list.append(interp_point)
                            self.left_collar_normal_list.append(attach_norm)
                        self.left_collar_point = wp.vec3(*self.left_collar_point_list[0])
                        self.left_collar_normal = wp.vec3(*self.left_collar_normal_list[0])
                        builder.add_attachment(
                            constaint_verts, 
                            self.left_collar_point,
                            self.left_collar_normal,
                            stiffness = config.attachment_stiffness[i],
                            damping = config.attachment_damping[i]
                        )
                    else:
                        print(f"{self.name}::WARNING::left_collar or neck indices not found in body_seg. Skipped.")
                elif attach_label == 'strapless_top':
                    lables_present = True

                    # Attach under arm 
                    level = 0.0
                    try:
                        with open(self.paths.in_body_mes, 'r') as file:
                            _body_dict = yaml.load(file, Loader=yaml.SafeLoader)
                        # historical formats: either flat keys or nested under 'body'
                        body_meas = _body_dict.get('body', _body_dict)
                        level = (
                            float(body_meas['height'])
                            - float(body_meas['head_l'])
                            - float(body_meas['armscye_depth'])
                        )
                    except Exception:
                        # Fallback: place attachment slightly below the shoulder center
                        try:
                            shoulder_idx = []
                            if 'leftShoulder' in body_seg:
                                shoulder_idx += body_seg['leftShoulder']
                            if 'rightShoulder' in body_seg:
                                shoulder_idx += body_seg['rightShoulder']
                            if shoulder_idx:
                                level = float(np.mean(body_vertices[shoulder_idx, 1]) - 0.10 * self.b_scale)
                        except Exception:
                            level = 0.0
                    builder.add_attachment(
                        constaint_verts, 
                        wp.vec3(0, level, 0),  
                        wp.vec3(0., 1., 0.),    # Vertical attachment
                        stiffness = config.attachment_stiffness[i],
                        damping = config.attachment_damping[i]
                    )
                else:
                    print(f'{self.name}::WARNING::Requested attachment label {attach_label} '
                          'is not supported. Skipped')
                    continue
                    
                print(f'Using attachment for {attach_label} with {len(constaint_verts)} vertices')

        if not lables_present:
            # Loaded garment is not labeled -- update config
            config.enable_attachment_constraint = False
            config.update_min_steps()
            print(f'{self.name}::WARNING::Requested attachment labels {config.attachment_labels} '
                  'are not present. Attachment is turned off'
                )

    def _load_panel_labels(self):
        pattern = BasicPattern(self.paths.g_specs)

        labels = {}
        for name, panel in pattern.pattern['panels'].items():
            labels[name] = panel['label'] if 'label' in panel else ''

        return labels     

    def _sim_frame_with_substeps(self):
        """Basic scheme for simulating a frame update"""
        
        wp.sim.collide(self.model, self.state_0, self.sim_dt * self.sim_substeps)  # Generates contact points for the particles and rigid bodies
        # in the model, to be used in the contact dynamics kernel of the integrator
        # launches kernels

        for s in range(self.sim_substeps):
            self.state_0.clear_forces()  # set particle and body forces to 0s
            self.integrator.simulate(self.model, self.state_0, self.state_1,
                                     self.sim_dt)  # calculate semi-implicit Euler step
            # launches kernels and calculates new particle (and body) positions and velocities
            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)  # swap prev, new state

    def create_graph(self):
        # create update graph
        wp.capture_begin()  # Captures all subsequent kernel launches and memory operations on CUDA devices.
        
        self._sim_frame_with_substeps()

        self.graph = wp.capture_end()  # returns a handle to a CUDA graph object that can be launched with :func:`~warp.capture_launch()`
        # do not capture kernel launches anymore

    def update(self, frame):

        if (not self.dynamic_body_simulation) and (self.static_detected_frame is None):
            self.sim_substeps = 10  # 或者你喜欢的默认值
        elif frame > self.static_detected_frame and frame < self.static_detected_frame + len(self.body_sequence):
            self.sim_substeps = 10
        else:
            self.sim_substeps = 10

        with wp.ScopedTimer("simulate", print=False, active=True):
            if self.model.enable_particle_particle_collisions:
                # FIXME: Produces cuda errors when activated together with "enable_cloth_reference_drag"
                # Reason is unknown. Or not?
                self.model.particle_grid.build(self.state_0.particle_q, self.model.particle_max_radius * 2.0)
            if frame == self.zero_gravity_steps:
                self.model.gravity = np.array((0.0, -9.81, 0.0))
                if self.sim_use_graph:
                    self.create_graph()
            if self.enable_body_smoothing and frame in self.body_smoothing_frames:
                self.update_smooth_body_shape()
                if self.sim_use_graph:
                    self.create_graph()
            # if (self.model.attachment_constraint 
            #         and frame >= self.config.attachment_frames):  
            #     self.model.attachment_constraint = False
            #     if self.sim_use_graph:
            #         self.create_graph()
            if self.dynamic_body_simulation:
                wp.launch(
                    kernel=update_body_pose,
                    dim=len(self.body_sequence[0]),
                    inputs=[self.body_mesh.mesh.id,
                            wp.array(self.body_sequence[(frame-self.static_detected_frame)], dtype=wp.vec3),
                            (1.0 / self.sim_fps)],
                    device=self.device,
                )
                self.body_mesh.mesh.refit()

                # Keep CPU-side vertices in sync for downstream renderers (pythonrender).
                seq_idx = frame - self.static_detected_frame
                if 0 <= seq_idx < len(self.body_sequence):
                    self.v_body = self.body_sequence[seq_idx]

                for part in self.part_v_sequence:
                    wp.launch(
                        kernel=update_body_pose,
                        dim=len(self.part_v_sequence[part][0]),
                        inputs=[self.mesh_dict[part].mesh.id,
                                wp.array(self.part_v_sequence[part][(frame-self.static_detected_frame)], dtype=wp.vec3),
                                (1.0 / self.sim_fps)],
                        device=self.device,
                    )
                    self.mesh_dict[part].mesh.refit()
                
                # Update attachment points
                if self.model.attachment_constraint:
                    self.update_attachment(frame-self.static_detected_frame)

                # Reset the main-body exclusion filter (filter_id=0) so that
                # sleeves can collide with the torso during dynamic body morphing.
                # Keep filter_id=1 (arm exclusion for skirt) active so that
                # arms don't disturb the settled skirt during shape changes.
                if not self.face_filter_reset:
                    face_filters = self.model.shape_geo.face_filter.numpy()
                    face_filters[0, 0] = 0  # filter_id=0: main-body exclusion for sleeves
                    self.model.shape_geo.face_filter.assign(face_filters)
                    self.face_filter_reset = True

                if self.sim_use_graph:
                    self.create_graph()
            if (self.static_detected_frame is not None) and (self.dynamic_body_simulation is False):
                
                if self.static_body_simulation is False:
                    zero_body_velocities = np.zeros_like(self.body_mesh.mesh.velocities.numpy())
                    self.body_mesh.mesh.velocities.assign(zero_body_velocities)
                    self.static_body_simulation = True

                if self.sim_use_graph:  # GPU
                    self.create_graph()

            if self.sim_use_graph:  # GPU
                wp.capture_launch(self.graph)

            else:  # CPU: launch kernels without graph
                self._sim_frame_with_substeps()

            if hasattr(self, '_apply_lower_body_clearance_constraint'):
                self._apply_lower_body_clearance_constraint(frame)
            if hasattr(self, '_apply_untucked_waistband_fit_constraint'):
                self._apply_untucked_waistband_fit_constraint(frame)
            if hasattr(self, '_apply_limb_end_stop_constraints'):
                self._apply_limb_end_stop_constraints(frame)
            if hasattr(self, '_apply_waistband_side_constraint'):
                self._apply_waistband_side_constraint(frame)

            # Update vertices of last frame
            self.last_verts = self.current_verts
            # NOTE Makes a copy if particle_q device is not CPU
            self.current_verts = wp.array.numpy(self.state_0.particle_q)  

    def update_attachment(self, frame):
        """
        动态更新 attachment 点和法向量，支持 waist、left_collar、right_collar。
        """
        # 记录所有可用的 attachment 类型
        attachment_types = []
        if hasattr(self, 'waist_point_list') and hasattr(self, 'waist_normal_list'):
            attachment_types.append('waist')
        if hasattr(self, 'right_collar_point_list') and hasattr(self, 'right_collar_normal_list'):
            attachment_types.append('right_collar')
        if hasattr(self, 'left_collar_point_list') and hasattr(self, 'left_collar_normal_list'):
            attachment_types.append('left_collar')


        # 检查 frame 是否越界
        # 需要分别更新每个 attachment 区域的索引范围，避免覆盖
        if hasattr(self.model, 'attachment_point') and hasattr(self.model, 'attachment_norm'):
            # 获取每个区域的索引范围，使用 constraint_verts 的长度
            region_indices = {}
            offset = 0
            for att in attachment_types:
                verts_attr = f'{att}_constraint_verts'
                if hasattr(self, verts_attr):
                    region_len = len(getattr(self, verts_attr))
                else:
                    print(f"[update_attachment] {att} missing constraint_verts!")
                    region_len = 0
                region_indices[att] = (offset, offset + region_len)
                offset += region_len

            # 按区域分别赋值
            for att in attachment_types:
                point_list = getattr(self, f'{att}_point_list')
                normal_list = getattr(self, f'{att}_normal_list')
                if frame < 0 or frame >= len(point_list):
                    print(f"[update_attachment] {att} frame {frame} out of range!")
                    continue
                point = wp.vec3(*point_list[frame])
                normal = wp.vec3(*normal_list[frame])
                start, end = region_indices[att]
                if end > start:
                    self.model.attachment_point[start:end].assign(wp.array([point] * (end - start), dtype=wp.vec3, device=self.device))
                    self.model.attachment_norm[start:end].assign(wp.array([normal] * (end - start), dtype=wp.vec3, device=self.device))
            
    def update_smooth_body_shape(self):
        body_vertices = self.body_smoothing_vertices_list.pop()
        self.v_body = body_vertices
        wp.copy(self.body_vertices_device_buffer,
                wp.array(body_vertices, dtype=wp.vec3, device='cpu', copy=False))

        # Apply new vertices and refit the sructures
        wp.launch(
            kernel=replace_mesh_points,
            dim = len(body_vertices),
            inputs=[self.body_mesh.mesh.id,
                    self.body_vertices_device_buffer],
            device=self.device
        )
        self.body_mesh.mesh.refit()

        #update render
        if self.caching: 
            self.renderer.render_mesh(
                            f'shape_{self.body_shape_index}',
                            body_vertices,
                            None,
                            is_template=True,
                        )

    def render_usd_frame(self, is_live=False):
        with wp.ScopedTimer("render", print=False, active=True):
            start_time = 0.0 if is_live else self.usd_frame_time

            self.renderer.begin_frame(start_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

        self.usd_frame_time += 1.0 / self.sim_fps
        if not is_live:
            self.renderer.save()

    def run_frame(self):
        self.update(self.frame)

        # NOTE: USD Render
        if self.caching:
            self.render_usd_frame()
    
    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    
    def load_obj(self, path):
        v, f = igl.read_triangle_mesh(str(path))
        return v, f.flatten(), f

    def get_shift_param(self,body_vertices):
        v_body_arr = np.array(body_vertices)
        min_y = (min(v_body_arr[:, 1]))
        if min_y < 0:
            return abs(min_y)
        elif min_y > 0:
            return -min_y
        return 0.0

    def calc_norm(self, a, b, c):
        """
        This function calculates the norm based on the three points a, b, and c.
        Input:
            * self (BoxMesh object): Instance of BoxMesh class from which the function is called
            * a (ndarray): first point taking part in norm calculation
            * b (ndarray): second point taking part in norm calculation
            * c (ndarray): third point taking part in norm calculation
        Output:
            * n_normalized (bool): norm(a,b,c) with length 1
        """
        # Calculate the vectors AB and AC
        AB = np.array(b - a)
        AC = np.array(c - a)

        # Calculate the cross product of AB and AC
        n = np.cross(AB, AC)
        n_normalized = n / np.linalg.norm(n)

        return n_normalized

    def calc_vertex_norms(self):
        vertex_normals = np.zeros((len(self.v_cloth_init), 4))
        for face in self.f_cloth:
            v0, v1, v2 = np.array(self.current_verts)[face]
            face_norm = list(self.calc_norm(v0, v1, v2))
            temp_update = face_norm + [1]
            vertex_normals[face] += temp_update

        vertex_normals = vertex_normals[:, :3] / (vertex_normals[:, 3][:, np.newaxis])
        return vertex_normals

    def save_frame(self, save_v_norms=False): 
        """Save current garment state as an obj file, 
        re-using all the information from boxmesh 
        except for vertices and vertex normals (e.g. textures and faces)
        """
        
        # NOTE: igl routine is not used here because it cannot write any extra info (e.g. texture coords) into obj

        # stores v, f, vf and vn
        # Save cloth with texture and normals
        if save_v_norms:
            vertex_normals = self.calc_vertex_norms()

        v_cloth_sim = self.current_verts
        # Store simulated cloth mesh
        # Read the boxmesh file
        with open(self.paths.g_box_mesh, 'r') as obj_file:
            lines = obj_file.readlines()

        # Modify the vertex positions and normals, if required
        with open(self.paths.g_sim, 'w') as obj_file:
            v_idx = 0
            vn_idx = 0
            for line in lines:
                if line.startswith('v '):
                    new_vertex = v_cloth_sim[v_idx]
                    obj_file.write(f'v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                    v_idx += 1
                elif line.startswith('vn '):
                    if save_v_norms:
                        new_vertex = vertex_normals[vn_idx]
                        obj_file.write(f'vn {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                        vn_idx += 1
                else:
                    obj_file.write(line)

    def is_static(self):
        """
            Checks whether garment is in the static equilibrium
            Compares current state with the last recorded state
        """
        threshold = self.config.static_threshold
        non_static_percent = self.config.non_static_percent

        curr_verts_arr = self.current_verts
        last_verts_arr = self.last_verts

        if self.last_verts is None:  # first iteration
            return False, len(curr_verts_arr)

        # Compare L1 norm per vertex
        # Checking vertices change is the same as checking if velocity is zero
        diff = np.abs(curr_verts_arr - last_verts_arr)
        diff_L1 = np.sum(diff, axis=1)

        non_static_len = len(
            diff_L1[diff_L1 > threshold])  # compare vertex-wise to allow accurate control over outliers

        if non_static_len == 0 or (non_static_len < len(curr_verts_arr) * 0.01 * non_static_percent):
            print('\nStatic with {} non-static vertices out of {}'.format(non_static_len, len(curr_verts_arr)))
            # Store last frame
            return True, non_static_len
        else:
            return False, non_static_len

    def count_self_intersections(self):
        model = self.model

        if model.particle_count and model.spring_count: 
            model.particle_self_intersection_count.zero_()
            wp.launch(
                kernel=count_self_intersections,
                dim=model.spring_count,
                inputs=[
                    model.spring_indices,
                    model.particle_shape.id,
                ],
                outputs=[
                    model.particle_self_intersection_count
                ],
                device=model.device,
            )
            return int(wp.array.numpy(self.model.particle_self_intersection_count)[0])
        else: 
            return 0

    def count_body_intersections(self):
        model = self.model

        if model.particle_count:
            model.body_cloth_intersection_count.zero_()
            wp.launch(
                kernel=count_body_cloth_intersections,
                dim=model.spring_count,
                inputs=[
                    model.spring_indices,
                    model.particle_shape.id,
                    model.shape_geo,
                    self.body_shape_index
                ],
                outputs=[
                    model.body_cloth_intersection_count
                ],
                device=model.device,
            )
            return int(wp.array.numpy(self.model.body_cloth_intersection_count)[0])
        else:
            return 0 
        
    def _build_vert_connectivity(self, vertices, indices):
        vert_connectivity = [[] for _ in range(len(vertices))]

        for face_id in range(int(len(indices) / 3)):
            v1, v2, v3 = indices[face_id*3 + 0], indices[face_id*3 + 1], indices[face_id*3 + 2]
            
            vert_connectivity[v1].append(v2)
            vert_connectivity[v1].append(v3)

            vert_connectivity[v2].append(v1)
            vert_connectivity[v2].append(v3)

            vert_connectivity[v3].append(v1)
            vert_connectivity[v3].append(v2)

        return vert_connectivity


class MultiCloth(Cloth):
    """Multiple garment meshes in one Warp model for cloth-cloth interaction."""

    def __init__(
            self,
            name,
            config: SimConfig,
            garment_paths,
            garment_names=None,
            caching=False,
            body_sequence=None,
            body_faces=None,
            initial_vertices=None,
            waistband_side_mode=None):
        self.caching = caching
        self.paths = garment_paths[0]
        self.garment_paths = garment_paths
        self.garment_names = garment_names or [paths.in_tag for paths in garment_paths]
        self.initial_vertices = initial_vertices or {}
        self.waistband_side_mode = waistband_side_mode
        self.name = name
        self.config = config

        self.sim_fps = config.sim_fps
        self.sim_substeps = config.sim_substeps
        self.zero_gravity_steps = config.zero_gravity_steps
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.usd_frame_time = 0.0
        self.sim_use_graph = wp.get_device().is_cuda
        self.device = wp.get_device() if wp.get_device().is_cuda else 'cpu'
        self.frame = -1

        self.c_scale = 1.0
        self.b_scale = 100.0
        self.body_sequence = np.array(body_sequence, copy=True) if body_sequence is not None else None
        self.body_faces = body_faces

        self.enable_body_smoothing = config.enable_body_smoothing
        self.enable_cloth_reference_drag = config.enable_cloth_reference_drag
        self.static_detected_frame = None
        self.dynamic_body_simulation = False
        self.face_filter_reset = False
        self.static_body_simulation = False
        self.original_face_filters = None
        self.part_v_sequence = {}
        self.part_inds_dict = {}
        self.mesh_dict = {}
        self.garment_infos = []
        self.attachment_updates = []
        self.waistband_side_constraint = None
        self.lower_body_clearance_constraint = None
        self.untucked_waistband_fit_constraint = None
        self.limb_end_stop_constraints = []

        self.build_stage(config)

        self.model.global_viscous_damping = wp.vec3(
            (config.global_damping_factor, config.global_damping_effective_velocity, config.global_max_velocity))
        self.model.particle_max_velocity = config.global_max_velocity

        self.model.ground = config.ground
        self.model.global_collision_filter = config.enable_global_collision_filter
        self.model.cloth_reference_drag = self.enable_cloth_reference_drag
        self.model.cloth_reference_margin = config.cloth_reference_margin
        self.model.cloth_reference_k = config.cloth_reference_k
        self.model.cloth_reference_watertight_whole_shape_index = 0
        self.model.enable_particle_particle_collisions = config.enable_particle_particle_collisions
        self.model.enable_triangle_particle_collisions = config.enable_triangle_particle_collisions
        self.model.enable_edge_edge_collisions = config.enable_edge_edge_collisions
        self.model.attachment_constraint = config.enable_attachment_constraint

        self.model.soft_contact_margin = config.soft_contact_margin
        self.model.soft_contact_ke = config.soft_contact_ke
        self.model.soft_contact_kd = config.soft_contact_kd
        self.model.soft_contact_kf = config.soft_contact_kf
        self.model.soft_contact_mu = config.soft_contact_mu

        self.model.particle_ke = config.particle_ke
        self.model.particle_kd = config.particle_kd
        self.model.particle_kf = config.particle_kf
        self.model.particle_mu = config.particle_mu
        self.model.particle_cohesion = config.particle_cohesion
        self.model.particle_adhesion = config.particle_adhesion

        self.integrator = wp.sim.XPBDIntegrator()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if self.caching:
            self.renderer = wp.sim.render.SimRenderer(self.model, str(self.paths.usd), scaling=1.0)

        if self.sim_use_graph:
            self.create_graph()

        self.last_verts = None
        self.current_verts = wp.array.numpy(self.state_0.particle_q)

    def build_stage(self, config):
        builder = wp.sim.ModelBuilder(gravity=0.0)

        has_body_sequence = self.body_sequence is not None and self.body_faces is not None
        if has_body_sequence:
            self.body_sequence *= self.b_scale
            self.shift_y = self.get_shift_param(self.body_sequence[-1])
            if self.shift_y:
                self.body_sequence[:, :, 1] += self.shift_y
            body_vertices = self.body_sequence[0]
            body_faces = np.asarray(self.body_faces, dtype=np.int32)
            body_indices = body_faces.flatten()
        else:
            body_vertices, body_indices, body_faces = self.load_obj(self.paths.in_body_obj)
            body_vertices = body_vertices * self.b_scale
            self.shift_y = self.get_shift_param(body_vertices)
            if self.shift_y:
                body_vertices[:, 1] = body_vertices[:, 1] + self.shift_y

        body_vertices = np.asarray(body_vertices, dtype=np.float32)
        body_faces = np.asarray(body_faces, dtype=np.int32)
        body_indices = np.asarray(body_indices, dtype=np.int32)
        body_seg = self.read_json(self.paths.body_seg)
        self.v_body = body_vertices
        self.f_body = body_faces
        self.body_indices = body_indices
        self.body_mesh = wp.sim.Mesh(body_vertices, body_indices)
        body_pos = wp.vec3(0.0, 0.0, 0.0)
        body_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0))
        cloth_pos = (0.0, 0.0, 0.0)
        cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(0.0))

        combined_reference_labels = []
        combined_particle_filter = []
        body_parts_union = {}

        for garment_name, paths in zip(self.garment_names, self.garment_paths):
            start_vertex = len(builder.particle_q)
            cloth_vertices, cloth_indices, cloth_faces = self.load_obj(paths.g_box_mesh)
            cloth_seg_dict = assign.read_segmentation(paths.g_mesh_segmentation)
            stitching_vertices = cloth_seg_dict["stitch"] if 'stitch' in cloth_seg_dict.keys() else []

            cloth_vertices = cloth_vertices * self.c_scale
            if self.shift_y:
                cloth_vertices[:, 1] = cloth_vertices[:, 1] + self.shift_y
            if garment_name in self.initial_vertices:
                cloth_vertices = np.array(self.initial_vertices[garment_name], copy=True)

            panel_labels = self._load_panel_labels_for_paths(paths)
            cloth_reference_labels, body_parts = assign.panel_assignment(
                cloth_seg_dict, cloth_vertices, cloth_indices, wp.transform(cloth_pos, cloth_rot),
                body_seg, body_vertices, body_indices, wp.transform(body_pos, body_rot),
                device=self.device,
                panel_init_labels=panel_labels,
                strategy='closest',
                merge_two_legs=True,
                smpl_body=paths.use_smpl_seg
            )
            combined_reference_labels.extend(cloth_reference_labels)
            body_parts_union.update(body_parts)

            particle_filter = []
            if config.enable_body_collision_filters:
                v_connectivity = self._build_vert_connectivity(cloth_vertices, cloth_indices)
                face_filter_0 = assign.assign_face_filter_points(
                    cloth_reference_labels,
                    ['left_arm', 'right_arm', 'arms'],
                    filter_id=0,
                    vert_connectivity=v_connectivity,
                )
                particle_filter = assign.assign_face_filter_points(
                    cloth_reference_labels,
                    ['left_leg', 'right_leg', 'legs'],
                    filter_id=1,
                    vert_connectivity=v_connectivity,
                    current_vertex_filter=face_filter_0
                )
            combined_particle_filter.extend(particle_filter)

            orig_lens_dict = None
            if paths.g_orig_edge_len.exists():
                with open(paths.g_orig_edge_len, 'rb') as file:
                    orig_lens_dict = pickle.load(file)
                if start_vertex:
                    orig_lens_dict = {
                        (i + start_vertex, j + start_vertex): length
                        for (i, j), length in orig_lens_dict.items()
                    }

            builder.add_cloth_mesh_sewing_spring(
                pos=cloth_pos,
                rot=cloth_rot,
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
                vertices=cloth_vertices,
                indices=cloth_indices,
                resolution_scale=config.resolution_scale,
                orig_lens=orig_lens_dict,
                stitching_vertices=stitching_vertices,
                density=config.garment_density,
                edge_ke=config.garment_edge_ke,
                edge_kd=config.garment_edge_kd,
                tri_ke=config.garment_tri_ke,
                tri_ka=config.garment_tri_ka,
                tri_kd=config.garment_tri_kd,
                tri_drag=config.garment_tri_drag,
                tri_lift=config.garment_tri_lift,
                radius=config.garment_radius,
                add_springs=True,
                spring_ke=config.spring_ke,
                spring_kd=config.spring_kd,
            )

            end_vertex = len(builder.particle_q)
            self.garment_infos.append({
                'name': garment_name,
                'paths': paths,
                'start': start_vertex,
                'end': end_vertex,
                'faces': cloth_faces,
                'vertices': np.array(cloth_vertices, copy=True),
                'reference_labels': list(cloth_reference_labels),
                'panel_names': list(panel_labels.keys()),
            })

            if config.enable_attachment_constraint:
                self._add_attachment_labels_for_paths(
                    builder, config, body_seg, paths, start_vertex)

        self._setup_waistband_side_constraint(config)
        self._setup_lower_body_clearance_constraint(config, body_seg, body_vertices)
        self._setup_untucked_waistband_fit_constraint(config, body_seg, body_vertices)
        self._setup_limb_end_stop_constraints(config, body_seg, body_vertices)

        face_filters = []
        if config.enable_body_collision_filters:
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, ['main_body'], smpl_body=self.paths.use_smpl_seg))
            face_filters.append(assign.create_face_filter(
                body_vertices, body_indices, body_seg, ['left_arm', 'right_arm', 'arms'], smpl_body=self.paths.use_smpl_seg))

        self.body_shape_index = 0
        builder.add_shape_mesh(
            body=-1,
            mesh=self.body_mesh,
            pos=body_pos,
            rot=body_rot,
            scale=wp.vec3(1.0, 1.0, 1.0),
            thickness=config.body_thickness,
            mu=config.body_friction,
            face_filters=face_filters if face_filters else [[]],
            model_particle_filter_ids=combined_particle_filter,
        )

        for part in body_parts_union:
            v_sub_indices = np.asarray(body_parts_union[part], dtype=np.int32)
            part_v, part_inds = assign.extract_submesh(body_vertices, body_indices, v_sub_indices)
            part_inds = np.asarray(part_inds, dtype=np.int32)
            self.part_inds_dict[part] = part_inds
            self.part_v_sequence[part] = self.body_sequence[:, v_sub_indices, :] if has_body_sequence else body_vertices[v_sub_indices, :]
            self.mesh_dict[part] = wp.sim.Mesh(part_v, part_inds)
            builder.add_cloth_reference_shape_mesh(
                mesh=self.mesh_dict[part],
                name=part,
                pos=body_pos,
                rot=body_rot,
                scale=(1.0, 1.0, 1.0)
            )

        self.body_parts_names2index = builder.add_cloth_reference_labels(
            combined_reference_labels,
            [
                ['left_arm', 'body'],
                ['right_arm', 'body'],
                ['left_leg', 'right_leg'],
                ['left_arm', 'left_leg'],
                ['right_arm', 'left_leg'],
                ['left_arm', 'right_leg'],
                ['right_arm', 'right_leg'],
                ['left_arm', 'legs'],
                ['right_arm', 'legs'],
            ]
        )

        self.model: wp.sim.Model = builder.finalize(device=self.device)

    def _load_panel_labels_for_paths(self, paths):
        pattern = BasicPattern(paths.g_specs)
        labels = {}
        for name, panel in pattern.pattern['panels'].items():
            labels[name] = panel['label'] if 'label' in panel else ''
        return labels

    def _add_attachment_labels_for_paths(self, builder, config, body_seg, paths, vertex_offset):
        with open(paths.g_vert_labels, 'r') as f:
            vertex_labels = yaml.load(f, Loader=yaml.SafeLoader)

        for i, attach_label in enumerate(config.attachment_labels):
            if attach_label not in vertex_labels or len(vertex_labels[attach_label]) == 0:
                continue

            constraint_verts = [vertex_offset + idx for idx in vertex_labels[attach_label]]
            point_list, normal_list = self._attachment_motion(attach_label, body_seg)
            if point_list is None:
                print(f'{paths.in_tag}::WARNING::Requested attachment label {attach_label} '
                      'is not supported. Skipped')
                continue

            update_start = len(builder.attachment_indices)
            builder.add_attachment(
                constraint_verts,
                wp.vec3(*point_list[0]),
                wp.vec3(*normal_list[0]),
                stiffness=config.attachment_stiffness[i],
                damping=config.attachment_damping[i]
            )
            update_end = len(builder.attachment_indices)
            self.attachment_updates.append({
                'start': update_start,
                'end': update_end,
                'points': point_list,
                'normals': normal_list,
            })
            print(f'Using attachment for {paths.in_tag}:{attach_label} with {len(constraint_verts)} vertices')

    def _setup_waistband_side_constraint(self, config):
        if (not config.enable_waistband_side_constraint or
                self.waistband_side_mode not in ('inside', 'outside')):
            return
        if (self.waistband_side_mode == 'outside' and
                config.disable_untucked_upper_hem_constraint):
            print(f'{self.name}::untucked upper hem side constraint disabled.')
            return

        upper_info = None
        lower_info = None
        for info in self.garment_infos:
            role = getattr(info['paths'], 'out_folder_tag', '')
            if role == 'Upper':
                upper_info = info
            elif role == 'Lower':
                lower_info = info

        if upper_info is None or lower_info is None:
            print(f'{self.name}::WARNING::waistband side constraint needs Upper and Lower garments. Skipped.')
            return

        try:
            with open(upper_info['paths'].g_vert_labels, 'r') as f:
                upper_labels = yaml.load(f, Loader=yaml.SafeLoader) or {}
            with open(lower_info['paths'].g_vert_labels, 'r') as f:
                lower_labels = yaml.load(f, Loader=yaml.SafeLoader) or {}
        except FileNotFoundError:
            print(f'{self.name}::WARNING::vertex labels missing for waistband side constraint. Skipped.')
            return

        upper_local = upper_labels.get(config.upper_hem_label, [])
        lower_local = lower_labels.get(config.waistband_label, [])
        if not upper_local:
            print(
                f'{self.name}::WARNING::waistband side constraint label missing '
                f'({config.upper_hem_label}=0). Skipped.'
            )
            return
        if not lower_local and config.waistband_side_lower_scope != 'garment':
            print(
                f'{self.name}::WARNING::waistband side constraint labels missing '
                f'({config.upper_hem_label}={len(upper_local)}, '
                f'{config.waistband_label}={len(lower_local)}). Skipped.'
            )
            return

        upper_indices = upper_local
        if self.waistband_side_mode == 'inside':
            overlap_hem = self._filter_upper_tuck_overlap(
                upper_info, upper_local, lower_info, lower_local, config)
            if len(overlap_hem) == 0:
                print(
                    f'{self.name}::WARNING::waistband side constraint skipped; '
                    'upper hem does not overlap the waistband/lower garment.'
                )
                return
            upper_indices = self._expand_upper_tuck_band(upper_info, overlap_hem, config)
            upper_indices = self._filter_upper_tuck_band_overlap(
                upper_info, upper_indices, lower_info, lower_local, config)
            if len(upper_indices) == 0:
                print(
                    f'{self.name}::WARNING::waistband side constraint skipped; '
                    'no upper vertices remain in the overlapping tuck band.'
                )
                return

        upper_global = np.asarray(
            [upper_info['start'] + idx for idx in upper_indices], dtype=np.int32)
        lower_indices = self._select_lower_constraint_indices(
            lower_info, lower_local, config.waistband_side_lower_scope)
        lower_global = np.asarray(
            [lower_info['start'] + idx for idx in lower_indices], dtype=np.int32)

        self.waistband_side_constraint = {
            'upper_indices': wp.array(upper_global, dtype=wp.int32, device=self.device),
            'lower_indices': wp.array(lower_global, dtype=wp.int32, device=self.device),
            'upper_count': len(upper_global),
            'lower_count': len(lower_global),
            'mode_id': 1 if self.waistband_side_mode == 'inside' else 2,
            'mode': self.waistband_side_mode,
        }
        print(
            f'Using waistband side constraint ({self.waistband_side_mode}) with '
            f'{len(upper_indices)} upper vertices and {len(lower_indices)} lower vertices'
        )

    def _filter_upper_tuck_overlap(self, upper_info, upper_indices, lower_info, lower_local, config):
        upper_vertices = upper_info.get('vertices')
        lower_vertices = lower_info.get('vertices')
        if upper_vertices is None or lower_vertices is None or len(upper_indices) == 0 or len(lower_local) == 0:
            return upper_indices

        lower_y = lower_vertices[np.asarray(lower_local, dtype=np.int32), 1]
        overlap_limit_y = float(np.max(lower_y) - config.waistband_overlap_margin)
        selected = [
            int(idx) for idx in upper_indices
            if upper_vertices[int(idx), 1] <= overlap_limit_y
        ]
        print(
            f'{self.name}::waistband overlap filter kept '
            f'{len(selected)}/{len(upper_indices)} upper hem vertices'
        )
        return selected

    def _filter_upper_tuck_band_overlap(self, upper_info, upper_indices, lower_info, lower_local, config):
        upper_vertices = upper_info.get('vertices')
        lower_vertices = lower_info.get('vertices')
        if upper_vertices is None or lower_vertices is None or len(upper_indices) == 0 or len(lower_local) == 0:
            return upper_indices

        lower_y = lower_vertices[np.asarray(lower_local, dtype=np.int32), 1]
        max_overlap_y = float(np.min(lower_y) + config.waistband_overlap_band_margin)
        selected = [
            int(idx) for idx in upper_indices
            if upper_vertices[int(idx), 1] <= max_overlap_y
        ]
        print(
            f'{self.name}::waistband tuck band overlap filter kept '
            f'{len(selected)}/{len(upper_indices)} upper vertices'
        )
        return selected

    def _setup_lower_body_clearance_constraint(self, config, body_seg, body_vertices):
        if (not config.enable_lower_body_clearance_constraint or
                self.waistband_side_mode != 'inside'):
            return

        lower_info = None
        for info in self.garment_infos:
            role = getattr(info['paths'], 'out_folder_tag', '')
            if role == 'Lower':
                lower_info = info
                break

        if lower_info is None:
            return

        try:
            with open(lower_info['paths'].g_vert_labels, 'r') as f:
                lower_labels = yaml.load(f, Loader=yaml.SafeLoader) or {}
        except FileNotFoundError:
            print(f'{self.name}::WARNING::vertex labels missing for lower body clearance constraint. Skipped.')
            return

        lower_local = lower_labels.get(config.waistband_label, [])
        if not lower_local and config.lower_body_clearance_scope != 'all':
            print(
                f'{self.name}::WARNING::lower body clearance label missing '
                f'({config.waistband_label}=0). Skipped.'
            )
            return

        lower_band = self._select_lower_constraint_indices(
            lower_info, lower_local, config.lower_body_clearance_scope)
        body_local = self._select_body_clearance_indices(body_seg, body_vertices, lower_info, lower_band, config)
        if len(lower_band) == 0 or len(body_local) == 0:
            print(
                f'{self.name}::WARNING::lower body clearance constraint has empty indices '
                f'(lower={len(lower_band)}, body={len(body_local)}). Skipped.'
            )
            return

        lower_global = np.asarray(
            [lower_info['start'] + idx for idx in lower_band], dtype=np.int32)
        body_indices = np.asarray(body_local, dtype=np.int32)
        self.lower_body_clearance_constraint = {
            'lower_indices': wp.array(lower_global, dtype=wp.int32, device=self.device),
            'body_indices': wp.array(body_indices, dtype=wp.int32, device=self.device),
            'lower_count': len(lower_global),
            'body_count': len(body_indices),
        }
        print(
            f'Using lower body clearance constraint with '
            f'{len(lower_global)} lower vertices and {len(body_indices)} body vertices'
        )

    def _setup_untucked_waistband_fit_constraint(self, config, body_seg, body_vertices):
        if (not config.enable_untucked_waistband_fit_constraint or
                self.waistband_side_mode != 'outside'):
            return

        lower_info = None
        for info in self.garment_infos:
            role = getattr(info['paths'], 'out_folder_tag', '')
            if role == 'Lower':
                lower_info = info
                break

        if lower_info is None:
            return

        try:
            with open(lower_info['paths'].g_vert_labels, 'r') as f:
                lower_labels = yaml.load(f, Loader=yaml.SafeLoader) or {}
        except FileNotFoundError:
            print(f'{self.name}::WARNING::vertex labels missing for untucked waistband fit constraint. Skipped.')
            return

        lower_local = lower_labels.get(config.waistband_label, [])
        if not lower_local and config.untucked_waistband_fit_scope != 'all':
            print(
                f'{self.name}::WARNING::untucked waistband fit label missing '
                f'({config.waistband_label}=0). Skipped.'
            )
            return

        lower_band = self._select_untucked_waistband_fit_indices(lower_info, lower_local, config)
        body_local = self._select_untucked_waistband_body_indices(
            body_seg, body_vertices, lower_info, lower_band, config)
        if len(lower_band) == 0 or len(body_local) == 0:
            print(
                f'{self.name}::WARNING::untucked waistband fit constraint has empty indices '
                f'(lower={len(lower_band)}, body={len(body_local)}). Skipped.'
            )
            return

        lower_global = np.asarray(
            [lower_info['start'] + idx for idx in lower_band], dtype=np.int32)
        body_indices = np.asarray(body_local, dtype=np.int32)
        self.untucked_waistband_fit_constraint = {
            'lower_indices': wp.array(lower_global, dtype=wp.int32, device=self.device),
            'body_indices': wp.array(body_indices, dtype=wp.int32, device=self.device),
            'lower_count': len(lower_global),
            'body_count': len(body_indices),
        }
        print(
            f'Using untucked waistband fit constraint with '
            f'{len(lower_global)} lower vertices and {len(body_indices)} body vertices'
        )

    def _setup_limb_end_stop_constraints(self, config, body_seg, body_vertices):
        if not config.enable_limb_end_stop_constraint:
            return

        specs = [
            {
                'name': 'left_sleeve_wrist_stop',
                'cloth_labels': ('left_arm', 'arms'),
                'proximal_keys': ('leftForeArm',),
                'distal_keys': ('leftHand', 'leftHandIndex1'),
                'requires_panel_prefix': None,
            },
            {
                'name': 'right_sleeve_wrist_stop',
                'cloth_labels': ('right_arm', 'arms'),
                'proximal_keys': ('rightForeArm',),
                'distal_keys': ('rightHand', 'rightHandIndex1'),
                'requires_panel_prefix': None,
            },
            {
                'name': 'left_pant_ankle_stop',
                'cloth_labels': ('left_leg', 'legs'),
                'proximal_keys': ('leftLeg',),
                'distal_keys': ('leftFoot', 'leftToeBase'),
                'requires_panel_prefix': ('pant_',),
            },
            {
                'name': 'right_pant_ankle_stop',
                'cloth_labels': ('right_leg', 'legs'),
                'proximal_keys': ('rightLeg',),
                'distal_keys': ('rightFoot', 'rightToeBase'),
                'requires_panel_prefix': ('pant_',),
            },
        ]

        for spec in specs:
            proximal = self._body_indices_for_keys(body_seg, body_vertices, spec['proximal_keys'])
            distal = self._body_indices_for_keys(body_seg, body_vertices, spec['distal_keys'])
            if len(proximal) == 0 or len(distal) == 0:
                continue

            for info in self.garment_infos:
                required_prefixes = spec.get('requires_panel_prefix')
                if required_prefixes is not None and not self._garment_has_panel_prefix(info, required_prefixes):
                    continue

                local_indices = self._select_limb_end_stop_vertices(
                    info,
                    spec['cloth_labels'],
                    proximal,
                    distal,
                    body_vertices,
                    config,
                )
                if len(local_indices) == 0:
                    continue

                global_indices = np.asarray(
                    [info['start'] + idx for idx in local_indices], dtype=np.int32)
                constraint = {
                    'name': f"{info['name']}:{spec['name']}",
                    'cloth_indices': wp.array(global_indices, dtype=wp.int32, device=self.device),
                    'cloth_count': len(global_indices),
                    'proximal_indices': wp.array(proximal, dtype=wp.int32, device=self.device),
                    'distal_indices': wp.array(distal, dtype=wp.int32, device=self.device),
                    'proximal_count': len(proximal),
                    'distal_count': len(distal),
                }
                self.limb_end_stop_constraints.append(constraint)
                print(
                    f"Using limb end stop constraint {constraint['name']} with "
                    f"{len(global_indices)} cloth vertices"
                )

    def _garment_has_panel_prefix(self, info, prefixes):
        return any(
            panel_name.startswith(prefix)
            for panel_name in info.get('panel_names', [])
            for prefix in prefixes
        )

    def _body_indices_for_keys(self, body_seg, body_vertices, keys):
        indices = []
        for key in keys:
            indices.extend(body_seg.get(key, []))
        if not indices:
            return np.asarray([], dtype=np.int32)

        body_count = len(body_vertices)
        indices = np.asarray(sorted(set(int(idx) for idx in indices)), dtype=np.int32)
        return indices[(indices >= 0) & (indices < body_count)]

    def _select_limb_end_stop_vertices(self, info, cloth_labels, proximal, distal, body_vertices, config):
        vertices = info.get('vertices')
        reference_labels = info.get('reference_labels') or []
        if vertices is None or len(reference_labels) != len(vertices):
            return []

        candidate_labels = {str(label) for label in cloth_labels}
        candidates = [
            idx for idx, label in enumerate(reference_labels)
            if str(label) in candidate_labels
        ]
        if not candidates:
            return []

        proximal_center = np.mean(body_vertices[proximal], axis=0)
        distal_center = np.mean(body_vertices[distal], axis=0)
        axis = distal_center - proximal_center
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-8:
            return []
        axis = axis / axis_norm

        candidates = np.asarray(candidates, dtype=np.int32)
        projections = (vertices[candidates] - proximal_center) @ axis
        max_projection = float(np.max(projections))
        min_projection = float(np.min(projections))
        span = max(max_projection - min_projection, 1e-6)
        fraction = float(np.clip(config.limb_end_stop_candidate_fraction, 0.01, 1.0))
        threshold = max_projection - span * fraction
        selected = candidates[projections >= threshold]
        return selected.tolist()

    def _select_lower_constraint_indices(self, lower_info, lower_local, scope):
        vertices = lower_info.get('vertices')
        if scope == 'all' and vertices is not None:
            return list(range(len(vertices)))
        if scope == 'waistband':
            return lower_local
        return self._expand_lower_vertical_band(
            lower_info, lower_local, float(max(self.config.lower_body_clearance_band_height, 0.0)))

    def _select_untucked_waistband_fit_indices(self, lower_info, lower_local, config):
        return lower_local

    def _select_untucked_waistband_body_indices(self, body_seg, body_vertices, lower_info, lower_band, config):
        candidate_keys = (
            'spine',
            'spine1',
            'spine2',
            'hips',
            'leftUpLeg',
            'rightUpLeg',
            'body',
            'main_body',
        )
        candidates = []
        for key in candidate_keys:
            if key in body_seg:
                candidates.extend(body_seg[key])

        if not candidates:
            return self._select_body_clearance_indices(body_seg, body_vertices, lower_info, lower_band, config)

        body_count = len(body_vertices)
        candidates = np.asarray(sorted(set(int(idx) for idx in candidates)), dtype=np.int32)
        candidates = candidates[(candidates >= 0) & (candidates < body_count)]
        if len(candidates) == 0:
            return []

        lower_vertices = lower_info.get('vertices')
        if lower_vertices is None or len(lower_band) == 0:
            return candidates.tolist()

        band_vertices = lower_vertices[np.asarray(lower_band, dtype=np.int32)]
        band_min_y = float(np.min(band_vertices[:, 1]))
        band_max_y = float(np.max(band_vertices[:, 1]))
        vertical_padding = max(
            2.0,
            float(max(config.untucked_waistband_fit_body_y_window, 0.0)) * 1.25,
        )
        body_y = body_vertices[candidates, 1]
        filtered = candidates[
            (body_y >= band_min_y - vertical_padding) &
            (body_y <= band_max_y + vertical_padding)
        ]
        if len(filtered) >= 8:
            return filtered.tolist()
        return candidates.tolist()

    def _expand_lower_vertical_band(self, lower_info, lower_local, band_height):
        """Use the waistband plus a connected vertical band below it."""
        vertices = lower_info.get('vertices')
        faces = lower_info.get('faces')
        if vertices is None or faces is None or len(lower_local) == 0:
            return lower_local

        band_height = float(max(band_height, 0.0))
        if band_height <= 0.0:
            return lower_local

        local_count = len(vertices)
        adjacency = [[] for _ in range(local_count)]
        for face in np.asarray(faces, dtype=np.int32):
            if len(face) != 3:
                continue
            a, b, c = [int(v) for v in face]
            adjacency[a].extend((b, c))
            adjacency[b].extend((a, c))
            adjacency[c].extend((a, b))

        waist_set = {int(idx) for idx in lower_local}
        waist_y = vertices[list(waist_set), 1]
        min_y = float(np.min(waist_y) - band_height)
        max_y = float(np.max(waist_y) + 0.5)

        selected = set(waist_set)
        queue = list(waist_set)
        while queue:
            idx = queue.pop(0)
            for neighbor in adjacency[idx]:
                if neighbor in selected:
                    continue
                neighbor_y = vertices[neighbor, 1]
                if neighbor_y < min_y or neighbor_y > max_y:
                    continue
                selected.add(neighbor)
                queue.append(neighbor)

        return sorted(selected)

    def _expand_lower_clearance_band(self, lower_info, lower_local, config):
        return self._expand_lower_vertical_band(
            lower_info,
            lower_local,
            float(max(config.lower_body_clearance_band_height, 0.0)),
        )

    def _select_body_clearance_indices(self, body_seg, body_vertices, lower_info, lower_band, config):
        candidate_keys = (
            'hips',
            'spine',
            'spine1',
            'leftUpLeg',
            'rightUpLeg',
            'leftLeg',
            'rightLeg',
            'legs',
            'body',
            'main_body',
        )
        candidates = []
        for key in candidate_keys:
            if key in body_seg:
                candidates.extend(body_seg[key])

        if not candidates:
            return []

        body_count = len(body_vertices)
        candidates = np.asarray(sorted(set(int(idx) for idx in candidates)), dtype=np.int32)
        candidates = candidates[(candidates >= 0) & (candidates < body_count)]
        if len(candidates) == 0:
            return []

        lower_vertices = lower_info.get('vertices')
        if lower_vertices is None or len(lower_band) == 0:
            return candidates.tolist()

        band_vertices = lower_vertices[np.asarray(lower_band, dtype=np.int32)]
        band_min_y = float(np.min(band_vertices[:, 1]))
        band_max_y = float(np.max(band_vertices[:, 1]))
        vertical_padding = max(3.0, float(config.lower_body_clearance_band_height) * 1.5)
        body_y = body_vertices[candidates, 1]
        filtered = candidates[
            (body_y >= band_min_y - vertical_padding) &
            (body_y <= band_max_y + vertical_padding)
        ]
        if len(filtered) >= 8:
            return filtered.tolist()
        return candidates.tolist()

    def _expand_upper_tuck_band(self, upper_info, upper_local, config):
        """Use a short connected band above the hem for tucked-in constraints."""
        vertices = upper_info.get('vertices')
        faces = upper_info.get('faces')
        if vertices is None or faces is None or len(upper_local) == 0:
            return upper_local

        local_count = len(vertices)
        adjacency = [[] for _ in range(local_count)]
        for face in np.asarray(faces, dtype=np.int32):
            if len(face) != 3:
                continue
            a, b, c = [int(v) for v in face]
            adjacency[a].extend((b, c))
            adjacency[b].extend((a, c))
            adjacency[c].extend((a, b))

        hem_set = {int(idx) for idx in upper_local}
        hem_y = vertices[list(hem_set), 1]
        max_y = float(np.max(hem_y) + config.upper_tuck_band_height)

        selected = set(hem_set)
        queue = list(hem_set)
        while queue:
            idx = queue.pop(0)
            for neighbor in adjacency[idx]:
                if neighbor in selected:
                    continue
                if vertices[neighbor, 1] > max_y:
                    continue
                selected.add(neighbor)
                queue.append(neighbor)

        return sorted(selected)

    def _attachment_motion(self, attach_label, body_seg):
        if self.body_sequence is None:
            return None, None

        if attach_label == 'lower_interface':
            waist_indices = body_seg['spine']
            point_list, normal_list = [], []
            for body_vertices in self.body_sequence:
                waist_points = body_vertices[waist_indices]
                center = waist_points.mean(axis=0)
                center[1] = center[1] - 5
                point_list.append(center)
                normal_list.append(np.array([0, 1, 0]))
            return point_list, normal_list

        shoulder_key = None
        fallback = np.array([1, 0, 0])
        if attach_label == 'right_collar':
            shoulder_key = 'rightShoulder'
            fallback = np.array([1, 0, 0])
        elif attach_label == 'left_collar':
            shoulder_key = 'leftShoulder'
            fallback = np.array([-1, 0, 0])
        if shoulder_key is not None:
            shoulder_indices = body_seg.get(shoulder_key, None)
            neck_indices = body_seg.get('spine2', None)
            if shoulder_indices is None or neck_indices is None:
                return None, None
            point_list, normal_list = [], []
            scale = 0.6
            for body_vertices in self.body_sequence:
                collar_points = body_vertices[shoulder_indices]
                neck_points = body_vertices[neck_indices]
                center = neck_points.mean(axis=0)
                dists = np.linalg.norm(collar_points - center, axis=1)
                attach_point = collar_points[np.argmin(dists)]
                interp_point = center + scale * (attach_point - center)
                attach_norm = center - attach_point
                norm = np.linalg.norm(attach_norm)
                attach_norm = attach_norm / norm if norm > 1e-8 else fallback
                point_list.append(interp_point)
                normal_list.append(attach_norm)
            return point_list, normal_list

        return None, None

    def update_attachment(self, frame):
        if not self.attachment_updates:
            return
        for update in self.attachment_updates:
            if frame < 0 or frame >= len(update['points']):
                continue
            start, end = update['start'], update['end']
            if end <= start:
                continue
            point = wp.vec3(*update['points'][frame])
            normal = wp.vec3(*update['normals'][frame])
            self.model.attachment_point[start:end].assign(
                wp.array([point] * (end - start), dtype=wp.vec3, device=self.device))
            self.model.attachment_norm[start:end].assign(
                wp.array([normal] * (end - start), dtype=wp.vec3, device=self.device))

    def _apply_lower_body_clearance_constraint(self, frame):
        constraint = self.lower_body_clearance_constraint
        if constraint is None:
            return

        max_frames = int(self.config.lower_body_clearance_frames)
        if max_frames > 0 and frame >= max_frames:
            return

        lower_count = constraint['lower_count']
        body_count = constraint['body_count']
        if lower_count == 0 or body_count == 0:
            return

        wp.launch(
            kernel=apply_lower_body_clearance_constraint_kernel,
            dim=lower_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                constraint['lower_indices'],
                self.body_mesh.mesh.points,
                constraint['body_indices'],
                body_count,
                float(self.config.lower_body_clearance_margin),
                float(np.clip(self.config.lower_body_clearance_stiffness, 0.0, 1.0)),
                float(np.clip(self.config.lower_body_clearance_damping, 0.0, 1.0)),
                float(max(self.config.lower_body_clearance_max_step, 0.0)),
                float(max(self.config.lower_body_clearance_body_y_window, 0.0)),
            ],
            device=self.device,
        )

    def _apply_untucked_waistband_fit_constraint(self, frame):
        constraint = self.untucked_waistband_fit_constraint
        if constraint is None:
            return

        max_frames = int(self.config.untucked_waistband_fit_frames)
        if max_frames > 0 and frame >= max_frames:
            return

        lower_count = constraint['lower_count']
        body_count = constraint['body_count']
        if lower_count == 0 or body_count == 0:
            return

        wp.launch(
            kernel=apply_untucked_waistband_fit_constraint_kernel,
            dim=lower_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                constraint['lower_indices'],
                self.body_mesh.mesh.points,
                constraint['body_indices'],
                body_count,
                float(self.config.untucked_waistband_fit_margin),
                float(np.clip(self.config.untucked_waistband_fit_stiffness, 0.0, 1.0)),
                float(np.clip(self.config.untucked_waistband_fit_damping, 0.0, 1.0)),
                float(max(self.config.untucked_waistband_fit_max_step, 0.0)),
                float(max(self.config.untucked_waistband_fit_body_y_window, 0.0)),
            ],
            device=self.device,
        )

    def _apply_limb_end_stop_constraints(self, frame):
        if not self.limb_end_stop_constraints:
            return

        for constraint in self.limb_end_stop_constraints:
            cloth_count = constraint['cloth_count']
            if cloth_count == 0:
                continue

            wp.launch(
                kernel=apply_limb_end_stop_constraint_kernel,
                dim=cloth_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    constraint['cloth_indices'],
                    self.body_mesh.mesh.points,
                    constraint['proximal_indices'],
                    constraint['distal_indices'],
                    constraint['proximal_count'],
                    constraint['distal_count'],
                    float(np.clip(self.config.limb_end_stop_stiffness, 0.0, 1.0)),
                    float(np.clip(self.config.limb_end_stop_damping, 0.0, 1.0)),
                    float(max(self.config.limb_end_stop_max_step, 0.0)),
                    float(self.config.limb_end_stop_margin),
                ],
                device=self.device,
            )

    def _apply_waistband_side_constraint(self, frame):
        constraint = self.waistband_side_constraint
        if constraint is None:
            return

        max_frames = int(self.config.waistband_constraint_frames)
        if max_frames > 0 and frame >= max_frames:
            return

        upper_count = constraint['upper_count']
        lower_count = constraint['lower_count']
        if upper_count == 0 or lower_count == 0:
            return

        wp.launch(
            kernel=apply_waistband_side_constraint_kernel,
            dim=upper_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                constraint['upper_indices'],
                constraint['lower_indices'],
                lower_count,
                constraint['mode_id'],
                float(self.config.waistband_side_margin),
                float(np.clip(self.config.waistband_constraint_stiffness, 0.0, 1.0)),
                float(np.clip(self.config.waistband_vertical_stiffness, 0.0, 1.0)),
                float(self.config.waistband_tuck_depth),
                float(np.clip(self.config.waistband_constraint_damping, 0.0, 1.0)),
                float(max(self.config.waistband_side_lower_y_window, 0.0)),
            ],
            device=self.device,
        )

    def save_frame(self, save_v_norms=False):
        all_verts = self.current_verts
        for info in self.garment_infos:
            paths = info['paths']
            v_cloth_sim = all_verts[info['start']:info['end']]
            with open(paths.g_box_mesh, 'r') as obj_file:
                lines = obj_file.readlines()
            with open(paths.g_sim, 'w') as obj_file:
                v_idx = 0
                for line in lines:
                    if line.startswith('v '):
                        new_vertex = v_cloth_sim[v_idx]
                        obj_file.write(f'v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n')
                        v_idx += 1
                    else:
                        obj_file.write(line)
