from pathlib import Path
import os
import re
import time
import yaml
import json
import shutil 
import string
import random
import trimesh
import numpy as np
from copy import deepcopy
from typing import Optional

# Custom 
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
import pygarment as pyg
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.simulation import run_sim, run_sim_new
import pygarment.data_config as data_config
from pygarment.meshgen.sim_config import PathCofig, PathCofigNew

# SMPL-X config
from pygarment.paths_config import smplx_model_path_str

SMPLX_MODEL_PATH = smplx_model_path_str("female")
POSE_VECS_PATH = './smplx'

verbose = False

def _id_generator(size=10, chars=string.ascii_uppercase + string.digits):
        """Generate a random string of a given size, see
        https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
        """
        return ''.join(random.choices(chars, k=size))

class GUIPattern:
    def __init__(self) -> None:
        # Unique id to distiguish tab sessions correctly
        self.id = _id_generator(20)

        # Paths setup
        self.save_path_root = Path.cwd() / 'tmp_gui' / 'downloads'  
        self.tmp_path_root = Path.cwd() / 'tmp_gui' / 'display'
        self.save_path = self.save_path_root / self.id
        self.svg_filename = None
        self.saved_garment_archive = ''
        self.saved_garment_folder = ''
        self.tmp_path = self.tmp_path_root / self.id 
        self.paths_3d = None

        # create paths
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        self.body_params = None
        self.design_params = {}
        self.design_sampler = pyg.DesignSampler()
        self.sew_pattern = None

        self.body_file = None
        self.design_file = None
        self._load_body_file(
            Path.cwd() / 'assets/bodies/mean_all.yaml'
        )
        self.default_body_params = deepcopy(self.body_params)
        self._load_design_file(
            Path.cwd() / 'assets/design_params/default.yaml'
        )

        # Status
        self.is_self_intersecting = False
        self.is_in_3D = False

        self.reload_garment()

    def release(self):
        """Clean up tmp files after the session"""
        self.clear_previous_download()
        shutil.rmtree(self.save_path)
        shutil.rmtree(self.tmp_path)

    def _load_body_file(self, path):
        self.body_file = path
        self.body_params = BodyParameters(path)

    def _load_design_file(self, path):
        self.design_file = path

        # Create values
        with open(path, 'r') as f:
            des = yaml.safe_load(f)['design']

        self.design_params.update(des)
        if 'left' in self.design_params and not self.design_params['left']['enable_asym']['v']:
            self.sync_left()

        # Update param sampler
        self.design_sampler.load(path)

    def svg_path(self):
        return self.tmp_path / self.svg_filename

    def set_new_design(self, design):
        self._nested_sync(design, self.design_params)

    def set_new_body_params(self, body_params):
        self.body_params.load_from_dict(body_params)

    def sample_design(self, reload=True):
        """Random design parameters"""

        new_design = self.design_sampler.randomize()
        # NOTE: re-assign the values instead up overwriting them
        self._nested_sync(new_design, self.design_params)

        if 'left' in self.design_params and not self.design_params['left']['enable_asym']['v']:
            self.sync_left()

        if reload:
            self.reload_garment()

    def restore_design(self, reload=True):
        """Restore design values to match the current loaded file"""
        new_design = self.design_sampler.default()
        # re-assign the values instead up overwriting them
        self._nested_sync(new_design, self.design_params)
        
        if reload:
            self.reload_garment()

    def reload_garment(self):
        """Reload sewing pattern with current body and design parameters
        
            NOTE: loading a pattern might be lagging, execute only when needed!
        """
        self.sew_pattern = MetaGarment(
            'Configured_design', self.body_params, self.design_params)
        self.is_self_intersecting = self.sew_pattern.is_self_intersecting()
        self._view_serialize()

    @staticmethod
    def _nested_sync(s_from, s_to):
        if 'v' in s_to:
            s_to['v'] = s_from['v']
        else:
            for key in s_to:
                if key in s_from:
                    GUIPattern._nested_sync(s_from[key], s_to[key])

    def sync_left(self, with_check=False):
        """Synchronize left and right design parameters"""
        # Check if needed in the first place
        if with_check and self.design_params['left']['enable_asym']['v']:
            # Asymmetry enabled, the params should not syncronise 
            return  
        for k in self.design_params['left']:
            if k != 'enable_asym':
                # Use proper value assignment instead of deepcopy
                self._nested_sync(self.design_params[k], self.design_params['left'][k])

    def _view_serialize(self):
        """Save a sewing pattern svg representation to tmp folder be used
        for display"""

        # Get the flat representation
        pattern = self.sew_pattern.assembly()

        # Clear up the folder from previous version -- it's not needed any more
        self.clear_previous_svg()
        try:
            self.svg_filename = f'pattern_{time.time()}.svg'
            dwg = pattern.get_svg(self.tmp_path / self.svg_filename, 
                                  with_text=False, 
                                  view_ids=False,
                                  flat=False,
                                  margin=0
            )
            dwg.save()

            self.svg_bbox_size = pattern.svg_bbox_size
            self.svg_bbox = pattern.svg_bbox
        except pyg.EmptyPatternError:
            self.svg_filename = ''
    
    # Cleaning
    def clear_previous_svg(self):
        """Clear previous svg display file"""
        if self.svg_filename:
            (self.tmp_path / self.svg_filename).unlink()
            self.svg_filename = ''
    
    def clear_previous_download(self):
        """Clear previous download package display file"""
        if self.saved_garment_folder:
            shutil.rmtree(self.saved_garment_folder)
            self.saved_garment_folder = ''
        if self.saved_garment_archive:
            self.saved_garment_archive.unlink()
            self.saved_garment_archive = ''

    def clear_3d(self):
        if self.paths_3d is not None:
            shutil.rmtree(self.paths_3d.out_el)
            self.paths_3d = None

    # 3D
    def drape_3d(self):
        """Run the draping of the current frame"""

        # Config setup 
        props = data_config.Properties('./assets/Sim_props/mid_bending.yaml')   # TODOLOW Parameter?
        props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
        props.set_section_stats('render', render_time={})

        # Force the design to be fitted to mean body shape 
        # TODOLOW Support body shape estimation from measurements

        def_sew_pattern = MetaGarment(
            'Configured_design', self.default_body_params, self.design_params)

        # Save the pattern
        pattern_folder = self.save(False, save_pattern=def_sew_pattern)

        # Paths
        paths = PathCofig(
            in_element_path=pattern_folder, 
            out_path=self.save_path,
            in_name=def_sew_pattern.name,
            out_name=self.sew_pattern.name + '_3D',
            body_name='mean_all',  
            smpl_body=False,   # NOTE: depends on chosen body model
            add_timestamp=False
        )

        # Generate and save garment box mesh (if not existent)
        garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
        garment_box_mesh.load()
        garment_box_mesh.serialize(
            paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

        # TODOLOW Don't print progress to console with so many lines
        run_sim(
            garment_box_mesh.name, 
            props, 
            paths,
            save_v_norms=False,
            store_usd=False,  # NOTE: False for fast simulation!, 
            optimize_storage=False,
            verbose=False
        )

        # Convert to displayable element
        mesh = trimesh.load_mesh(paths.g_sim)

        # enable double-sided material for nice viewing
        pbr_material = mesh.visual.material.to_pbr()
        pbr_material.doubleSided = True
        mesh.visual.material = pbr_material
        # export
        mesh.export(paths.g_sim_glb)

        self.paths_3d = paths
        self.is_in_3D = True

        return paths.out_el, paths.g_sim_glb.name

    # ── Dynamic (multi-pose) draping ────────────────────────────────────
    @staticmethod
    def _split_pose_params(pose_params):
        global_orient = pose_params[:, :3]
        body_pose = pose_params[:, 3:66]
        left_hand_pose = pose_params[:, 66:111]
        right_hand_pose = pose_params[:, 111:156]
        jaw_pose = pose_params[:, 156:159]
        leye_pose = pose_params[:, 159:162]
        reye_pose = pose_params[:, 162:165]
        return global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose

    @staticmethod
    def _generate_smooth_shape_sequence(smpl_model, pose_params, beta_start, beta_end,
                                        threshold=0.01, max_depth=15, depth=0):
        import torch
        go, bp, lh, rh, jp, le, re = GUIPattern._split_pose_params(pose_params)
        v_start = smpl_model.forward(
            global_orient=go, body_pose=bp.reshape(1, -1, 3),
            left_hand_pose=lh.reshape(1, -1, 3), right_hand_pose=rh.reshape(1, -1, 3),
            betas=beta_start
        ).vertices[0].detach().cpu().numpy()
        v_end = smpl_model.forward(
            global_orient=go, body_pose=bp.reshape(1, -1, 3),
            left_hand_pose=lh.reshape(1, -1, 3), right_hand_pose=rh.reshape(1, -1, 3),
            betas=beta_end
        ).vertices[0].detach().cpu().numpy()

        with open('assets/bodies/smplx_body_segmentation.json', 'r') as f:
            seg = json.load(f)
        hand_idx = set(seg.get('leftHand', []) + seg.get('rightHand', []) +
                       seg.get('leftHandIndex1', []) + seg.get('rightHandIndex1', []))
        non_hand = np.setdiff1d(np.arange(v_start.shape[0]), list(hand_idx))
        mx = np.max(np.linalg.norm(v_end[non_hand] - v_start[non_hand], axis=1))
        if mx < threshold or depth >= max_depth:
            return [v_start, v_end]
        mid = (beta_start + beta_end) / 2.0
        first = GUIPattern._generate_smooth_shape_sequence(smpl_model, pose_params, beta_start, mid, threshold, max_depth, depth + 1)
        second = GUIPattern._generate_smooth_shape_sequence(smpl_model, pose_params, mid, beta_end, threshold, max_depth, depth + 1)
        return first[:-1] + second

    @staticmethod
    def _generate_smooth_pose_sequence(smpl_model, pose_start, pose_end, beta_params,
                                       threshold=0.01, max_depth=15, depth=0):
        import torch
        go_s, bp_s, lh_s, rh_s, *_ = GUIPattern._split_pose_params(pose_start)
        go_e, bp_e, lh_e, rh_e, *_ = GUIPattern._split_pose_params(pose_end)
        v_start = smpl_model.forward(global_orient=go_s, body_pose=bp_s, betas=beta_params).vertices[0].detach().cpu().numpy()
        v_end = smpl_model.forward(global_orient=go_e, body_pose=bp_e.reshape(1, -1, 3),
                                    left_hand_pose=lh_e.reshape(1, -1, 3), right_hand_pose=rh_e.reshape(1, -1, 3),
                                    betas=beta_params).vertices[0].detach().cpu().numpy()
        with open('assets/bodies/smplx_body_segmentation.json', 'r') as f:
            seg = json.load(f)
        hand_idx = set(seg.get('leftHand', []) + seg.get('rightHand', []) +
                       seg.get('leftHandIndex1', []) + seg.get('rightHandIndex1', []))
        non_hand = np.setdiff1d(np.arange(v_start.shape[0]), list(hand_idx))
        mx = np.max(np.linalg.norm(v_end[non_hand] - v_start[non_hand], axis=1))
        if mx < threshold or depth >= max_depth:
            return [v_start, v_end]
        mid = (pose_start + pose_end) / 2.0
        first = GUIPattern._generate_smooth_pose_sequence(smpl_model, pose_start, mid, beta_params, threshold, max_depth, depth + 1)
        second = GUIPattern._generate_smooth_pose_sequence(smpl_model, mid, pose_end, beta_params, threshold, max_depth, depth + 1)
        return first[:-1] + second

    @staticmethod
    def _align_smplx_body_y_axis(body_seq, smpl_seg, smplx_seg, default_verts):
        ref_labels = ['head', 'neck']
        smpl_idx = []
        smplx_idx = []
        for lbl in ref_labels:
            smpl_idx.extend(smpl_seg[lbl])
            smplx_idx.extend(smplx_seg[lbl])
        delta = np.mean(default_verts[:, 1][np.array(smpl_idx)]) - np.mean(body_seq[0][:, 1][np.array(smplx_idx)]) - 0.02
        aligned = body_seq.copy()
        aligned[:, :, 1] += delta
        return aligned

    @staticmethod
    def _batch_render_video(collected_frames, paths, body_faces, render_config, output_path, fps=15):
        """Batch-render collected vertex snapshots to a video file.
        
        collected_frames: list of (cloth_verts, body_verts) numpy arrays from simulation
        paths: PathCofigNew — used to load garment OBJ template (for faces/UVs/texture)
        body_faces: numpy array of body face indices
        render_config: render properties dict
        output_path: path to output mp4 file
        """
        import cv2
        import pyrender
        from PIL import Image
        import copy
        from scipy.spatial import cKDTree

        if not collected_frames:
            return

        # Load the garment mesh template (has faces, UVs, texture from the final save_frame)
        # NOTE: trimesh splits vertices at UV seams, so template may have MORE vertices
        # than simulation particles (current_verts). We build a mapping to handle this.
        garm_template = trimesh.load_mesh(str(paths.g_sim))
        garm_template.vertices = garm_template.vertices / 100  # scale to m

        # Build mapping from trimesh UV-split vertices to original OBJ vertex indices
        obj_verts = []
        with open(str(paths.g_sim), 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    obj_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        obj_verts = np.array(obj_verts) / 100  # same scale as template
        tree = cKDTree(obj_verts)
        _, vert_map = tree.query(garm_template.vertices)  # vert_map[i] = OBJ vertex index for trimesh vertex i

        # Prepare garment material
        material = garm_template.visual.material.to_pbr()
        material.baseColorFactor = [1., 1., 1., 1.]
        material.doubleSided = True
        white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
        white_back.paste(material.baseColorTexture)
        material.baseColorTexture = white_back.convert('RGB')
        garm_template.visual.material = material

        # Body material
        body_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.2, 0.17, 0.15, 1.0)
        )

        if render_config and 'resolution' in render_config:
            view_width, view_height = render_config['resolution']
        else:
            view_width, view_height = 768, 1024

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (view_width, view_height))

        renderer = None
        try:
            renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)

            for cloth_verts, body_verts in collected_frames:
                # Create garment mesh with updated vertices, using vert_map to handle UV splits
                garm_mesh = garm_template.copy()
                garm_mesh.vertices = (cloth_verts / 100)[vert_map]

                pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garm_mesh, smooth=True)

                # Create body mesh
                if body_verts is not None:
                    body_tm = trimesh.Trimesh(body_verts, body_faces)
                    body_tm.vertices = body_tm.vertices / 100
                    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_tm, material=body_material)
                else:
                    pyrender_body_mesh = pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(vertices=np.zeros((1, 3)), faces=np.zeros((0, 3), dtype=int)),
                        material=body_material
                    )

                # Build scene
                scene = pyrender.Scene(bg_color=(1., 1., 1., 0.), ambient_light=(0.3, 0.3, 0.3))
                scene.add(pyrender_garm_mesh)
                scene.add(pyrender_body_mesh)

                # Camera — front view (same as pythonrender.py)
                camera_location = render_config.get('front_camera_location', None)
                from pygarment.meshgen.render.pythonrender import create_camera, create_lights
                create_camera(pyrender, pyrender_body_mesh, scene, 'front', camera_location=camera_location)
                create_lights(scene, intensity=50.)

                color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

                # Convert RGBA to BGR for OpenCV
                frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
                writer.write(frame_bgr)

        finally:
            if renderer is not None:
                renderer.delete()
            writer.release()

        # Re-encode to H.264 for browser compatibility (mp4v is not supported by browsers)
        import subprocess
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-i', output_path, '-c:v', 'libx264',
             '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23', h264_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os.replace(h264_path, output_path)

        print(f"Batch-rendered {len(collected_frames)} frames to {output_path}")

    @staticmethod
    def _export_frame_glbs(collected_frames, paths, body_faces, out_dir):
        """Export each collected frame as a GLB file for 3D animation playback.

        Returns list of GLB filenames.
        """
        from scipy.spatial import cKDTree

        if not collected_frames:
            return []

        # Load garment template (faces, UVs, texture)
        garm_template = trimesh.load_mesh(str(paths.g_sim))

        # Build vert_map: trimesh UV-splits vertices, simulation has fewer
        obj_verts = []
        with open(str(paths.g_sim), 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    obj_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        obj_verts = np.array(obj_verts)
        tree = cKDTree(obj_verts)
        _, vert_map = tree.query(garm_template.vertices)

        # Make garment material double-sided
        pbr = garm_template.visual.material.to_pbr()
        pbr.doubleSided = True
        garm_template.visual.material = pbr

        out_dir = str(out_dir)
        # Pre-extract template data for thread workers
        garm_faces = garm_template.faces.copy()
        garm_visual = garm_template.visual

        def _export_one(args):
            i, cloth_verts, body_verts = args
            scene = trimesh.Scene()
            garm = trimesh.Trimesh(
                vertices=cloth_verts[vert_map],
                faces=garm_faces,
                visual=garm_visual.copy(),
                process=False
            )
            scene.add_geometry(garm, node_name='garment')
            if body_verts is not None:
                body = trimesh.Trimesh(body_verts, body_faces, process=False)
                # Shrink body slightly inward along vertex normals to avoid
                # z-fighting with the cloth mesh in the 3D viewer
                body.vertices -= body.vertex_normals * 0.5
                body.visual.face_colors = [80, 70, 65, 255]
                scene.add_geometry(body, node_name='body')
            fname = f'frame_{i:04d}.glb'
            scene.export(os.path.join(out_dir, fname))
            return i, fname

        from concurrent.futures import ThreadPoolExecutor
        tasks = [(i, cv, bv) for i, (cv, bv) in enumerate(collected_frames)]
        frame_glb_names = [None] * len(collected_frames)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            for idx, fname in pool.map(_export_one, tasks):
                frame_glb_names[idx] = fname

        print(f"Exported {len(frame_glb_names)} frame GLBs to {out_dir}")
        return frame_glb_names

    def drape_3d_dynamic(self, pose_idx, beta_params_list, gpu_id=0):
        """Run dynamic simulation with SMPL-X body sequence.
        
        Returns (out_dir, glb_filename, frames_dir) where frames_dir contains per-frame renders.
        """
        import torch
        import smplx

        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        # Load SMPL-X model
        smpl_model = smplx.SMPLX(SMPLX_MODEL_PATH, use_pca=False).to(device)
        body_faces = smpl_model.faces

        # Load pose
        pose_vec_file = os.path.join(POSE_VECS_PATH, f'{(pose_idx + 1):05d}_0.npz')
        pose_vec = np.load(pose_vec_file)

        # Beta params
        beta_params = torch.tensor(beta_params_list, dtype=torch.float32, device=device).unsqueeze(0)

        # Determine sleeve type from current design
        sleeve_type = 'sleeve'
        try:
            if self.design_params['sleeve']['sleeveless']['v']:
                sleeve_type = 'sleeveless'
        except (KeyError, TypeError):
            pass

        # Start pose
        pose_params_start = torch.zeros([1, 165], device=device).float()
        if sleeve_type == 'sleeve':
            pose_params_start[:, 16 * 3 + 2] = -0.6
            pose_params_start[:, 17 * 3 + 2] = 0.6
        # End pose from file
        global_orient = torch.from_numpy(pose_vec["global_orient"]).to(device)
        body_pose = torch.from_numpy(pose_vec["body_pose"]).reshape(1, -1).to(device)
        left_hand_pose = torch.from_numpy(pose_vec["left_hand_pose"]).reshape(1, -1).to(device)
        right_hand_pose = torch.from_numpy(pose_vec["right_hand_pose"]).reshape(1, -1).to(device)
        jaw_pose = torch.from_numpy(pose_vec["jaw_pose"]).reshape(1, -1).to(device)
        leye_pose = torch.from_numpy(pose_vec["leye_pose"]).reshape(1, -1).to(device)
        reye_pose = torch.from_numpy(pose_vec["reye_pose"]).reshape(1, -1).to(device)
        pose_params_end = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose,
                                     jaw_pose, leye_pose, reye_pose], dim=1).float()

        # Generate body sequences
        beta_zero = torch.zeros([1, 10], dtype=torch.float32, device=device)
        beta_zero[0, 0] = 1.5

        shape_seq = self._generate_smooth_shape_sequence(
            smpl_model, pose_params_start, beta_zero, beta_params, threshold=0.0025)
        shape_seq = np.stack(shape_seq, axis=0)

        pose_seq = self._generate_smooth_pose_sequence(
            smpl_model, pose_params_start, pose_params_end, beta_params, threshold=0.0025)
        pose_seq = np.stack(pose_seq, axis=0)[1:]

        body_sequence = np.concatenate([shape_seq, pose_seq], axis=0)

        # Align Y axis
        default_body_vertices = []
        with open('assets/bodies/f_smpl_average_A40.obj', 'rb') as f:
            for line in f:
                if line.startswith(b'v '):
                    default_body_vertices.append([float(x) for x in line.strip().split()[1:4]])
        default_body_vertices = np.array(default_body_vertices, dtype=np.float32)

        with open('assets/bodies/smpl_vert_segmentation.json', 'r') as f:
            smpl_seg = json.load(f)
        with open('assets/bodies/smplx_body_segmentation.json', 'r') as f:
            smplx_seg = json.load(f)

        body_sequence = self._align_smplx_body_y_axis(body_sequence, smpl_seg, smplx_seg, default_body_vertices)
        dynamic_frames = len(body_sequence)

        # Config setup
        props = data_config.Properties('./assets/Sim_props/default_sim_props.yaml')
        props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
        props.set_section_stats('render', render_time={})

        # Save the pattern
        def_sew_pattern = MetaGarment('Configured_design', self.default_body_params, self.design_params)
        pattern_folder = self.save(False, save_pattern=def_sew_pattern)

        # Build paths using PathCofigNew (SMPL body)
        out_path = str(self.save_path / 'dynamic_sim')
        paths = PathCofigNew(
            in_element_path=pattern_folder,
            out_path=out_path,
            in_name=def_sew_pattern.name,
            out_name='DynSim',
            body_name='smplx_body',
            smpl_body=True,
            add_timestamp=False
        )

        # Generate garment box mesh
        garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
        garment_box_mesh.load()
        garment_box_mesh.serialize(paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])
        props.serialize(paths.element_sim_props)

        # Run dynamic simulation — collect vertex snapshots in memory (no per-frame I/O)
        collected_frames = run_sim_new(
            garment_box_mesh.name,
            props,
            paths,
            body_sequence=body_sequence,
            body_faces=body_faces,
            save_v_norms=False,
            store_usd=False,
            optimize_storage=False,
            verbose=False,
            dynamic_frames=dynamic_frames,
            gpu_id=gpu_id,
            render_each_frame=False,
            render_stride=20,
            collect_frame_verts=True,
        )
        props.serialize(paths.element_sim_props)

        # Convert final garment mesh to GLB
        mesh = trimesh.load_mesh(paths.g_sim)
        pbr_material = mesh.visual.material.to_pbr()
        pbr_material.doubleSided = True
        mesh.visual.material = pbr_material
        mesh.export(paths.g_sim_glb)

        # Export final body mesh as GLB (last frame of body_sequence)
        body_glb_name = 'body_final.glb'
        body_glb_path = paths.out_el / body_glb_name
        final_body_verts = body_sequence[-1]
        body_mesh = trimesh.Trimesh(vertices=final_body_verts, faces=body_faces)
        body_mesh.visual.face_colors = [80, 70, 65, 255]
        body_mesh.export(str(body_glb_path))

        self.paths_3d = paths
        self.is_in_3D = True

        # Export per-frame GLBs for interactive 3D playback
        frame_glb_names = []
        if collected_frames:
            frame_glb_names = self._export_frame_glbs(
                collected_frames, paths, body_faces, paths.out_el
            )

        return paths.out_el, paths.g_sim_glb.name, body_glb_name, frame_glb_names

    # Current state
    def is_design_sectioned(self):
        """Check if design parameters are grouped by sections: 
            the top level of design dictionary does not contain actual parameters    
        """
        for param in self.design_params:
            if 'v' in self.design_params[param]:
                return False
        return True

    def is_slow_design(self) -> bool:
        """Check is parameters that result in slow pattern generation are enabled

            E.g. curved armhole evaluation
        """
        # Pants
        if (self.design_params['meta']['bottom']['v'] == 'Pants'):
            return True

        # Upper garment
        is_not_upper = self.design_params['meta']['upper']['v'] is None
        if is_not_upper:
            return False
        
        # Upper + fitted + strapless
        is_asymm = self.design_params['left']['enable_asym']['v']
        is_fitted = 'Fitted' in self.design_params['meta']['upper']['v']
        is_strapless = self.design_params['shirt']['strapless']['v']
        is_asymm_strapless = self.design_params['left']['shirt']['strapless']['v']

        is_strapless = is_fitted and is_strapless
        is_asymm_strapless = is_fitted and is_asymm_strapless

        # Has a hoody
        collar_component = self.design_params['collar']['component']['style']['v']
        has_hoody = collar_component is not None and 'Hood' in collar_component

        # Sleeve potential setup
        sleeves = self.design_params['sleeve']        
        is_sleeveless = sleeves['sleeveless']['v']
        is_curve = sleeves['armhole_shape']['v'] == 'ArmholeCurve'
        is_curve = not is_sleeveless and is_curve
        
        is_asym_sleeveless = self.design_params['left']['sleeve']['sleeveless']['v']
        is_asymm_curve = self.design_params['left']['sleeve']['armhole_shape']['v'] == 'ArmholeCurve'
        is_asymm_curve = not is_asym_sleeveless and is_asymm_curve

        if is_asymm:
            right_check = (not is_strapless) and is_curve
            left_check = (not is_asymm_strapless) and is_asymm_curve
            return right_check or left_check
        else:
            return (not is_strapless) and is_curve or has_hoody

    def save(self, pack=True, save_pattern: Optional[MetaGarment]=None):
        """Save current garment design to self.save_path """

        # Save current pattern
        if save_pattern is None:
            save_pattern = self.sew_pattern

        pattern = save_pattern.assembly()

        # Save as json file
        self.saved_garment_folder = pattern.serialize(
            self.save_path, 
            to_subfolder=True, 
            with_3d=False, with_text=False, view_ids=False, 
            with_printable=True,
            empty_ok=True
        )

        self.saved_garment_folder = Path(self.saved_garment_folder)
        self.body_params.save(self.saved_garment_folder)

        with open(self.saved_garment_folder / 'design_params.yaml', 'w') as f:
            yaml.dump(
                {'design': self.design_params}, 
                f,
                default_flow_style=False,
                sort_keys=False
            )

        # pack
        if pack: 
            # Only add geometry if design didn't change since last drape
            if not self.is_in_3D:
                self.clear_3d()  # Clean any saved 3D if it's not synced with current design
            self.saved_garment_archive = Path(shutil.make_archive(
                self.save_path / '..' / f'{self.saved_garment_folder.name}_{self.id}', 'zip',
                root_dir=self.save_path
            ))

        print(f'Success! {self.sew_pattern.name} saved to {self.saved_garment_folder}')

        return self.saved_garment_archive if pack else self.saved_garment_folder

