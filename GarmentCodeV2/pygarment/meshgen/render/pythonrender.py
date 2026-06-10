import os
import platform
import ctypes
import ctypes.util


def _egl_device_count():
    egl_lib = ctypes.util.find_library("EGL")
    if not egl_lib:
        return 0
    try:
        egl = ctypes.CDLL(egl_lib)
    except OSError:
        return 0

    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query_addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not query_addr:
        return 0

    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(query_addr)
    count = ctypes.c_int(0)
    if not query_devices(0, None, ctypes.byref(count)):
        return 0
    return count.value


if platform.system() == "Linux" and "PYOPENGL_PLATFORM" not in os.environ:
    # Prefer GPU EGL when a device is visible; otherwise use CPU OSMesa for headless containers.
    os.environ["PYOPENGL_PLATFORM"] = "egl" if _egl_device_count() > 0 else "osmesa"
import numpy as np
import trimesh
import pyrender
from PIL import Image
import copy
from pathlib import Path
from OpenGL.error import GLError
from pygarment.meshgen.sim_config import PathCofig


_RENDERER_DELETE_WARNING_EMITTED = False


def _safe_delete_renderer(renderer):
    """Release OffscreenRenderer resources without propagating cleanup errors."""
    global _RENDERER_DELETE_WARNING_EMITTED
    if renderer is None:
        return
    try:
        renderer.delete()
    except GLError as cleanup_err:
        if not _RENDERER_DELETE_WARNING_EMITTED:
            print("Warning: renderer cleanup skipped due to EGL context error; continuing without freeing GPU context.")
            _RENDERER_DELETE_WARNING_EMITTED = True
        # Prevent repeated delete attempts on the same renderer instance
        try:
            renderer._renderer = None
            renderer._platform = None
        except AttributeError:
            pass
    except Exception as cleanup_err:
        if not _RENDERER_DELETE_WARNING_EMITTED:
            print(f"Warning: renderer cleanup skipped ({cleanup_err}); continuing.")
            _RENDERER_DELETE_WARNING_EMITTED = True


def rotate_matrix_y(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def rotate_matrix_x(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def get_bounding_box_edges(mesh):
    # Calculate the bounding box of the mesh
    min_coords = mesh.bounds[0]
    max_coords = mesh.bounds[1]

    # Compute the corner points of the bounding box
    corners = [
        min_coords,
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        max_coords
    ]

    return corners

def create_camera(pyrender, pyrender_body_mesh, scene, side, camera_location=None):

    # Create a camera
    y_fov = np.pi / 6. 
    camera = pyrender.PerspectiveCamera(yfov=y_fov)
    

    if camera_location is None:
        # Evaluate w.r.t. body

        fov = 50  # Set your desired field of view in degrees 

        # # Calculate the bounding box center of the mesh
        bounding_box_center = pyrender_body_mesh.bounds.mean(axis=0)

        # Calculate the diagonal length of the bounding box
        diagonal_length = np.linalg.norm(pyrender_body_mesh.bounds[1] - pyrender_body_mesh.bounds[0])

        # Calculate the distance of the camera from the object based on the diagonal length
        distance = 1.5 * diagonal_length / (2 * np.tan(np.radians(fov / 2)))

        camera_location = bounding_box_center
        camera_location[-1] += distance

    # Calculate the camera pose
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_location[0]],
        [0.0, 1.0, 0.0, camera_location[1]],
        [0.0, 0.0, 1.0, camera_location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # camera_pose = rotate_matrix_x(camera_pose, -15)
    # camera_pose = rotate_matrix_y(camera_pose, 20)
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)

    # Set camera's pose in the scene
    scene.add(camera, pose=camera_pose)

def create_lights(scene, intensity=30.0):
    # [x,y,z]: x: 左右（左负右正）y：上下（上正下负）z：前后（前正后负）
    light_positions = [
        np.array([1.60614, 1.5341, 1.23701]),
        np.array([1.31844, 1.92831, -2.52238]),
        np.array([-2.80522, 1.2594, 2.34624]),
        np.array([0.160261, 1.81789, 3.52215]),
        np.array([-2.65752, 1.41194, -1.26328])
    ]
    light_colors = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

    # Add lights to the scene
    for i in range(5):
        light = pyrender.PointLight(color=light_colors[i], intensity=intensity)
        light_pose = np.eye(4)
        light_pose[:3, 3] = light_positions[i]
        scene.add(light, pose=light_pose)


def color_with_matte_alpha(white_color, black_color):
    """Recover smooth alpha from white/black background renders."""
    white_rgb = white_color[..., :3].astype(np.float32)
    black_rgb = black_color[..., :3].astype(np.float32)
    alpha = 1.0 - np.mean(white_rgb - black_rgb, axis=-1) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)

    rgb = np.zeros_like(black_rgb)
    visible = alpha > 1e-6
    rgb[visible] = black_rgb[visible] / alpha[visible, None]
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    alpha_u8 = np.round(alpha * 255.0).astype(np.uint8)
    return np.dstack([rgb, alpha_u8])


def render_scene_rgba(renderer, scene):
    """Render RGBA; use matte extraction when the backend drops alpha."""
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    if color.ndim == 3 and color.shape[-1] == 4:
        alpha = color[..., 3]
        if alpha.min() < alpha.max():
            return color
        color = color[..., :3]

    original_bg = scene.bg_color
    try:
        scene.bg_color = (0.0, 0.0, 0.0, 0.0)
        black_color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        scene.bg_color = original_bg
    return color_with_matte_alpha(color, black_color)


def render_supersample_factor(render_props=None):
    if render_props and "supersample" in render_props:
        return max(int(render_props["supersample"]), 1)
    if os.environ.get("PYOPENGL_PLATFORM") == "osmesa":
        return max(int(os.environ.get("PYGARMENT_RENDER_SUPERSAMPLE", "2")), 1)
    return 1


def downsample_rgba(color, target_size):
    """Downsample RGBA using premultiplied alpha to keep smooth transparent edges."""
    if color.shape[1::-1] == target_size:
        return color

    rgba = color.astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    premul = np.dstack([rgba[..., :3] * alpha, alpha])
    premul_img = Image.fromarray(np.round(premul * 255.0).astype(np.uint8), mode="RGBA")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = np.asarray(premul_img.resize(target_size, resampling)).astype(np.float32) / 255.0

    out_alpha = resized[..., 3:4]
    out_rgb = np.zeros_like(resized[..., :3])
    visible = out_alpha[..., 0] > 1e-6
    out_rgb[visible] = resized[..., :3][visible] / out_alpha[visible]
    return np.dstack([
        np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8),
        np.clip(out_alpha[..., 0] * 255.0, 0.0, 255.0).astype(np.uint8),
    ])


def render(
    pyrender_garm_mesh, pyrender_body_mesh, 
    side, 
    paths: PathCofig, 
    render_props=None,
    output_path=None,
    ):
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080
    # Create a pyrender scene
    scene = pyrender.Scene(bg_color=(1., 1., 1., 0.), ambient_light=(0.3, 0.3, 0.3))  # Transparent!
    
    # Create a pyrender mesh object from the trimesh object
    # Add the mesh to the scene
    scene.add(pyrender_garm_mesh)
    if not (render_props and render_props.get('hide_body', False)):
        scene.add(pyrender_body_mesh)

    camera_location=render_props['front_camera_location'] if 'front_camera_location' in render_props else None
    create_camera(
        pyrender, pyrender_body_mesh, scene, side,
        camera_location=camera_location
    )

    create_lights(scene, intensity=50.)

    renderer = None
    try:
        supersample = render_supersample_factor(render_props)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=view_width * supersample,
            viewport_height=view_height * supersample,
        )
        color = render_scene_rgba(renderer, scene)
        color = downsample_rgba(color, (view_width, view_height))

        image = Image.fromarray(color)
        out_path = output_path if output_path is not None else paths.render_path(side)
        # Ensure output directory exists (e.g., per-frame renders under frames/)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        image.save(out_path, "PNG")
    finally:
        _safe_delete_renderer(renderer)

def render_body(
        pyrender_body_mesh, 
        side, 
        paths: PathCofig, 
        render_props=None
    ):
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080
    # Create a pyrender scene
    scene = pyrender.Scene(bg_color=(0., 0., 0., 1.), ambient_light=(0.3, 0.3, 0.3))  # Transparent!

    # Create a pyrender mesh object from the trimesh object
    # Add the mesh to the scene
    scene.add(pyrender_body_mesh)

    camera_location=render_props['front_camera_location'] if 'front_camera_location' in render_props else None
    create_camera(
        pyrender, pyrender_body_mesh, scene, side,
        camera_location=camera_location
    )

    # light
    create_lights(scene, intensity=50.)

    renderer = None
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        import numpy as np
        from PIL import Image

        # 只取R通道（白色为255，黑色为0），阈值化
        mask = (color[..., 0] > 100).astype(np.uint8)
        mask_img = Image.fromarray(mask * 255, mode='L')
        mask_path = paths.render_path(side).with_name(paths.render_path(side).stem + "_bodymask.png")
        mask_img.save(mask_path)
    finally:
        _safe_delete_renderer(renderer)

    # image = Image.fromarray(color)
    # image.save(paths.render_path(side).with_name(paths.render_path(side).stem + "_body.png"))

def render_garment_mask(
        garm_mesh_white, 
        side, 
        paths: PathCofig, 
        render_props=None
    ):
    # 解析分辨率
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080

    # 创建黑色背景的scene
    scene = pyrender.Scene(bg_color=(0., 0., 0., 1.))
    scene.add(garm_mesh_white)

    # 相机参数与主渲染保持一致
    camera_location = render_props['front_camera_location'] if render_props and 'front_camera_location' in render_props else None
    create_camera(pyrender, garm_mesh_white, scene, side, camera_location=camera_location)

    renderer = None
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)

        # 转为灰度二值mask（garment为1，背景为0），输出单通道
        import numpy as np
        from PIL import Image

        # 只取R通道（白色为255，黑色为0），阈值化
        mask = (color[..., 0] > 127).astype(np.uint8)
        mask_img = Image.fromarray(mask * 255, mode='L')
        mask_path = paths.render_path(side).with_name(paths.render_path(side).stem + "_mask.png")
        mask_img.save(mask_path)
    finally:
        _safe_delete_renderer(renderer)


def render_body_mask(
    pyrender_body_mesh, pyrender_garm_mesh,
    side,
    paths: PathCofig,
    render_props=None,
    base_render_path=None,
    ):
    renderer_body = None
    renderer_garm = None
    try:
        # 解析分辨率
        if render_props and 'resolution' in render_props:
            view_width, view_height = render_props['resolution']
        else:
            view_width, view_height = 1080, 1080

        # 渲染 body mask
        scene_body = pyrender.Scene(bg_color=(0., 0., 0., 1.))
        scene_body.add(pyrender_body_mesh)
        camera_location = render_props['front_camera_location'] if render_props and 'front_camera_location' in render_props else None
        create_camera(pyrender, pyrender_body_mesh, scene_body, side, camera_location=camera_location)
        renderer_body = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)

        # 渲染 body，获取 mask 和 depth
        color_body, depth_body = renderer_body.render(scene_body, flags=pyrender.RenderFlags.FLAT)
        body_mask = (color_body[..., 0] > 200).astype(np.uint8)
        _safe_delete_renderer(renderer_body)
        renderer_body = None

        # 渲染 garment，获取 mask 和 depth
        scene_garment = pyrender.Scene(bg_color=(0., 0., 0., 1.))
        scene_garment.add(pyrender_garm_mesh)
        create_camera(pyrender, pyrender_garm_mesh, scene_garment, side, camera_location=camera_location)
        renderer_garm = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color_garment, depth_garment = renderer_garm.render(scene_garment, flags=pyrender.RenderFlags.FLAT)
        garment_mask = (color_garment[..., 0] > 200).astype(np.uint8)
        _safe_delete_renderer(renderer_garm)
        renderer_garm = None

        visible_body_mask, visible_garment_mask = visible_body_garment_masks(
            body_mask, garment_mask, depth_body, depth_garment)

        # 保存 mask
        
        base_path = base_render_path if base_render_path is not None else paths.render_path(side)
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        mask_img_0 = Image.fromarray(visible_body_mask * 255, mode='L')
        mask_path = base_path.with_name(base_path.stem + "_bodymask.png")
        mask_img_0.save(mask_path)

        mask_img_1 = Image.fromarray(visible_garment_mask * 255, mode='L')
        mask_path_1 = base_path.with_name(base_path.stem + "_mask.png")
        mask_img_1.save(mask_path_1)
        
    except Exception as e:
        print(f"Warning: render_body_mask failed with error: {e}")
    finally:
        _safe_delete_renderer(renderer_body)
        _safe_delete_renderer(renderer_garm)


def load_meshes(paths:PathCofig, body_v, body_f):
    # Load body mesh
    body_mesh = trimesh.Trimesh(body_v, body_f)
    body_mesh.vertices = body_mesh.vertices / 100
    # Color body mesh
    # body_material = pyrender.MetallicRoughnessMaterial(
    #     baseColorFactor=(0.0, 0.0, 0.0, 1.0),  # RGB color, Alpha
    #     metallicFactor=0.658,  # Range: [0.0, 1.0]
    #     roughnessFactor=0.5  # Range: [0.0, 1.0]
    # )
    body_material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.2, 0.17, 0.15, 1.0))
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)


    #Load garment mesh
    garm_mesh = trimesh.load_mesh(str(paths.g_sim))  # NOTE: Includes the texture
    garm_mesh.vertices = garm_mesh.vertices / 100   # scale to m

    # Material adjustments
    material = garm_mesh.visual.material.to_pbr()
    material.baseColorFactor = [1., 1., 1., 1.]
    material.doubleSided = True  # color both face sides  
    # NOTE remove transparency -- add white background just in case
    white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
    white_back.paste(material.baseColorTexture)
    material.baseColorTexture = white_back.convert('RGB')  

    garm_mesh.visual.material = material

    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garm_mesh, smooth=True) 
    
    return pyrender_garm_mesh, pyrender_body_mesh

def render_images(paths: PathCofig, body_v, body_f, render_props):

    return render_images_for_frame(paths, body_v, body_f, render_props, frame_idx=None, include_masks=True)


def render_images_for_frame(
        paths: PathCofig,
        body_v,
        body_f,
        render_props,
        frame_idx=None,
        include_masks=True,
    ):

    pyrender_garm_mesh, pyrender_body_mesh = load_meshes(paths, body_v, body_f)

    for side in render_props['sides']:
        garm_mesh = copy.deepcopy(pyrender_garm_mesh)
        body_mesh = copy.deepcopy(pyrender_body_mesh)
        out_path = paths.render_frame_path(frame_idx, side) if frame_idx is not None else paths.render_path(side)
        render(garm_mesh, body_mesh, side, paths, render_props, output_path=out_path)

        if include_masks and side == 'front':
            new_material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            new_pyrender_body_mesh = copy.deepcopy(pyrender_body_mesh)
            new_pyrender_body_mesh.primitives[0].material = new_material
            white_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(1.0, 1.0, 1.0, 1.0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True
            )
            garm_mesh_white = copy.deepcopy(pyrender_garm_mesh)
            garm_mesh_white.primitives[0].material = white_material
            # render_body(new_pyrender_body_mesh, side, paths, render_props)
            # render_garment_mask(garm_mesh_white, side, paths, render_props)
            render_body_mask(new_pyrender_body_mesh, garm_mesh_white, side, paths, render_props, base_render_path=out_path)


def load_garment_mesh(paths: PathCofig):
    garm_mesh = trimesh.load_mesh(str(paths.g_sim))
    garm_mesh.vertices = garm_mesh.vertices / 100

    material = garm_mesh.visual.material.to_pbr()
    material.baseColorFactor = [1., 1., 1., 1.]
    material.doubleSided = True
    if material.baseColorTexture is not None:
        white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
        white_back.paste(material.baseColorTexture)
        material.baseColorTexture = white_back.convert('RGB')

    garm_mesh.visual.material = material
    return pyrender.Mesh.from_trimesh(garm_mesh, smooth=True)


def visible_body_garment_masks(body_mask, garment_mask, depth_body, depth_garment):
    body_mask = body_mask.astype(bool)
    garment_mask = garment_mask.astype(bool)
    body_has_depth = depth_body > 0
    garment_has_depth = depth_garment > 0
    body_in_front = body_has_depth & garment_has_depth & (depth_body < depth_garment)
    garment_in_front = body_has_depth & garment_has_depth & (depth_garment < depth_body)

    visible_body_mask = body_mask & (~garment_mask | body_in_front)
    visible_garment_mask = garment_mask & (~body_mask | garment_in_front)
    return visible_body_mask.astype(np.uint8), visible_garment_mask.astype(np.uint8)


def set_mesh_material(mesh, material):
    for primitive in mesh.primitives:
        primitive.material = material


def render_multi_garments(pyrender_garm_meshes, pyrender_body_mesh, side, render_props, output_path):
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080

    scene = pyrender.Scene(bg_color=(1., 1., 1., 0.), ambient_light=(0.3, 0.3, 0.3))
    for mesh in pyrender_garm_meshes:
        scene.add(mesh)
    if not (render_props and render_props.get('hide_body', False)):
        scene.add(pyrender_body_mesh)

    camera_location = render_props['front_camera_location'] if 'front_camera_location' in render_props else None
    create_camera(pyrender, pyrender_body_mesh, scene, side, camera_location=camera_location)
    create_lights(scene, intensity=50.)

    renderer = None
    try:
        supersample = render_supersample_factor(render_props)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=view_width * supersample,
            viewport_height=view_height * supersample,
        )
        color = render_scene_rgba(renderer, scene)
        color = downsample_rgba(color, (view_width, view_height))
        image = Image.fromarray(color)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "PNG")
    finally:
        _safe_delete_renderer(renderer)


def render_multi_body_mask(pyrender_garm_meshes, pyrender_body_mesh, side, render_props, base_render_path):
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080

    camera_location = render_props['front_camera_location'] if 'front_camera_location' in render_props else None
    renderer_body = None
    renderer_garm = None
    try:
        scene_body = pyrender.Scene(bg_color=(0., 0., 0., 1.))
        scene_body.add(pyrender_body_mesh)
        create_camera(pyrender, pyrender_body_mesh, scene_body, side, camera_location=camera_location)
        renderer_body = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color_body, depth_body = renderer_body.render(scene_body, flags=pyrender.RenderFlags.FLAT)
        body_mask = (color_body[..., 0] > 200).astype(np.uint8)
        _safe_delete_renderer(renderer_body)
        renderer_body = None

        scene_garment = pyrender.Scene(bg_color=(0., 0., 0., 1.))
        for mesh in pyrender_garm_meshes:
            scene_garment.add(mesh)
        create_camera(pyrender, pyrender_body_mesh, scene_garment, side, camera_location=camera_location)
        renderer_garm = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color_garment, depth_garment = renderer_garm.render(scene_garment, flags=pyrender.RenderFlags.FLAT)
        garment_mask = (color_garment[..., 0] > 200).astype(np.uint8)
        _safe_delete_renderer(renderer_garm)
        renderer_garm = None

        visible_body_mask, visible_garment_mask = visible_body_garment_masks(
            body_mask, garment_mask, depth_body, depth_garment)
        base_render_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(visible_body_mask * 255, mode='L').save(
            base_render_path.with_name(base_render_path.stem + "_bodymask.png"))
        Image.fromarray(visible_garment_mask * 255, mode='L').save(
            base_render_path.with_name(base_render_path.stem + "_mask.png"))
    except Exception as e:
        print(f"Warning: render_multi_body_mask failed with error: {e}")
    finally:
        _safe_delete_renderer(renderer_body)
        _safe_delete_renderer(renderer_garm)


def render_multi_images(paths_list, body_v, body_f, render_props, output_dir, name='multi', include_masks=True):
    body_mesh = trimesh.Trimesh(body_v, body_f)
    body_mesh.vertices = body_mesh.vertices / 100
    body_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.2, 0.17, 0.15, 1.0)
    )
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)
    pyrender_garm_meshes = [load_garment_mesh(paths) for paths in paths_list]

    output_dir = Path(output_dir)
    mask_body_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    mask_garment_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(1.0, 1.0, 1.0, 1.0),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True
    )
    for side in render_props['sides']:
        garment_meshes = [copy.deepcopy(mesh) for mesh in pyrender_garm_meshes]
        body_mesh_copy = copy.deepcopy(pyrender_body_mesh)
        out_path = output_dir / f'{name}_render_{side}.png'
        render_multi_garments(
            garment_meshes,
            body_mesh_copy,
            side,
            render_props,
            out_path
        )
        if include_masks and side == 'front':
            mask_body = copy.deepcopy(pyrender_body_mesh)
            set_mesh_material(mask_body, mask_body_material)
            mask_garments = [copy.deepcopy(mesh) for mesh in pyrender_garm_meshes]
            for mesh in mask_garments:
                set_mesh_material(mesh, mask_garment_material)
            render_multi_body_mask(mask_garments, mask_body, side, render_props, out_path)



