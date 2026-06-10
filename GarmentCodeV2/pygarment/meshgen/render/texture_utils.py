"""Routines for processing UV coordinated for garments and generating texture maps"""
import hashlib
import json
import numpy as np
import igl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from pathlib import Path
import random
import re

# 全局panel分组索引（可根据实际需求补充/调整）

COMPONENT_PANEL = [
    'left_collar_back', 'left_collar_front', 'right_collar_back', 'right_collar_front'  
]
UPPER_PANEL = [
    'left_btorso', 'left_ftorso', 'left_hood', 'left_sleeve_b', 'left_sleeve_f',
    'right_btorso', 'right_ftorso', 'right_hood', 'right_sleeve_b', 'right_sleeve_f',
    'sl_left_cuff_b', 'sl_left_cuff_f', 'sl_left_cuff_skirt_b', 'sl_left_cuff_skirt_f',
    'sl_right_cuff_b', 'sl_right_cuff_f', 'sl_right_cuff_skirt_b', 'sl_right_cuff_skirt_f',
]
WAIST_PANEL = [
    'wb_back', 'wb_front'
]
LOWER_PANEL = [
    'pant_b_l', 'pant_b_r', 'pant_f_l', 'pant_f_r', 'pant_l_cuff_b', 'pant_l_cuff_f', 'pant_l_cuff_skirt_b', 'pant_l_cuff_skirt_f',
    'pant_r_cuff_b', 'pant_r_cuff_f', 'pant_r_cuff_skirt_b', 'pant_r_cuff_skirt_f',
    'skirt_back', 'skirt_back_0', 'skirt_back_1', 'skirt_back_2', 'skirt_back_3', 'skirt_back_4',
    'skirt_front', 'skirt_front_0', 'skirt_front_1', 'skirt_front_2', 'skirt_front_3', 'skirt_front_4',
    'skirt_panel_0', 'skirt_panel_1', 'skirt_panel_2', 'skirt_panel_3', 'skirt_panel_4', 'skirt_panel_5',
    'skirt_panel_6', 'skirt_panel_7', 'skirt_panel_8', 'skirt_panel_9', 'skirt_panel_10', 'skirt_panel_11',
    'skirt_panel_12', 'skirt_panel_13', 'skirt_panel_14',
    'ins_skirt_back_0', 'ins_skirt_back_1', 'ins_skirt_back_2', 'ins_skirt_back_3', 'ins_skirt_back_4', 'ins_skirt_back_5',
    'ins_skirt_front_0', 'ins_skirt_front_1', 'ins_skirt_front_2', 'ins_skirt_front_3', 'ins_skirt_front_4', 'ins_skirt_front_5'
]

# 10种常见服饰颜色（RGB，0~1）
FASHION_COLORS = [
    (0.6, 0.6, 0.6),  # 灰色 Gray
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEXTURE_BINDINGS_PATH = PROJECT_ROOT / 'assets' / 'newcloth_texture_bindings.json'
_TEXTURE_BINDINGS_CACHE = {}

# SECTION UV islands texture creation 
def texture_mesh_islands(
        texture_coords, face_texture_coords, 
        out_texture_image_path: Path, 
        out_fabric_tex_image_path: Path = None, 
        out_mtl_file_path: Path = None, 
        boundary_width=0.3, 
        dpi=1200, 
        background_img_path=None,
        background_resolution=1.,
        uv_padding=3, 
        mat_name='islands_texture',
        panel_names=None,
        spec_file_path=None,
        texture_bindings_path=None
):
    """
        Returns updated uv coordinates (properly normalized and aligned with the created texture)
    """
    all_uvs, boundary_uv_to_draw = unwarp_UV(texture_coords, face_texture_coords, padding=uv_padding)
        
    uv_list, width, height = normalize_UVs(all_uvs, axis_padding=uv_padding)   # NOTE !! Axis padding should match the uv padding

    # Create image
    create_UV_island_texture_random_color(
        boundary_uv_to_draw, width, height,
        texture_image_path=out_texture_image_path,
        boundary_width=boundary_width,
        dpi=dpi,
        preserve_alpha=True,
        panel_names=panel_names,
        spec_file_path=spec_file_path,
        texture_bindings_path=texture_bindings_path
    )

    # Create image with fabric background
    if out_fabric_tex_image_path is not None:
        create_UV_island_texture_random_color(
            boundary_uv_to_draw, width, height,
            texture_image_path=out_fabric_tex_image_path,
            boundary_width=boundary_width,
            dpi=dpi,
            background_img_path=background_img_path,
            background_resolution=background_resolution,
            preserve_alpha=False,
            panel_names=panel_names,
            spec_file_path=spec_file_path,
            texture_bindings_path=texture_bindings_path
        )

    # Save mtl is requested
    if out_mtl_file_path:
        save_texture_mtl(
            out_mtl_file_path, 
            out_fabric_tex_image_path.name if out_fabric_tex_image_path is not None else out_texture_image_path.name, 
            mat_name=mat_name)

    return uv_list

def _uv_connected_components(face_texture_coords):

    # Find connected components of face and vertex texture coords
    face_components = igl.facet_components(face_texture_coords)
    vert_components = igl.vertex_components(face_texture_coords)
    num_ccs = max(face_components) + 1

    return vert_components, face_components, num_ccs

def unwarp_UV(texture_coords, face_texture_coords, padding=3):
    # Unwrap uvs for each connected component------------------------

    vert_components, face_components, num_ccs = _uv_connected_components(face_texture_coords)

    all_uvs = [] # transform all UVs to update obj file
    boundary_uv_to_draw = [] # only draw the boundary UVs

    translate_Y = 0
    translate_X = 0

    shells_per_row = int(num_ccs ** 0.5)
    column_x_shift = 0

    # Loop through each connected component
    for i in range(num_ccs):
        
        # Get faces and vertices of connected component
        faces_in_cc = np.where(face_components == i)[0]
        face_vts_in_cc = face_texture_coords[faces_in_cc]

        # get all vertices of connected component
        verts_in_cc = np.where(vert_components == i)[0]

        all_vert_pos = texture_coords[verts_in_cc]
        
        # Find boundary loop
        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_vert_pos = texture_coords[bound_verts]

        # Shift component by bounding box
        bbox = bound_vert_pos.min(axis=0), bound_vert_pos.max(axis=0)
        bbox_len_Y = (bbox[1][1] - bbox[0][1])
        bbox_len_X = (bbox[1][0] - bbox[0][0])
    
        if (i % shells_per_row == 0):
            # Start new column
            translate_Y = padding
            translate_X += (column_x_shift + padding)
            column_x_shift = 0  # restart BBOX collection

        # Update shift
        column_x_shift = max(bbox_len_X, column_x_shift)

        # translate boundary positions
        verts_translated_bound = [(x + translate_X, y + translate_Y) for x, y in bound_vert_pos]
        boundary_uv_to_draw.append(verts_translated_bound)
        
        # translate all positions
        verts_translated = [(x + translate_X, y + translate_Y) for x, y in all_vert_pos]
        all_uvs.extend(verts_translated)
        
        translate_Y = translate_Y + bbox_len_Y + padding

    return all_uvs, boundary_uv_to_draw  

def normalize_UVs(all_uvs, axis_padding=3):
    # normalize all_uvs
    uv_list_raw = np.array(all_uvs)
    uv_list = uv_list_raw

    norm_x = max(uv_list_raw[:,0]) + axis_padding
    uv_list[:,0] = uv_list_raw[:,0] / norm_x
    norm_y = max(uv_list_raw[:,1]) + axis_padding
    uv_list[:,1] = uv_list_raw[:,1] / norm_y

    return uv_list, norm_x, norm_y

import random

def create_UV_island_texture_random_color(
        boundary_uv_to_draw, 
        width, height, 
        texture_image_path, 
        boundary_width=0.0, 
        dpi=1200,
        color_alpha=0.65,
        background_alpha=0.8,
        background_img_path=None,
        background_resolution=5,
        preserve_alpha=True,
        panel_names=None,
        spec_file_path=None,
        texture_bindings_path=None
    ):
    n_components = len(boundary_uv_to_draw)
    texture_lookup = build_panel_texture_lookup(panel_names, spec_file_path, texture_bindings_path)

    # Figure size
    fig, ax = plt.subplots()
    fig.set_size_inches(width / 100, height / 100)

    color_map = {
        'upper': (0.75, 0.75, 0.75, color_alpha),   # 白色
        'waist': (0.42, 0.22, 0.10, color_alpha), # 棕色
        'lower': (0.08, 0.08, 0.08, color_alpha),   # 黑色
        'component': (1.0, 1.0, 1.0, color_alpha)
    }
    def get_panel_type(name):
        if name in WAIST_PANEL:
            return 'waist'
        elif name in LOWER_PANEL:
            return 'lower'
        elif name in COMPONENT_PANEL:
            return 'component'
        else:
            return 'upper'

    def has_visible_tint(binding):
        tint = str(binding.get('tint', '#FFFFFF')).strip().lstrip('#').lower()
        try:
            alpha = float(binding.get('tint_alpha', 0.25))
        except (TypeError, ValueError):
            alpha = 0.25
        return tint not in ('', 'ffffff') and alpha > 0

    def select_background_binding():
        if texture_lookup is None:
            return None
        ordered_panel_names = panel_names if panel_names is not None else texture_lookup.keys()
        for name in ordered_panel_names:
            binding = texture_lookup.get(name)
            if binding is not None and not has_visible_tint(binding):
                return binding
        return None

    background_binding = None
    if texture_lookup is not None:
        background_binding = select_background_binding()
        if background_binding is not None:
            draw_texture_background(ax, width, height, background_binding, background_resolution)
    elif background_img_path is not None:
        back_crop_scale = background_resolution
        back_img = plt.imread(background_img_path)
        ax.imshow(
            back_img[:int(width * back_crop_scale), :int(height * back_crop_scale), :], 
            extent=[0, width, 0, height], 
            alpha=background_alpha,
            aspect='equal'
        )

    for i in range(n_components):
        polygon_x = [vert[0] for vert in boundary_uv_to_draw[i]]
        polygon_x.append(polygon_x[0])
        polygon_y = [vert[1] for vert in boundary_uv_to_draw[i]]
        polygon_y.append(polygon_y[0])

        panel_name = panel_names[i] if panel_names is not None and i < len(panel_names) else None
        texture_binding = texture_lookup.get(panel_name) if texture_lookup is not None else None
        if texture_binding is not None:
            if background_binding is None or binding_signature(texture_binding) != binding_signature(background_binding):
                draw_textured_polygon(
                    ax,
                    polygon_x,
                    polygon_y,
                    texture_binding,
                    background_resolution,
                    extent=[0, width, 0, height],
                )
            continue

        if panel_name is not None:
            panel_type = get_panel_type(panel_names[i])
            fill_color = color_map[panel_type]
        else:
            fill_color = color_map['upper']

        plt.fill(polygon_x, polygon_y,
                 color=fill_color,
                 edgecolor=fill_color, linestyle='-', linewidth=boundary_width / 2
        )

    ax.set_aspect('equal')
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    plt.axis('off')
    plt.savefig(texture_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=preserve_alpha)
    plt.close()


def load_texture_bindings(texture_bindings_path=None):
    path = Path(texture_bindings_path) if texture_bindings_path else DEFAULT_TEXTURE_BINDINGS_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None

    cache_key = str(path)
    if cache_key not in _TEXTURE_BINDINGS_CACHE:
        with path.open('r', encoding='utf-8') as file:
            _TEXTURE_BINDINGS_CACHE[cache_key] = json.load(file)
    return _TEXTURE_BINDINGS_CACHE[cache_key]


def build_panel_texture_lookup(panel_names, spec_file_path=None, texture_bindings_path=None):
    if panel_names is None:
        return None

    bindings = load_texture_bindings(texture_bindings_path)
    if not bindings:
        return None

    units = infer_newcloth_units(spec_file_path, panel_names)
    if not units:
        return None

    garments = bindings.get('garments', {})
    defaults = bindings.get('defaults', {})
    lookup = {}
    for panel_name in panel_names:
        binding = binding_for_panel(panel_name, units, garments, defaults)
        if binding is not None:
            lookup[panel_name] = binding
    return lookup if lookup else None


def infer_newcloth_units(spec_file_path, panel_names):
    if spec_file_path is None:
        return None

    path = Path(spec_file_path)
    stem = path.stem
    if stem in ('Upper_specification', 'Lower_specification'):
        source_name = path.parent.name
    else:
        source_name = stem.replace('_specification', '')
    source_name = source_name.split('__')[0]
    source_name = re.sub(r'_[0-9a-f]{16}$', '', source_name)

    dress_match = re.match(r'^(dress\d+)$', source_name)
    if dress_match:
        return {'dress': dress_match.group(1)}

    if re.match(r'^upper\d+$', source_name):
        return {'upper': source_name, 'lower': None}

    if is_lower_unit_name(source_name):
        return {'upper': None, 'lower': source_name}

    match = re.match(r'^(upper\d+)_(.+)$', source_name)
    if not match:
        return None

    upper_id, lower_id = match.groups()
    has_lower_panel = any(is_lower_panel_name(name) for name in panel_names)
    has_upper_panel = any(not is_lower_panel_name(name) for name in panel_names)
    return {
        'upper': upper_id if has_upper_panel else None,
        'lower': lower_id if has_lower_panel else None,
    }


def resolve_waist_binding(units, garments, defaults):
    waist_cfg = defaults.get('waist')
    if waist_cfg is None:
        return None
    if waist_cfg.get('use_lower_texture'):
        source_binding = None
        if units.get('lower'):
            source_binding = garments.get(units.get('lower'))
        elif 'dress' in units:
            source_binding = garments.get(units['dress'])
        if source_binding is not None:
            merged = dict(source_binding)
            for key in ('tint', 'tint_alpha', 'texture_scale', 'texture_tile_mode'):
                if key in waist_cfg and waist_cfg[key] is not None:
                    merged[key] = waist_cfg[key]
            return merged
    return waist_cfg


def binding_for_panel(panel_name, units, garments, defaults):
    if panel_name in defaults.get('component', {}).get('panel_names', []):
        return defaults.get('component')
    waist_prefixes = defaults.get('waist', {}).get('panel_prefixes', [])
    if waist_prefixes and panel_name.startswith(tuple(waist_prefixes)):
        return resolve_waist_binding(units, garments, defaults)
    if 'dress' in units:
        return garments.get(units['dress'])
    if is_lower_panel_name(panel_name):
        return garments.get(units.get('lower'))
    return garments.get(units.get('upper'))


def binding_signature(binding):
    if binding is None:
        return None
    return (
        str(binding.get('texture_path', '')),
        str(binding.get('texture_tile_mode', 'sample')),
        str(binding.get('texture_scale', '')),
        str(binding.get('tint', '#FFFFFF')),
        str(binding.get('tint_alpha', '')),
    )


def is_lower_panel_name(panel_name):
    return panel_name.startswith((
        'wb_',
        'pant_',
        'skirt_',
        'ins_skirt_',
        'skirt_panel_',
    ))


def is_lower_unit_name(unit_name):
    return bool(re.match(r'^(pants|pencilskirt|circleskirt)\d+$', unit_name))


def draw_textured_polygon(ax, polygon_x, polygon_y, binding, grain_resolution=1, extent=None):
    texture_path = resolve_texture_path(binding.get('texture_path', ''))
    if texture_path is None or not texture_path.exists():
        return

    seed = int(hashlib.sha256(str(texture_path).encode('utf-8')).hexdigest()[:8], 16)
    texture = prepare_tiled_texture(plt.imread(texture_path), binding, grain_resolution, seed)

    if extent is None:
        xmin, xmax = min(polygon_x), max(polygon_x)
        ymin, ymax = min(polygon_y), max(polygon_y)
        extent = [xmin, xmax, ymin, ymax]
    patch = Polygon(
        list(zip(polygon_x, polygon_y)),
        closed=True,
        facecolor='none',
        edgecolor='none',
    )
    ax.add_patch(patch)
    image = ax.imshow(texture, extent=extent, aspect='auto')
    image.set_clip_path(patch)

    tint_alpha = float(binding.get('tint_alpha', 0.25))
    tint = hex_to_rgba(binding.get('tint', '#FFFFFF'), alpha=tint_alpha)
    if tint is not None and tint[:3] != (1.0, 1.0, 1.0):
        ax.fill(polygon_x, polygon_y, color=tint, linewidth=0)


def draw_texture_background(ax, width, height, binding, grain_resolution=1):
    texture_path = resolve_texture_path(binding.get('texture_path', ''))
    if texture_path is None or not texture_path.exists():
        return

    texture = prepare_tiled_texture(plt.imread(texture_path), binding, grain_resolution, seed=0)
    ax.imshow(texture, extent=[0, width, 0, height], aspect='auto')


def prepare_tiled_texture(texture, binding, grain_resolution=1, seed=0):
    if texture.ndim == 2:
        texture = np.repeat(texture[:, :, None], 3, axis=2)

    tile_mode = binding.get('texture_tile_mode', 'sample')
    if tile_mode == 'crop':
        return texture

    scale = max(int(round(float(grain_resolution))), 1)
    if scale <= 1:
        return texture

    if tile_mode == 'sample':
        return sample_texture_patches(texture, scale, seed)
    if tile_mode == 'repeat':
        return np.tile(texture, (scale, scale, 1))
    return mirror_tile_texture(texture, scale)


def sample_texture_patches(texture, repeats, seed=0, overlap_fraction=0.25):
    height, width = texture.shape[:2]
    if repeats <= 1 or height < 2 or width < 2:
        return texture

    rng = np.random.default_rng(seed)
    overlap_y = max(1, int(round(height * overlap_fraction)))
    overlap_x = max(1, int(round(width * overlap_fraction)))
    step_y = max(1, height - overlap_y)
    step_x = max(1, width - overlap_x)
    out_h = height + step_y * (repeats - 1)
    out_w = width + step_x * (repeats - 1)

    canvas = np.zeros((out_h, out_w, texture.shape[2]), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)
    mask = feather_mask(height, width, overlap_y, overlap_x)

    padded = np.pad(
        texture,
        ((height // 2, height // 2), (width // 2, width // 2), (0, 0)),
        mode='reflect'
    )
    max_y = padded.shape[0] - height
    max_x = padded.shape[1] - width

    for row in range(repeats):
        for col in range(repeats):
            start_y = int(rng.integers(0, max_y + 1))
            start_x = int(rng.integers(0, max_x + 1))
            patch = padded[start_y:start_y + height, start_x:start_x + width]

            y0 = row * step_y
            x0 = col * step_x
            canvas[y0:y0 + height, x0:x0 + width] += patch * mask
            weights[y0:y0 + height, x0:x0 + width] += mask

    return canvas / np.maximum(weights, 1.0e-6)


def feather_mask(height, width, overlap_y, overlap_x):
    mask_y = np.ones(height, dtype=np.float32)
    if overlap_y > 0:
        ramp = np.linspace(0.05, 1.0, overlap_y, dtype=np.float32)
        mask_y[:overlap_y] *= ramp
        mask_y[-overlap_y:] *= ramp[::-1]

    mask_x = np.ones(width, dtype=np.float32)
    if overlap_x > 0:
        ramp = np.linspace(0.05, 1.0, overlap_x, dtype=np.float32)
        mask_x[:overlap_x] *= ramp
        mask_x[-overlap_x:] *= ramp[::-1]

    return (mask_y[:, None] * mask_x[None, :])[:, :, None]


def mirror_tile_texture(texture, repeats):
    rows = []
    for y in range(repeats):
        cols = []
        for x in range(repeats):
            tile = texture
            if x % 2 == 1:
                tile = np.flip(tile, axis=1)
            if y % 2 == 1:
                tile = np.flip(tile, axis=0)
            cols.append(tile)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def resolve_texture_path(texture_path):
    if not texture_path:
        return None
    path = Path(texture_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def hex_to_rgba(color, alpha=0.25):
    if not isinstance(color, str):
        return None
    color = color.strip().lstrip('#')
    if len(color) != 6:
        return None
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)

def create_UV_island_texture(
        boundary_uv_to_draw, 
        width, height, 
        texture_image_path, 
        boundary_width=0.3, 
        boundary_color='black',
        dpi=1200,
        color_alpha=0.65,
        background_alpha=0.8,
        background_img_path=None,
        background_resolution=5,
        preserve_alpha=True
    ):
    """Create texture image from the set of UV boundary loops (e.g. sewing pattern panels). 
        It renders the border of the loops and fills them in with color 
        Params: 
            * boundary_uv_to_draw -- 2D list -- sequence of 2D vertices on each of the boundaries. The order is IMPORTANT. The vertices will be connected 
                by boundary edges sequentially
            * width, height -- the dimentions of the UV map  
            * texture_image_path -- filepath to same a texture image to
            * boundary_width -- width of the boundary outline 
            * dpi -- resolution of the output image
    """
    n_components = len(boundary_uv_to_draw)

    # Figure size
    fig, ax = plt.subplots()
    fig.set_size_inches(width / 100, height / 100)  # width & height are usually given in cm

    # Colors
    shift = 0.17
    divisor = max(5, n_components)
    cmap = matplotlib.colormaps['twilight']   # copper cool  spring winter twilight  # Using smooth Matplotlib colormaps
    color_sample = [cmap((1 - shift) * id / divisor) for id in range(divisor)]

    # Background -- garment style
    if background_img_path is not None:
        back_crop_scale = background_resolution
        back_img = plt.imread(background_img_path)
        ax.imshow(
            back_img[:int(width * back_crop_scale), :int(height * back_crop_scale), :], 
            extent=[0, width, 0, height], 
            alpha=background_alpha,
            aspect='equal'
        )

    # Draw the UV island boundaries and fill them up
    for i in range(n_components):
        polygon_x = [vert[0] for vert in boundary_uv_to_draw[i]]
        polygon_x.append(polygon_x[0])  # Loop
        polygon_y = [vert[1] for vert in boundary_uv_to_draw[i]]
        polygon_y.append(polygon_y[0])  # Loop

        color = list(color_sample[i])
        color[-1] = color_alpha   # Alpha - transparency for blending with backround

        plt.fill(polygon_x, polygon_y, 
                 color=color, 
                 edgecolor=boundary_color, linestyle='-', linewidth=boundary_width / 2  # Boundary stylings
        )
        
    ax.set_aspect('equal')

    # Set the axis to be tight
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # Hide the axis
    plt.axis('off')

    # Save image
    plt.savefig(texture_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=preserve_alpha)

    # Cleanup
    plt.close()

# !SECTION

# SECTION Saving textures information to files
def save_texture_mtl(mtl_file_path, texture_image_name, mat_name='uv_texture'):
    new_material_lines = [
        f'newmtl {mat_name}\n',
        'Ns 0.000000\n',
        'Ka 1.000000 1.000000 1.000000\n',
        'Ks 0.000000 0.000000 0.000000\n',
        'Ke 0.000000 0.000000 0.000000\n',
        'Ni 1.000000\n',
        'd 1.000000\n',
        'illum 1\n',
        f'map_Kd {texture_image_name}\n'
    ]

    with open(mtl_file_path, 'w') as file:
        file.writelines(new_material_lines)

    return mat_name

def save_obj(
        output_file_path, 
        vertices, faces_with_texture, uv_list, 
        vert_normals=None, mtl_file_name=None, mat_name=None):
    """Save an obj file with a texture information (if provided)"""

    with open(output_file_path, 'w') as f:
        if mtl_file_name is not None:
            f.write(f'mtllib {mtl_file_name}\n')

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for vt in uv_list:
            f.write(f"vt {vt[0]} {vt[1]}\n")

        if vert_normals is not None:
            for vn in vert_normals:
                f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
            
        f.write('s 1\n')
        if mtl_file_name is not None:
            f.write(f'usemtl {mat_name}\n')

        if vert_normals is not None:
            for v_id0, tex_id0, v_id1, tex_id1, v_id2, tex_id2, in faces_with_texture:
                f.write(f"f {v_id0 + 1}/{tex_id0 + 1}/{v_id0 + 1} "
                        f"{v_id1 + 1}/{tex_id1 + 1}/{v_id1 + 1} "
                        f"{v_id2 + 1}/{tex_id2 + 1}/{v_id2 + 1}\n")
        else:
            for v_id0, tex_id0, v_id1, tex_id1, v_id2, tex_id2, in faces_with_texture :
                f.write(f"f {v_id0 + 1}/{tex_id0 + 1} "
                        f"{v_id1 + 1}/{tex_id1 + 1} "
                        f"{v_id2 + 1}/{tex_id2 + 1}\n")

def add_texture_to_obj(obj_file_path, output_file_path, uv_list, mtl_file_name, mat_name):
    # Update OBJ-----------------------------------------------------

    with open(obj_file_path, 'r') as file:
        lines = file.readlines()

    uv_index = 0
    updated_lines = []
    mtllib_exists = False
    inserted = False

    s_and_usemtl_lines = ['s 1\n', f'usemtl {mat_name}\n']

    for line in lines:
        if line.startswith('vt '):
            # Format the new UV coordinates
            uv = uv_list[uv_index]
            new_uv_line = f'vt {uv[0]:.6f} {uv[1]:.6f}\n'
            updated_lines.append(new_uv_line)
            uv_index += 1
        elif line.startswith('mtllib '):
            # Ensure the mtllib line points to the correct MTL file
            new_mtl_line = f'mtllib {mtl_file_name}\n'
            updated_lines.append(new_mtl_line)
            mtllib_exists = True
        elif line.startswith('f') and not inserted:
            # Insert the s and usemtl lines before the first face line
            updated_lines.extend(s_and_usemtl_lines)
            inserted = True
            updated_lines.append(line)
        else:
            updated_lines.append(line)
            
    # If mtllib line does not exist, add it at the beginning
    if not mtllib_exists:
        updated_lines.insert(0, f'mtllib {mtl_file_name}\n')

    with open(output_file_path, 'w') as file:
        file.writelines(updated_lines)

# !SECTION