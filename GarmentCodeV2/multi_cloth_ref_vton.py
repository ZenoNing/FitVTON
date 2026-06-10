import argparse
import json
import multiprocessing as mp
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import smplx
import torch

from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.sim_config import PathCofigNew
from pygarment.meshgen.simulation import run_sim_new

from scripts.body_sequence_utils import align_smplx_body_y_axis, generate_smooth_pose_sequence
from scripts.garment_spec_utils import (
    classify_panel,
    filtered_spec,
    garment_name_from_spec,
    label_upper_hem_edges,
    validate_lower_waistband_label,
)
from multi_cloth_batch_vton import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_NEWCLOTH_DIR,
    DEFAULT_SIM_CONFIG,
    load_segmentations,
    props_for_task,
    read_default_body_vertices,
    smplx_model_path,
)


ROOT = Path(__file__).resolve().parent


@dataclass
class RefTask:
    unit_name: str
    spec_path: str
    gpu_id: int


def parse_csv(value, cast=str):
    if value is None or value == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def source_outfit_units(outfit_name):
    if outfit_name.startswith("dress"):
        return outfit_name, None
    match = re.match(r"^(upper\d+)_(.+)$", outfit_name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def write_ref_spec(source_spec_path, unit_name, unit_role, output_path):
    with open(source_spec_path, "r") as f:
        source_spec = json.load(f)
    panels = source_spec["pattern"]["panels"]
    panel_groups = {name: classify_panel(name, panel) for name, panel in panels.items()}
    unit_spec = filtered_spec(source_spec, panel_groups, unit_role)
    if unit_role == "Upper":
        label_upper_hem_edges(unit_spec)
    else:
        validate_lower_waistband_label(unit_spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unit_spec, f, indent=2)
    return output_path


def prepare_ref_specs(newcloth_dir, cache_root):
    newcloth_dir = Path(newcloth_dir)
    ref_spec_root = Path(cache_root) / "generated_ref_specs"
    units = {}

    for spec_path in sorted(newcloth_dir.glob("*_specification.json")):
        outfit_name = garment_name_from_spec(spec_path)
        upper_id, lower_id = source_outfit_units(outfit_name)
        if upper_id is not None and lower_id is None:
            units.setdefault(upper_id, spec_path)
            continue
        if upper_id is None:
            continue

        if upper_id not in units:
            out_path = ref_spec_root / upper_id / f"{upper_id}_specification.json"
            if not out_path.exists():
                write_ref_spec(spec_path, upper_id, "Upper", out_path)
            units[upper_id] = out_path

        if lower_id not in units:
            out_path = ref_spec_root / lower_id / f"{lower_id}_specification.json"
            if not out_path.exists():
                write_ref_spec(spec_path, lower_id, "Lower", out_path)
            units[lower_id] = out_path

    expected = (
        [f"dress{i}" for i in range(1, 9)]
        + [f"upper{i}" for i in range(1, 4)]
        + [f"pants{i}" for i in range(1, 5)]
        + [f"circleskirt{i}" for i in range(1, 3)]
        + [f"pencilskirt{i}" for i in range(1, 3)]
    )

    for spec_path in sorted(newcloth_dir.glob("dress*_specification.json")):
        units[garment_name_from_spec(spec_path)] = spec_path

    missing = [unit for unit in expected if unit not in units]
    if missing:
        raise RuntimeError(f"Missing reference specs for units: {', '.join(missing)}")

    return [(unit, Path(units[unit])) for unit in expected]


def build_ref_body_sequence(smpl_model, device):
    smpl_body_segmentation, smplx_body_segmentation = load_segmentations()
    default_body_vertices = read_default_body_vertices("female", "female")

    pose_params_start = torch.zeros([1, 165], device=device).float()
    pose_params_start[:, 16 * 3 + 2] = -0.6
    pose_params_start[:, 17 * 3 + 2] = 0.6

    pose_params_end = torch.zeros([1, 165], device=device).float()
    pose_params_end[:, 16 * 3 + 2] = -1.1
    pose_params_end[:, 17 * 3 + 2] = 1.1

    beta_params = torch.zeros([1, 10], dtype=torch.float32, device=device)
    beta_params[0, 0] = 1.0

    body_sequence = generate_smooth_pose_sequence(
        smpl_model, pose_params_start, pose_params_end, beta_params,
        threshold=0.0025, device=device
    )
    body_sequence = np.stack(body_sequence, axis=0)
    body_sequence = align_smplx_body_y_axis(
        body_sequence,
        smpl_body_segmentation,
        smplx_body_segmentation,
        default_body_vertices,
    )
    return body_sequence.astype(np.float32), len(body_sequence)


def copy_ref_outputs(paths, final_dir, sides):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for side in sides:
        src = paths.render_path(side)
        if src.exists():
            center_garment_bbox(src, final_dir / f"render_{side}.png")


def center_garment_bbox(src_path, dst_path):
    image = Image.open(src_path).convert("RGBA")
    arr = np.asarray(image)
    alpha_mask = arr[..., 3] > 0
    if alpha_mask.any():
        mask = alpha_mask
    else:
        mask = np.any(arr[..., :3] < 245, axis=2)
    if not mask.any():
        shutil.copy2(src_path, dst_path)
        return

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = image.crop((x0, y0, x1, y1))

    width, height = image.size
    crop_width, crop_height = crop.size
    paste_x = max((width - crop_width) // 2, 0)
    paste_y = max((height - crop_height) // 2, 0)

    canvas = Image.new("RGBA", image.size, (255, 255, 255, 0))
    canvas.alpha_composite(crop, (paste_x, paste_y))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst_path)


def ref_output_complete(final_dir, sides):
    final_dir = Path(final_dir)
    return all((final_dir / f"render_{side}.png").exists() for side in sides)


def ref_camera_location(unit_name):
    if unit_name.startswith("upper"):
        return [0.0, 1.1, 2.0]
    if unit_name == "circleskirt2":
        return [0.0, 0.6, 3.4]
    if unit_name.startswith(("pants", "circleskirt", "pencilskirt")):
        return [0.0, 0.6, 2.6]
    return [0.0, 0.85, 3.35]


def run_ref_task(args_tuple):
    task, args = args_tuple
    gpu_id = task.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id) if torch.cuda.is_available() else None

    props = props_for_task(args["sim_config"])
    render_config = props["render"]["config"]
    render_config["hide_body"] = True
    render_config["front_camera_location"] = ref_camera_location(task.unit_name)
    sides = render_config.get("sides", ["front", "back"])

    final_dir = Path(args["dataset_root"]) / "Ref" / task.unit_name
    if not args["force"] and ref_output_complete(final_dir, sides):
        return f"skipped {task.unit_name}"

    scratch_dir = Path(args["cache_root"]) / "ref_workdirs" / task.unit_name
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    smpl_model = smplx.SMPLX(smplx_model_path("female"), gender="female", use_pca=False).to(device)
    body_faces = smpl_model.faces
    body_sequence, dynamic_frames = build_ref_body_sequence(smpl_model, device)

    paths = PathCofigNew(
        in_element_path=Path(task.spec_path).parent,
        out_path=scratch_dir,
        in_name=task.unit_name,
        out_name="Ref",
        body_name="",
        smpl_body=True,
        add_timestamp=False,
    )

    garment_box_mesh = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
    garment_box_mesh.load()
    garment_box_mesh.serialize(paths, store_panels=False, uv_config=render_config["uv_texture"])
    props.serialize(paths.element_sim_props)

    run_sim_new(
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
    )
    props.serialize(paths.element_sim_props)
    copy_ref_outputs(paths, final_dir, sides)

    if not args["keep_workdir"]:
        shutil.rmtree(scratch_dir)
    return f"done {task.unit_name} gpu={gpu_id}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--newcloth-dir", default=str(DEFAULT_NEWCLOTH_DIR))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--sim-config", default=str(DEFAULT_SIM_CONFIG))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--unit-filter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    gpus = parse_csv(args.gpus, int)
    units = prepare_ref_specs(args.newcloth_dir, args.cache_root)
    if args.unit_filter:
        keep = set(parse_csv(args.unit_filter))
        units = [(unit, path) for unit, path in units if unit in keep]
    if args.limit is not None:
        units = units[:args.limit]

    tasks = [
        RefTask(unit_name=unit, spec_path=str(path), gpu_id=gpus[i % len(gpus)])
        for i, (unit, path) in enumerate(units)
    ]
    print(f"Prepared {len(tasks)} reference tasks", flush=True)

    worker_args = {
        "dataset_root": str(args.dataset_root),
        "cache_root": str(args.cache_root),
        "sim_config": str(args.sim_config),
        "force": bool(args.force),
        "keep_workdir": bool(args.keep_workdir),
    }

    if args.debug or len(tasks) <= 1:
        for i, task in enumerate(tasks, start=1):
            print(f"[{i}/{len(tasks)}] {run_ref_task((task, worker_args))}", flush=True)
        return

    num_workers = max(len(gpus) * max(args.workers_per_gpu, 1), 1)
    with mp.get_context("spawn").Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(run_ref_task, [(task, worker_args) for task in tasks]), start=1):
            print(f"[{i}/{len(tasks)}] {result}", flush=True)


if __name__ == "__main__":
    main()
