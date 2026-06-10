import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch

import pygarment.data_config as data_config
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.sim_config import PathCofigNew
from pygarment.meshgen.simulation import run_sim_multi_new, run_sim_new

from pygarment.paths_config import get_path, smplx_model_path_str
from scripts.body_sequence_utils import build_body_sequence
from scripts.garment_spec_utils import (
    classify_panel,
    filtered_spec,
    garment_name_from_spec,
    has_cross_group_stitches,
    label_upper_hem_edges,
    validate_lower_waistband_label,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = ROOT / "precomputed"
DEFAULT_DATASET_ROOT = get_path("dataset_root")
DEFAULT_NEWCLOTH_DIR = ROOT / "assets" / "newcloth"
DEFAULT_POSE_DIR = get_path("pose_dir")
DEFAULT_SIM_CONFIG = ROOT / "assets" / "Sim_props" / "untucked_waistband_fit_sim_props.yaml"
DEFAULT_TEXTURE_BINDINGS = ROOT / "assets" / "newcloth_texture_bindings.json"
TEXTURE_RESOLVER_VERSION = "generated-spec-global-atlas-v5-waist-tint"
# Invalidate cached male body sequences built before beta0 gender correction.
BETA_CORRECTION_VERSION = "male_beta0_invert_v1"

ARTIFACT_ATTRS = (
    "g_box_mesh",
    "g_mesh_segmentation",
    "g_orig_edge_len",
    "g_vert_labels",
    "g_texture_fabric",
    "g_texture",
    "g_mtl",
    "g_specs",
)


@dataclass
class OutfitInput:
    outfit: str
    mode: str
    is_one_piece: bool
    upper_spec: str = ""
    lower_spec: str = ""
    spec_file: str = ""


@dataclass
class BatchTask:
    outfit: str
    mode: str
    is_one_piece: bool
    gender: str
    body_idx: int
    pose_idx: int
    pose_id: str
    pose_file: str
    beta_params: list
    upper_spec: str = ""
    lower_spec: str = ""
    spec_file: str = ""


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def parse_csv(value, cast=str):
    if value is None or value == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def stable_json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_default_body_vertices(gender, alignment_reference="female"):
    if alignment_reference == "gender":
        reference_gender = gender
    else:
        reference_gender = alignment_reference
    body_name = "m_smpl_average_A40.obj" if reference_gender == "male" else "f_smpl_average_A40.obj"
    vertices = []
    with open(ROOT / "assets" / "bodies" / body_name, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
    return np.asarray(vertices, dtype=np.float32)


def smplx_model_path(gender):
    return smplx_model_path_str(gender)


def beta_grid():
    """16-body grid indexed by body_idx (4x4 beta0 x beta1).

    beta0 controls height tier (beta1 is secondary):
    0-3 short, 4-7 medium-short, 8-11 medium-tall, 12-15 tall.
    """
    beta_values = [-1.5, -0.5, 0.5, 1.5]
    return [[b0, b1] + [0.0] * 8 for b0, b1 in itertools.product(beta_values, beta_values)]


def apply_gender_beta_correction(beta_params, gender):
    """SMPL-X beta0 height is inverted for male vs female; flip beta0 for male."""
    beta = list(beta_params)
    if gender == "male" and beta:
        beta[0] = -beta[0]
    return beta


def load_segmentations():
    with open(ROOT / "assets" / "bodies" / "smpl_vert_segmentation.json", "r") as f:
        smpl_body_segmentation = json.load(f)
    with open(ROOT / "assets" / "bodies" / "smplx_body_segmentation.json", "r") as f:
        smplx_body_segmentation = json.load(f)
    return smpl_body_segmentation, smplx_body_segmentation


def prepare_outfit_inputs(spec_file, cache_root):
    spec_file = Path(spec_file)
    outfit = garment_name_from_spec(spec_file)
    spec_hash = sha256_bytes(spec_file.read_bytes())[:16]
    split_dir = Path(cache_root) / "generated_specs" / f"{outfit}_{spec_hash}"
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(spec_file, "r") as f:
        source_spec = json.load(f)
    panels = source_spec["pattern"]["panels"]
    panel_groups = {name: classify_panel(name, panel) for name, panel in panels.items()}
    stitches = source_spec["pattern"].get("stitches", [])

    if is_dress_outfit_name(outfit) or has_cross_group_stitches(stitches, panel_groups):
        return [OutfitInput(outfit=outfit, mode="one_piece", is_one_piece=True, spec_file=str(spec_file))]

    upper_path = split_dir / "Upper_specification.json"
    lower_path = split_dir / "Lower_specification.json"
    if not upper_path.exists() or not lower_path.exists():
        upper_spec = filtered_spec(source_spec, panel_groups, "Upper")
        lower_spec = filtered_spec(source_spec, panel_groups, "Lower")
        label_upper_hem_edges(upper_spec)
        validate_lower_waistband_label(lower_spec)
        with open(upper_path, "w") as f:
            json.dump(upper_spec, f, indent=2)
        with open(lower_path, "w") as f:
            json.dump(lower_spec, f, indent=2)

    return [
        OutfitInput(outfit=outfit, mode="tucked_in", is_one_piece=False,
                    upper_spec=str(upper_path), lower_spec=str(lower_path)),
        OutfitInput(outfit=outfit, mode="untucked", is_one_piece=False,
                    upper_spec=str(upper_path), lower_spec=str(lower_path)),
    ]


def props_for_task(sim_config):
    props = data_config.Properties(str(sim_config))
    props.set_section_stats("sim", fails={}, sim_time={}, spf={}, fin_frame={},
                            body_collisions={}, self_collisions={})
    props.set_section_stats("render", render_time={})
    return props


def texture_binding_hash(uv_config):
    binding_path = uv_config.get("texture_bindings_path") or DEFAULT_TEXTURE_BINDINGS
    binding_path = Path(binding_path)
    if not binding_path.is_absolute():
        binding_path = ROOT / binding_path
    if not binding_path.exists():
        return ""
    return sha256_bytes(binding_path.read_bytes())


def garment_cache_key(spec_path, role, props):
    uv_config = props["render"]["config"]["uv_texture"]
    payload = {
        "role": role,
        "spec": sha256_bytes(Path(spec_path).read_bytes()),
        "resolution_scale": props["sim"]["config"]["resolution_scale"],
        "uv_texture": uv_config,
        "texture_bindings_hash": texture_binding_hash(uv_config),
        "texture_resolver_version": TEXTURE_RESOLVER_VERSION,
    }
    return sha256_bytes(stable_json(payload))[:20]


def lock_dir(path):
    path = Path(path)
    while True:
        try:
            path.mkdir(parents=True)
            return
        except FileExistsError:
            time.sleep(0.5)


def unlock_dir(path):
    try:
        Path(path).rmdir()
    except FileNotFoundError:
        pass


def _garment_cache_artifacts_complete(manifest_path):
    """Return True if manifest exists and every listed artifact file is on disk."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        cache_out_el = Path(manifest["out_el"])
        names = manifest["artifacts"]
    except (OSError, json.JSONDecodeError, KeyError):
        return False
    for attr in ARTIFACT_ATTRS:
        try:
            rel = names[attr]
        except KeyError:
            return False
        if not (cache_out_el / rel).is_file():
            return False
    return True


def ensure_garment_cache(spec_path, role, props, cache_root):
    spec_path = Path(spec_path)
    garment_name = garment_name_from_spec(spec_path)
    key = garment_cache_key(spec_path, role, props)
    cache_dir = Path(cache_root) / "garments" / role / f"{garment_name}_{key}"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and _garment_cache_artifacts_complete(manifest_path):
        return cache_dir

    lock_path = cache_dir.with_name(cache_dir.name + ".lock")
    lock_dir(lock_path)
    try:
        if manifest_path.exists() and _garment_cache_artifacts_complete(manifest_path):
            return cache_dir
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = PathCofigNew(
            in_element_path=spec_path.parent,
            out_path=cache_dir,
            in_name=garment_name,
            out_name=role,
            body_name="cache",
            smpl_body=True,
            add_timestamp=False,
        )
        boxmesh = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
        boxmesh.load()
        boxmesh.serialize(paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"])
        manifest = {
            "spec_path": str(spec_path),
            "role": role,
            "key": key,
            "out_el": str(paths.out_el),
            "artifacts": {attr: Path(getattr(paths, attr)).name for attr in ARTIFACT_ATTRS},
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    finally:
        unlock_dir(lock_path)
    return cache_dir


def copy_text_with_replacements(src, dst, replacements):
    text = Path(src).read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    Path(dst).write_text(text)


def link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def populate_cached_garment(spec_path, role, scratch_out, body_name, props, cache_root):
    spec_path = Path(spec_path)
    garment_name = garment_name_from_spec(spec_path)
    paths = PathCofigNew(
        in_element_path=spec_path.parent,
        out_path=scratch_out,
        in_name=garment_name,
        out_name=role,
        body_name=body_name,
        smpl_body=True,
        add_timestamp=False,
    )
    cache_dir = ensure_garment_cache(spec_path, role, props, cache_root)
    with open(cache_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    cache_out_el = Path(manifest["out_el"])

    paths.out_el.mkdir(parents=True, exist_ok=True)
    cache_mtl = cache_out_el / manifest["artifacts"]["g_mtl"]
    cache_texture = cache_out_el / manifest["artifacts"]["g_texture_fabric"]
    for attr in ARTIFACT_ATTRS:
        src = cache_out_el / manifest["artifacts"][attr]
        dst = Path(getattr(paths, attr))
        if not src.exists():
            if attr == "g_box_mesh":
                raise FileNotFoundError(
                    f"Garment cache is incomplete: missing {src} (role={role}, spec={spec_path}). "
                    "Try deleting the cache folder for this garment under "
                    f"{Path(cache_root) / 'garments' / role!s} or pass a fresh --cache-root."
                )
            continue
        if attr == "g_box_mesh":
            copy_text_with_replacements(src, dst, {cache_mtl.name: paths.g_mtl.name})
        elif attr == "g_mtl":
            copy_text_with_replacements(src, dst, {cache_texture.name: paths.g_texture_fabric.name})
        else:
            link_or_copy(src, dst)
    props.serialize(paths.element_sim_props)
    return garment_name, paths


def body_sequence_cache_path(cache_root, gender, body_name, pose_id):
    return Path(cache_root) / "body_sequences" / gender / body_name / f"{pose_id}.npz"


def load_valid_body_sequence_cache(cache_path, alignment_reference):
    if not Path(cache_path).exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        cached_mode = str(data["start_shape_mode"]) if "start_shape_mode" in data else "morph"
        cached_alignment = str(data["alignment_reference"]) if "alignment_reference" in data else "gender"
        cached_format = str(data["cache_format"]) if "cache_format" in data else "compressed"
        if cached_mode != "morph" or cached_alignment != alignment_reference or cached_format != "uncompressed":
            return None
        if str(data.get("beta_correction_version", "")) != BETA_CORRECTION_VERSION:
            return None
        return data["body_sequence"].astype(np.float32), int(data["dynamic_frames"])


def get_body_sequence(cache_root, gender, body_idx, beta_params, pose_id, pose_file, model,
                      smpl_body_segmentation, smplx_body_segmentation, default_body_vertices,
                      device, alignment_reference="female"):
    body_name = f"{gender}{body_idx}"
    cache_path = body_sequence_cache_path(cache_root, gender, body_name, pose_id)
    cached = load_valid_body_sequence_cache(cache_path, alignment_reference)
    if cached is not None:
        return cached

    smpl_betas = apply_gender_beta_correction(beta_params, gender)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    lock_dir(lock_path)
    try:
        cached = load_valid_body_sequence_cache(cache_path, alignment_reference)
        if cached is not None:
            return cached

        body_sequence = build_body_sequence(
            model,
            np.asarray(smpl_betas, dtype=np.float32),
            device,
            smpl_body_segmentation,
            smplx_body_segmentation,
            default_body_vertices,
            pose_file=Path(pose_file),
        ).astype(np.float32)
        tmp_path = cache_path.with_suffix(".tmp.npz")
        np.savez(
            tmp_path,
            body_sequence=body_sequence,
            dynamic_frames=np.asarray([len(body_sequence)], dtype=np.int32),
            beta_params=np.asarray(smpl_betas, dtype=np.float32),
            pose_file=np.asarray(str(pose_file)),
            gender=np.asarray(gender),
            start_shape_mode=np.asarray("morph"),
            alignment_reference=np.asarray(alignment_reference),
            cache_format=np.asarray("uncompressed"),
            beta_correction_version=np.asarray(BETA_CORRECTION_VERSION),
        )
        os.replace(tmp_path, cache_path)
        return body_sequence, len(body_sequence)
    finally:
        unlock_dir(lock_path)


def task_body_cache_key(task, alignment_reference):
    return (task.gender, task.body_idx, task.pose_id, alignment_reference)


def get_worker_body_sequence(task, args, worker_state, model, segmentations, default_body_vertices):
    key = task_body_cache_key(task, args["alignment_reference"])
    cached = worker_state.get("body_sequence_cache")
    if cached is not None and cached.get("key") == key:
        return cached["body_sequence"], cached["dynamic_frames"]

    smpl_body_segmentation, smplx_body_segmentation = segmentations
    body_sequence, dynamic_frames = get_body_sequence(
        args["cache_root"], task.gender, task.body_idx, task.beta_params, task.pose_id, task.pose_file,
        model, smpl_body_segmentation, smplx_body_segmentation, default_body_vertices, worker_state["device"],
        alignment_reference=args["alignment_reference"])
    worker_state["body_sequence_cache"] = {
        "key": key,
        "body_sequence": body_sequence,
        "dynamic_frames": dynamic_frames,
    }
    return body_sequence, dynamic_frames


def expected_outputs(output_dir, sides):
    outputs = [Path(output_dir) / f"render_{side}.png" for side in sides]
    if "front" in sides:
        outputs.extend([
            Path(output_dir) / "garment_mask_front.png",
            Path(output_dir) / "body_mask_front.png",
        ])
    return outputs


def output_complete(output_dir, sides):
    return all(path.exists() for path in expected_outputs(output_dir, sides))


def copy_outputs_from_scratch(src_dir, src_name, dst_dir, sides):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)
    for side in sides:
        src = src_dir / f"{src_name}_render_{side}.png"
        if src.exists():
            shutil.copy2(src, dst_dir / f"render_{side}.png")
    if "front" in sides:
        mask_src = src_dir / f"{src_name}_render_front_mask.png"
        body_src = src_dir / f"{src_name}_render_front_bodymask.png"
        if mask_src.exists():
            shutil.copy2(mask_src, dst_dir / "garment_mask_front.png")
        if body_src.exists():
            shutil.copy2(body_src, dst_dir / "body_mask_front.png")


def copy_single_outputs(paths, dst_dir, sides):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for side in sides:
        src = paths.render_path(side)
        if src.exists():
            shutil.copy2(src, dst_dir / f"render_{side}.png")
    if "front" in sides:
        front = paths.render_path("front")
        mask_src = front.with_name(front.stem + "_mask.png")
        body_src = front.with_name(front.stem + "_bodymask.png")
        if mask_src.exists():
            shutil.copy2(mask_src, dst_dir / "garment_mask_front.png")
        if body_src.exists():
            shutil.copy2(body_src, dst_dir / "body_mask_front.png")


def task_output_dir(dataset_root, task):
    return Path(dataset_root) / task.gender / f"{task.gender}{task.body_idx}" / task.outfit / task.mode / task.pose_id


def run_task(task, args, worker_state):
    props = props_for_task(args["sim_config"])
    render_sides = props["render"]["config"].get("sides", ["front", "back"])
    final_dir = task_output_dir(args["dataset_root"], task)
    if not args["force"] and output_complete(final_dir, render_sides):
        return {"status": "skipped", "task": task.__dict__, "output_dir": str(final_dir)}

    gpu_id = args["gpu_id"]
    device = worker_state["device"]
    gender = task.gender
    if gender not in worker_state["models"]:
        worker_state["models"][gender] = smplx.SMPLX(
            smplx_model_path(gender), gender=gender, use_pca=False).to(device)
    alignment_key = (gender, args["alignment_reference"])
    if alignment_key not in worker_state["default_vertices"]:
        worker_state["default_vertices"][alignment_key] = read_default_body_vertices(
            gender, args["alignment_reference"])
    model = worker_state["models"][gender]
    body_faces = model.faces
    default_body_vertices = worker_state["default_vertices"][alignment_key]

    body_name = f"{gender}{task.body_idx}"
    body_sequence, dynamic_frames = get_worker_body_sequence(
        task, args, worker_state, model, worker_state["segmentations"], default_body_vertices)

    task_id = f"{gender}_{task.body_idx}_{task.outfit}_{task.mode}_{task.pose_id}"
    scratch_out = Path(args["cache_root"]) / "runs" / task_id
    if scratch_out.exists():
        shutil.rmtree(scratch_out)
    scratch_out.mkdir(parents=True, exist_ok=True)

    try:
        if task.is_one_piece:
            garment_name, paths = populate_cached_garment(
                task.spec_file, "Garment", scratch_out, body_name, props, args["cache_root"])
            run_sim_new(
                garment_name,
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
            copy_single_outputs(paths, final_dir, render_sides)
        else:
            upper_name, upper_paths = populate_cached_garment(
                task.upper_spec, "Upper", scratch_out, body_name, props, args["cache_root"])
            lower_name, lower_paths = populate_cached_garment(
                task.lower_spec, "Lower", scratch_out, body_name, props, args["cache_root"])
            tucked_in = task.mode == "tucked_in"
            if tucked_in:
                ordered_names = [upper_name, lower_name]
                ordered_paths = [upper_paths, lower_paths]
            else:
                ordered_names = [lower_name, upper_name]
                ordered_paths = [lower_paths, upper_paths]
            render_dir = scratch_out / "combined" / body_name
            render_name = f"{task.outfit}__{task.mode}__{task.pose_id}"
            run_sim_multi_new(
                ordered_names,
                props,
                ordered_paths,
                first_garment_index=0,
                body_sequence=body_sequence,
                body_faces=body_faces,
                save_v_norms=False,
                store_usd=False,
                optimize_storage=False,
                verbose=False,
                dynamic_frames=dynamic_frames,
                gpu_id=gpu_id,
                combined_render_dir=render_dir,
                combined_render_name=render_name,
                waistband_side_mode="inside" if tucked_in else "outside",
            )
            copy_outputs_from_scratch(render_dir, render_name, final_dir, render_sides)
    finally:
        if not args["keep_failed_workdir"] and scratch_out.exists():
            shutil.rmtree(scratch_out)

    if not output_complete(final_dir, render_sides):
        raise RuntimeError(f"missing expected outputs in {final_dir}")
    return {"status": "done", "task": task.__dict__, "output_dir": str(final_dir)}


def worker_loop(gpu_id, task_queue, result_queue, args):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    state = {
        "device": device,
        "models": {},
        "default_vertices": {},
        "segmentations": load_segmentations(),
        "body_sequence_cache": None,
    }
    args = dict(args)
    args["gpu_id"] = gpu_id
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            result = run_task(task, args, state)
            result["gpu_id"] = gpu_id
            result_queue.put(result)
        except Exception as exc:
            result_queue.put({
                "status": "failed",
                "gpu_id": gpu_id,
                "task": task.__dict__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })


def enumerate_tasks(args):
    cache_root = Path(args.cache_root)
    outfit_specs = sorted(Path(args.newcloth_dir).glob("*_specification.json"))
    if args.outfit_filter:
        filters = set(parse_csv(args.outfit_filter))
        outfit_specs = [path for path in outfit_specs if garment_name_from_spec(path) in filters]
    if args.outfit_limit is not None:
        outfit_specs = outfit_specs[:args.outfit_limit]

    outfits = []
    for spec in outfit_specs:
        outfits.extend(prepare_outfit_inputs(spec, cache_root))

    poses = sorted(Path(args.pose_dir).glob("*.npz"))
    if args.pose_limit is not None:
        poses = poses[:args.pose_limit]
    if not poses:
        raise ValueError(f"No .npz pose files found in {args.pose_dir}")

    genders = parse_csv(args.genders)
    betas = beta_grid()[:args.num_bodies]
    tasks = []
    skipped_male_outfits = set()
    for outfit, gender, (body_idx, beta_params), (pose_idx, pose_file) in itertools.product(
            outfits, genders, enumerate(betas), enumerate(poses)):
        if gender == "male" and is_skirt_or_dress_outfit(outfit):
            skipped_male_outfits.add(outfit.outfit)
            continue
        pose_id = f"pose{pose_idx}"
        tasks.append(BatchTask(
            outfit=outfit.outfit,
            mode=outfit.mode,
            is_one_piece=outfit.is_one_piece,
            gender=gender,
            body_idx=body_idx,
            pose_idx=pose_idx,
            pose_id=pose_id,
            pose_file=str(pose_file),
            beta_params=beta_params,
            upper_spec=outfit.upper_spec,
            lower_spec=outfit.lower_spec,
            spec_file=outfit.spec_file,
        ))
    if skipped_male_outfits:
        print(
            "Skipping skirt/dress outfits for male: "
            + ", ".join(sorted(skipped_male_outfits)),
            flush=True,
        )
    tasks.sort(key=task_sort_key)
    return tasks


def task_sort_key(task):
    return (
        task.gender,
        task.body_idx,
        task.pose_idx,
        task.outfit,
        task.mode,
    )


def is_skirt_or_dress_outfit(outfit):
    outfit_name = outfit.outfit.lower()
    return outfit.is_one_piece or is_dress_outfit_name(outfit_name) or "skirt" in outfit_name


def is_dress_outfit_name(outfit_name):
    return outfit_name.lower().startswith("dress")


def write_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def unique_body_sequence_tasks(tasks, alignment_reference):
    seen = set()
    unique = []
    for task in tasks:
        key = task_body_cache_key(task, alignment_reference)
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    return unique


def warm_body_sequence_cache(args, tasks):
    unique_tasks = unique_body_sequence_tasks(tasks, args.alignment_reference)
    if not unique_tasks:
        return

    gpus = parse_csv(args.gpus, int)
    device = torch.device(
        f"cuda:{gpus[0]}" if gpus and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    segmentations = load_segmentations()
    smpl_body_segmentation, smplx_body_segmentation = segmentations
    models = {}
    default_vertices = {}
    print(f"Precomputing {len(unique_tasks)} body sequences on {device}", flush=True)
    for index, task in enumerate(unique_tasks, start=1):
        if task.gender not in models:
            models[task.gender] = smplx.SMPLX(
                smplx_model_path(task.gender), gender=task.gender, use_pca=False).to(device)
        alignment_key = (task.gender, args.alignment_reference)
        if alignment_key not in default_vertices:
            default_vertices[alignment_key] = read_default_body_vertices(
                task.gender, args.alignment_reference)

        get_body_sequence(
            args.cache_root,
            task.gender,
            task.body_idx,
            task.beta_params,
            task.pose_id,
            task.pose_file,
            models[task.gender],
            smpl_body_segmentation,
            smplx_body_segmentation,
            default_vertices[alignment_key],
            device,
            alignment_reference=args.alignment_reference,
        )
        if index == len(unique_tasks) or index % 10 == 0:
            print(f"Precomputed body sequences: {index}/{len(unique_tasks)}", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--newcloth-dir", default=str(DEFAULT_NEWCLOTH_DIR))
    parser.add_argument("--pose-dir", default=str(DEFAULT_POSE_DIR))
    parser.add_argument("--sim-config", default=str(DEFAULT_SIM_CONFIG))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--genders", default="female,male")
    parser.add_argument("--num-bodies", type=int, default=16)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--outfit-filter", default=None)
    parser.add_argument("--outfit-limit", type=int, default=None)
    parser.add_argument("--pose-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-failed-workdir", action="store_true")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--alignment-reference", choices=("female", "male", "gender"), default="female",
                        help="Body OBJ used as the vertical garment-alignment reference. Default keeps garment coordinates on the female GarmentCode basis.")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    (cache_root / "garments").mkdir(parents=True, exist_ok=True)
    (cache_root / "body_sequences").mkdir(parents=True, exist_ok=True)
    Path(args.dataset_root).mkdir(parents=True, exist_ok=True)

    tasks = enumerate_tasks(args)
    print(f"Prepared {len(tasks)} tasks", flush=True)

    # Warm garment cache once in the parent process so GPU workers do not all race on BoxMesh generation.
    warm_props = props_for_task(args.sim_config)
    seen_specs = set()
    for task in tasks:
        if task.is_one_piece:
            seen_specs.add((task.spec_file, "Garment"))
        else:
            seen_specs.add((task.upper_spec, "Upper"))
            seen_specs.add((task.lower_spec, "Lower"))
    seen_specs = sorted(seen_specs)
    print(f"Warming {len(seen_specs)} garment cache entries", flush=True)
    for index, (spec_path, role) in enumerate(seen_specs, start=1):
        print(
            f"[garment-cache {index}/{len(seen_specs)}] {role} {garment_name_from_spec(Path(spec_path))}",
            flush=True,
        )
        ensure_garment_cache(spec_path, role, warm_props, args.cache_root)
    if args.precompute_only:
        warm_body_sequence_cache(args, tasks)
        print("Precompute finished.", flush=True)
        return

    gpus = parse_csv(args.gpus, int)
    total_workers = max(len(gpus) * max(args.workers_per_gpu, 1), 1)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker_args = {
        "cache_root": str(cache_root),
        "dataset_root": str(args.dataset_root),
        "sim_config": str(args.sim_config),
        "force": bool(args.force),
        "keep_failed_workdir": bool(args.keep_failed_workdir),
        "alignment_reference": args.alignment_reference,
    }
    workers = []
    for gpu_id in gpus:
        for _ in range(max(args.workers_per_gpu, 1)):
            proc = ctx.Process(target=worker_loop, args=(gpu_id, task_queue, result_queue, worker_args))
            proc.start()
            workers.append(proc)
    print(f"Started {len(workers)} workers on GPUs {gpus}", flush=True)

    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)

    done = 0
    results_path = cache_root / "batch_results.jsonl"
    while done < len(tasks):
        try:
            result = result_queue.get(timeout=30)
        except queue.Empty:
            alive = [proc for proc in workers if proc.is_alive()]
            failed_workers = [proc for proc in workers if proc.exitcode not in (None, 0)]
            if failed_workers and not alive:
                raise RuntimeError(
                    f"All workers exited before finishing tasks. "
                    f"done={done}/{len(tasks)}, failed_workers={len(failed_workers)}"
                )
            print(
                f"Waiting for workers... done={done}/{len(tasks)}, "
                f"alive={len(alive)}, failed_workers={len(failed_workers)}",
                flush=True,
            )
            continue
        done += 1
        write_jsonl(results_path, result)
        print(f"[{done}/{len(tasks)}] {result['status']} gpu={result.get('gpu_id')} "
              f"{result['task']['gender']}{result['task']['body_idx']} "
              f"{result['task']['outfit']} {result['task']['mode']} {result['task']['pose_id']}",
              flush=True)

    failed = 0
    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            failed += 1
    if failed:
        print(f"WARNING: {failed} workers exited with non-zero status", flush=True)
    print(f"Results log: {results_path}", flush=True)


if __name__ == "__main__":
    main()
