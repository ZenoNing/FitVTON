import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrender
import smplx
import torch
import trimesh
from PIL import Image

from pygarment.meshgen.render.pythonrender import create_camera, create_lights
from pygarment.meshgen.simulation import run_sim_multi_new, run_sim_new
from pygarment.paths_config import get_path
from multi_cloth_batch_vton import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_NEWCLOTH_DIR,
    DEFAULT_POSE_DIR,
    DEFAULT_SIM_CONFIG,
    apply_gender_beta_correction,
    beta_grid,
    copy_outputs_from_scratch,
    copy_single_outputs,
    get_body_sequence,
    load_segmentations,
    populate_cached_garment,
    prepare_outfit_inputs,
    props_for_task,
    read_default_body_vertices,
    smplx_model_path,
)
from scripts.body_sequence_utils import build_body_sequence

DEFAULT_DEMO_ROOT = get_path("demo_root")


@dataclass
class DemoTask:
    outfit: str
    mode: str
    is_one_piece: bool
    gender: str
    body_idx: int
    pose_id: str
    pose_file: str
    beta_params: list
    upper_spec: str = ""
    lower_spec: str = ""
    spec_file: str = ""


def sample_id(task):
    return f"{task.gender}__{task.gender}{task.body_idx}__{task.outfit}__{task.mode}__{task.pose_id}"


def copy_frames(src_frames, dst_frames):
    src_frames = Path(src_frames)
    if not src_frames.exists():
        return
    dst_frames = Path(dst_frames)
    dst_frames.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_frames.glob("*.png")):
        shutil.copy2(src, dst_frames / src.name)


def front_camera_location(task):
    if task.gender == "female" and task.outfit == "dress1" and task.pose_id == "pose5":
        return [0.0, 0.78, 4.15]
    return None


def render_body_apose(body_vertices, body_faces, output_path, render_config):
    view_width, view_height = render_config.get("resolution", [768, 1024])
    body_mesh = trimesh.Trimesh(body_vertices, body_faces)
    body_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(0.55, 0.50, 0.45, 1.0),
    )
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)
    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 0.0), ambient_light=(0.12, 0.12, 0.12))
    scene.add(pyrender_body_mesh)
    create_camera(
        pyrender,
        pyrender_body_mesh,
        scene,
        "front",
        camera_location=render_config.get("front_camera_location"),
    )
    create_lights(scene, intensity=12.0)

    renderer = None
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(output_path, "PNG")
    finally:
        if renderer is not None:
            renderer.delete()


def parse_sample_id(sample_id):
    parts = sample_id.split("__")
    if len(parts) != 5:
        raise ValueError(
            "Expected <gender>__<gender><body_idx>__<outfit>__<mode>__pose<pose_idx>"
        )
    gender, body_name, outfit_name, mode, pose_id = parts
    if not body_name.startswith(gender):
        raise ValueError(f"Body token {body_name!r} must start with gender {gender!r}")
    if not pose_id.startswith("pose"):
        raise ValueError(f"Pose token {pose_id!r} must start with 'pose'")
    return gender, int(body_name[len(gender):]), outfit_name, mode, int(pose_id[4:])


def build_tasks(newcloth_dir, pose_dir, cache_root, sample_id=None, compare_modes=False):
    pose_files = sorted(Path(pose_dir).glob("*.npz"))
    outfit_inputs = {}
    for spec in sorted(Path(newcloth_dir).glob("*_specification.json")):
        for outfit in prepare_outfit_inputs(spec, cache_root):
            outfit_inputs[(outfit.outfit, outfit.mode)] = outfit

    betas = beta_grid()
    if sample_id:
        gender, body_idx, outfit_name, mode, pose_idx = parse_sample_id(sample_id)
        modes = [mode]
        if compare_modes and mode in ("tucked_in", "untucked"):
            modes = ["tucked_in", "untucked"]
        selected = [(gender, body_idx, outfit_name, item_mode, pose_idx) for item_mode in modes]
    else:
        selected = [
            ("male", 11, "upper1_pants4", "tucked_in", 0),
            ("female", 0, "dress1", "one_piece", 5),
        ]

    tasks = []
    for gender, body_idx, outfit_name, mode, pose_idx in selected:
        if pose_idx >= len(pose_files):
            raise RuntimeError(f"Need pose{pose_idx} in {pose_dir}, found {len(pose_files)} pose files")
        if body_idx >= len(betas):
            raise RuntimeError(f"Need body index {body_idx}, beta grid has {len(betas)} bodies")
        outfit = outfit_inputs[(outfit_name, mode)]
        tasks.append(DemoTask(
            outfit=outfit.outfit,
            mode=outfit.mode,
            is_one_piece=outfit.is_one_piece,
            gender=gender,
            body_idx=body_idx,
            pose_id=f"pose{pose_idx}",
            pose_file=str(pose_files[pose_idx]),
            beta_params=betas[body_idx],
            upper_spec=outfit.upper_spec,
            lower_spec=outfit.lower_spec,
            spec_file=outfit.spec_file,
        ))
    return tasks


def run_demo(task, args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    props = props_for_task(args.sim_config)
    if args.disable_limb_end_stop:
        props["sim"]["config"]["options"]["enable_limb_end_stop_constraint"] = False

    render_config = props["render"]["config"]
    render_config["sides"] = ["front"]
    camera = front_camera_location(task)
    if camera is not None:
        render_config["front_camera_location"] = camera

    segmentations = load_segmentations()
    model = smplx.SMPLX(smplx_model_path(task.gender), gender=task.gender, use_pca=False).to(device)
    default_body_vertices = read_default_body_vertices(task.gender, args.alignment_reference)
    body_sequence, dynamic_frames = get_body_sequence(
        args.cache_root,
        task.gender,
        task.body_idx,
        task.beta_params,
        task.pose_id,
        task.pose_file,
        model,
        segmentations[0],
        segmentations[1],
        default_body_vertices,
        device,
        alignment_reference=args.alignment_reference,
    )

    out_id = sample_id(task)
    scratch_out = Path(args.cache_root) / "demo_runs" / out_id
    final_dir = Path(args.demo_root) / out_id
    if scratch_out.exists():
        shutil.rmtree(scratch_out)
    if final_dir.exists() and args.force:
        shutil.rmtree(final_dir)
    scratch_out.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    apose_sequence = build_body_sequence(
        model,
        np.asarray(apply_gender_beta_correction(task.beta_params, task.gender), dtype=np.float32),
        device,
        segmentations[0],
        segmentations[1],
        default_body_vertices,
        pose_file=None,
    )
    render_body_apose(
        apose_sequence[-1],
        model.faces,
        final_dir / "frames" / "00000_body_apose_render_front.png",
        render_config,
    )

    body_name = f"{task.gender}{task.body_idx}"
    if task.is_one_piece:
        garment_name, paths = populate_cached_garment(
            task.spec_file, "Garment", scratch_out, body_name, props, args.cache_root)
        run_sim_new(
            garment_name,
            props,
            paths,
            body_sequence=body_sequence,
            body_faces=model.faces,
            save_v_norms=False,
            store_usd=False,
            optimize_storage=False,
            verbose=False,
            dynamic_frames=dynamic_frames,
            gpu_id=args.gpu_id,
            render_each_frame=True,
            render_stride=args.render_stride,
            render_each_frame_include_masks=False,
        )
        copy_single_outputs(paths, final_dir, ["front"])
        copy_frames(paths.out_el / "frames", final_dir / "frames")
    else:
        upper_name, upper_paths = populate_cached_garment(
            task.upper_spec, "Upper", scratch_out, body_name, props, args.cache_root)
        lower_name, lower_paths = populate_cached_garment(
            task.lower_spec, "Lower", scratch_out, body_name, props, args.cache_root)
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
            body_faces=model.faces,
            save_v_norms=False,
            store_usd=False,
            optimize_storage=False,
            verbose=False,
            dynamic_frames=dynamic_frames,
            gpu_id=args.gpu_id,
            combined_render_dir=render_dir,
            combined_render_name=render_name,
            waistband_side_mode="inside" if tucked_in else "outside",
            render_each_frame=True,
            render_stride=args.render_stride,
            render_each_frame_include_masks=False,
        )
        copy_outputs_from_scratch(render_dir, render_name, final_dir, ["front"])
        copy_frames(render_dir / "frames", final_dir / "frames")

    if not args.keep_workdir:
        shutil.rmtree(scratch_out)
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Run a small set of frame-by-frame garment demos.")
    parser.add_argument("--newcloth-dir", default=str(DEFAULT_NEWCLOTH_DIR))
    parser.add_argument("--pose-dir", default=str(DEFAULT_POSE_DIR))
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--sim-config", default=str(DEFAULT_SIM_CONFIG))
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--render-stride", type=int, default=20)
    parser.add_argument("--alignment-reference", choices=("female", "male", "gender"), default="female")
    parser.add_argument("--sample-id", default=None,
                        help="e.g. female__female0__dress1__one_piece__pose5")
    parser.add_argument("--compare-modes", action="store_true",
                        help="With --sample-id, run both tucked_in and untucked")
    parser.add_argument("--disable-limb-end-stop", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(
        args.newcloth_dir,
        args.pose_dir,
        args.cache_root,
        sample_id=args.sample_id,
        compare_modes=args.compare_modes,
    )
    for index, task in enumerate(tasks, start=1):
        out_id = sample_id(task)
        print(f"[{index}/{len(tasks)}] running {out_id}", flush=True)
        final_dir = run_demo(task, args)
        print(f"saved {out_id} to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
