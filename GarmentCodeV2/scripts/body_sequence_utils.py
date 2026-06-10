import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SMPLX_SEGMENTATION_PATH = ROOT / "assets" / "bodies" / "smplx_body_segmentation.json"

POSE_PARAM_SIZES = {
    "global_orient": 3,
    "body_pose": 63,
    "left_hand_pose": 45,
    "right_hand_pose": 45,
    "jaw_pose": 3,
    "leye_pose": 3,
    "reye_pose": 3,
}


def split_pose_params(pose_params):
    global_orient = pose_params[:, :3]
    body_pose = pose_params[:, 3:66]
    left_hand_pose = pose_params[:, 66:111]
    right_hand_pose = pose_params[:, 111:156]
    jaw_pose = pose_params[:, 156:159]
    leye_pose = pose_params[:, 159:162]
    reye_pose = pose_params[:, 162:165]
    return global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose


def _non_hand_vertex_indices(vert_start, vert_end):
    with open(SMPLX_SEGMENTATION_PATH, "r") as f:
        smplx_body_segmentation = json.load(f)
    hand_indices = set(smplx_body_segmentation.get("leftHand", []) +
                       smplx_body_segmentation.get("rightHand", []) +
                       smplx_body_segmentation.get("leftHandIndex1", []) +
                       smplx_body_segmentation.get("rightHandIndex1", []))
    all_indices = np.arange(vert_start.shape[0])
    return np.setdiff1d(all_indices, list(hand_indices))


def generate_smooth_shape_sequence(
    smpl_model, pose_params, beta_params_start, beta_params_end,
    threshold=0.01, device="cpu", max_depth=15, depth=0,
):
    global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose = split_pose_params(pose_params)
    vert_start = smpl_model.forward(
        global_orient=global_orient,
        body_pose=body_pose.reshape(1, -1, 3),
        left_hand_pose=left_hand_pose.reshape(1, -1, 3),
        right_hand_pose=right_hand_pose.reshape(1, -1, 3),
        betas=beta_params_start,
    ).vertices[0].detach().cpu().numpy()
    vert_end = smpl_model.forward(
        global_orient=global_orient,
        body_pose=body_pose.reshape(1, -1, 3),
        left_hand_pose=left_hand_pose.reshape(1, -1, 3),
        right_hand_pose=right_hand_pose.reshape(1, -1, 3),
        betas=beta_params_end,
    ).vertices[0].detach().cpu().numpy()

    non_hand_indices = _non_hand_vertex_indices(vert_start, vert_end)
    max_delta = np.max(np.linalg.norm(vert_end[non_hand_indices] - vert_start[non_hand_indices], axis=1))
    if max_delta < threshold or depth >= max_depth:
        return [vert_start, vert_end]

    beta_params_mid = (beta_params_start + beta_params_end) / 2.0
    first_half = generate_smooth_shape_sequence(
        smpl_model, pose_params, beta_params_start, beta_params_mid, threshold, device, max_depth, depth + 1)
    second_half = generate_smooth_shape_sequence(
        smpl_model, pose_params, beta_params_mid, beta_params_end, threshold, device, max_depth, depth + 1)
    return first_half[:-1] + second_half


def generate_smooth_pose_sequence(
    smpl_model, pose_params_start, pose_params_end, beta_params,
    threshold=0.01, device="cpu", max_depth=15, depth=0,
):
    global_orient_start, body_pose_start, left_hand_start, right_hand_start, jaw_start, leye_start, reye_start = split_pose_params(pose_params_start)
    global_orient_end, body_pose_end, left_hand_end, right_hand_end, jaw_end, leye_end, reye_end = split_pose_params(pose_params_end)

    vert_start = smpl_model.forward(
        global_orient=global_orient_start,
        body_pose=body_pose_start,
        betas=beta_params,
    ).vertices[0].detach().cpu().numpy()

    vert_end = smpl_model.forward(
        global_orient=global_orient_end,
        body_pose=body_pose_end.reshape(1, -1, 3),
        left_hand_pose=left_hand_end.reshape(1, -1, 3),
        right_hand_pose=right_hand_end.reshape(1, -1, 3),
        betas=beta_params,
    ).vertices[0].detach().cpu().numpy()

    non_hand_indices = _non_hand_vertex_indices(vert_start, vert_end)
    max_delta = np.max(np.linalg.norm(vert_end[non_hand_indices] - vert_start[non_hand_indices], axis=1))
    if max_delta < threshold or depth >= max_depth:
        return [vert_start, vert_end]

    pose_params_mid = (pose_params_start + pose_params_end) / 2.0
    first_half = generate_smooth_pose_sequence(
        smpl_model, pose_params_start, pose_params_mid, beta_params, threshold, device, max_depth, depth + 1)
    second_half = generate_smooth_pose_sequence(
        smpl_model, pose_params_mid, pose_params_end, beta_params, threshold, device, max_depth, depth + 1)
    return first_half[:-1] + second_half


def align_smplx_body_y_axis(body_sequence, smpl_body_segmentation, smplx_body_segmentation, default_body_vertices):
    ref_labels = ["head", "neck"]
    smpl_ref_indices = []
    smplx_ref_indices = []
    for label in ref_labels:
        smpl_ref_indices.extend(smpl_body_segmentation[label])
        smplx_ref_indices.extend(smplx_body_segmentation[label])
    smpl_ref_indices = np.array(smpl_ref_indices)
    smplx_ref_indices = np.array(smplx_ref_indices)
    delta = np.mean(default_body_vertices[:, 1][smpl_ref_indices]) - np.mean(body_sequence[0][:, 1][smplx_ref_indices]) - 0.02
    body_sequence_aligned = body_sequence.copy()
    body_sequence_aligned[:, :, 1] += delta
    return body_sequence_aligned


def pose_params_from_npz(pose_file, device):
    pose_vec = np.load(pose_file)
    parts = []
    for key, size in POSE_PARAM_SIZES.items():
        if key in pose_vec:
            value = np.asarray(pose_vec[key], dtype=np.float32).reshape(1, -1)
            if value.shape[1] != size:
                raise ValueError(
                    f"Pose file {pose_file} key '{key}' has {value.shape[1]} values; expected {size}."
                )
        else:
            value = np.zeros((1, size), dtype=np.float32)
        parts.append(torch.from_numpy(value).to(device))
    return torch.cat(parts, dim=1).float()


def build_body_sequence(
    smpl_model, beta_params_np, device, smpl_body_segmentation,
    smplx_body_segmentation, default_body_vertices, pose_file=None,
):
    beta_params = torch.tensor(beta_params_np, dtype=torch.float32, device=device).unsqueeze(0)

    pose_params_start = torch.zeros([1, 165], device=device).float()
    pose_params_start[:, 16 * 3 + 2] = -0.6
    pose_params_start[:, 17 * 3 + 2] = 0.6

    beta_params_zero = torch.zeros([1, 10], dtype=torch.float32, device=device)
    beta_params_zero[0, 0] = 1.5

    if pose_file is None:
        pose_params_end = pose_params_start.clone()
        pose_params_end[:, 16 * 3 + 2] = -0.6
        pose_params_end[:, 17 * 3 + 2] = 0.6
    else:
        pose_params_end = pose_params_from_npz(pose_file, device)

    shape_body_sequence = generate_smooth_shape_sequence(
        smpl_model, pose_params_start, beta_params_zero, beta_params,
        threshold=0.0025, device=device,
    )
    shape_body_sequence = np.stack(shape_body_sequence, axis=0)

    pose_body_sequence = generate_smooth_pose_sequence(
        smpl_model, pose_params_start, pose_params_end, beta_params,
        threshold=0.0025, device=device,
    )
    pose_body_sequence = np.stack(pose_body_sequence, axis=0)[1:]

    body_sequence = np.concatenate([shape_body_sequence, pose_body_sequence], axis=0)
    return align_smplx_body_y_axis(
        body_sequence,
        smpl_body_segmentation,
        smplx_body_segmentation,
        default_body_vertices,
    )
