"""Create a video from per-frame front renders.

This repo can render per simulation frame into:
  <sample_out>/frames/<sim_tag>_frame_00000_render_front.png

This script collects those PNGs, sorts them by frame index, and uses ffmpeg to
encode a video (default: MP4/H.264).

Examples
--------
# Single sample directory that contains a frames/ subdir
python post_processing_scripts/frames_front_to_video.py \
  --input /path/to/sample_out \
  --output /path/to/sample_out/front.mp4 \
  --fps 30

# Or point directly at the frames directory
python post_processing_scripts/frames_front_to_video.py \
  --frames-dir /path/to/sample_out/frames \
  --output /path/to/sample_out/front.mp4

Notes
-----
- Requires ffmpeg on PATH.
- If your images are not all the same size, ffmpeg may fail.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_FRONT_GLOB = "*_render_front.png"
_FRAME_RE = re.compile(r"_frame_(\d+)_render_front\.png$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a video from per-frame front PNG renders")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Sample output dir that contains frames/")
    src.add_argument("--frames-dir", type=Path, help="Directory that contains per-frame PNGs")

    p.add_argument("--output", type=Path, required=True, help="Output video path (e.g., out.mp4)")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    p.add_argument(
        "--hold-first-seconds",
        type=float,
        default=1.0,
        help="Hold frame_00000 for this many seconds (set 0 to disable)",
    )
    p.add_argument("--glob", dest="glob_pattern", default=_FRONT_GLOB, help="Glob for input PNGs")

    # Encoding knobs
    p.add_argument("--crf", type=int, default=18, help="H.264 quality (lower is higher quality)")
    p.add_argument("--preset", default="medium", help="ffmpeg preset (e.g., veryfast, fast, medium)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")

    return p.parse_args()


def _frame_index(path: Path) -> int:
    m = _FRAME_RE.search(path.name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except ValueError:
        return -1


def _find_frames(frames_dir: Path, glob_pattern: str) -> list[Path]:
    paths = sorted(frames_dir.glob(glob_pattern), key=lambda p: (_frame_index(p), p.name))
    return [p for p in paths if p.is_file()]


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it, e.g.\n"
            "- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "- Windows: winget install Gyan.FFmpeg (or download from https://ffmpeg.org/)\n"
        )
    return ffmpeg


def _write_concat_file(imgs: list[Path], fps: float, concat_path: Path, hold_first_seconds: float) -> None:
    # ffmpeg concat demuxer for images needs per-file durations.
    # The last file must be listed without duration per ffmpeg docs.
    frame_duration = 1.0 / float(fps)

    hold_first_seconds = max(float(hold_first_seconds), 0.0)

    lines: list[str] = []

    # Prefer holding the true frame_00000 if present; otherwise hold the first image.
    hold_target: Path | None = None
    if hold_first_seconds > 0:
        for img in imgs:
            if _frame_index(img) == 0:
                hold_target = img
                break
        if hold_target is None:
            hold_target = imgs[0]

    for img in imgs[:-1]:
        lines.append(f"file '{img.as_posix()}'")
        if hold_first_seconds > 0 and hold_target is not None and img == hold_target:
            lines.append(f"duration {hold_first_seconds:.10f}")
        else:
            lines.append(f"duration {frame_duration:.10f}")
    lines.append(f"file '{imgs[-1].as_posix()}'")

    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_ffmpeg(
    ffmpeg: str,
    concat_list: Path,
    out_path: Path,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        # Ensure even dimensions for H.264 and standard pixel format
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-r",
        str(float(fps)),
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
    ]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")

    cmd.append(str(out_path))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ffmpeg failed. Common causes: images have different sizes, "
            "or a corrupted PNG. Try checking a few frames or rendering at a fixed resolution."
        ) from e


def main() -> int:
    args = _parse_args()

    frames_dir = args.frames_dir
    if frames_dir is None:
        # Allow --input to be either:
        #  1) a sample output directory that contains frames/
        #  2) the frames/ directory itself
        candidate = args.input
        if candidate is None:
            print("Error: missing --input/--frames-dir", file=sys.stderr)
            return 2

        if (candidate / "frames").exists():
            frames_dir = candidate / "frames"
        else:
            frames_dir = candidate

    frames_dir = frames_dir.resolve()
    if not frames_dir.exists():
        print(
            f"Error: frames dir not found: {frames_dir}\n"
            "Tip: if you already point to the frames folder, use --frames-dir (or keep --input, it should work too).",
            file=sys.stderr,
        )
        return 2

    imgs = _find_frames(frames_dir, args.glob_pattern)
    if not imgs:
        print(f"Error: no images matched {args.glob_pattern} under {frames_dir}", file=sys.stderr)
        return 2

    # Basic sanity: enforce monotonic frame indices if present
    idxs = [_frame_index(p) for p in imgs]
    if any(i < 0 for i in idxs):
        print(
            "Warning: some files don't match the expected '_frame_XXXXX_render_front.png' pattern; "
            "sorting may be imperfect.",
            file=sys.stderr,
        )

    ffmpeg = _require_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        concat_list = Path(tmp) / "concat.txt"
        _write_concat_file(imgs, args.fps, concat_list, args.hold_first_seconds)
        _run_ffmpeg(
            ffmpeg=ffmpeg,
            concat_list=concat_list,
            out_path=args.output,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )

    print(f"Wrote video: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
