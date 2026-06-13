# Environment Setup

Run commands from the `FitVTON/` directory unless noted otherwise.

## Create the Main Environment

```bash
conda env create -f environment.yml
bash conda-hooks/install.sh flux
conda activate flux
python -m pip install pyrender==0.1.45 --no-deps
```

Copy the custom FLUX Kontext pipeline into the active environment:

```bash
python - <<'PY'
from pathlib import Path
import shutil, diffusers

src = Path("pipeline_flux_kontext_multiple_images.py").resolve()
dst = Path(diffusers.__file__).resolve().parent / "pipelines/flux/pipeline_flux_kontext_multiple_images.py"
shutil.copy2(src, dst)
print(dst)
PY
```

This is the main environment for FLUX training/inference and GarmentCode simulation.

## CUDA Hooks

The repo ships `conda-hooks/` so the active conda environment controls `CUDA_HOME`, `CUDA_PATH`, and `PATH`.

Before training or simulation:

```bash
conda activate flux
bash conda-hooks/install.sh flux
```

Expected checks:

```bash
echo "$CUDA_HOME"
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Expected for `flux`:

- `CUDA_HOME` points to `$CONDA_PREFIX`
- `nvcc --version` reports CUDA 12.6

If your shell exports system CUDA globally, prefer this `~/.bashrc` pattern:

```bash
if [ -z "${CONDA_PREFIX:-}" ]; then
  export PATH="/usr/local/cuda-12.6/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
  export CUDA_HOME="/usr/local/cuda-12.6"
  export CUDA_PATH="/usr/local/cuda-12.6"
fi
```

## If Activate Breaks Basic Commands

If activation prints errors like `grep/tr/sed: command not found`, reset `PATH` and re-activate:

```bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
conda deactivate
conda activate flux
```

Opening a fresh terminal is often easier.

## Build Warp

The GarmentCodeV2 simulation pipeline uses the bundled NVIDIA Warp fork (`NvidiaWarp-GarmentCode/`) for XPBD cloth simulation. Build it inside the active `flux` environment. Do not reuse Warp binaries compiled in another environment.

```bash
conda activate flux
cd NvidiaWarp-GarmentCode

TARGETS="$CONDA_PREFIX/targets/x86_64-linux"
TARGETS_INCLUDE="$TARGETS/include"
TARGETS_LIB="$TARGETS/lib"

ln -sfn "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"

for item in "$TARGETS_INCLUDE"/*; do
  base=$(basename "$item")
  if [ ! -e "$CONDA_PREFIX/include/$base" ]; then
    ln -sfn "$item" "$CONDA_PREFIX/include/$base"
  fi
done
for lib in "$TARGETS_LIB"/*static*.a; do
  ln -sfn "$lib" "$CONDA_PREFIX/lib64/$(basename "$lib")"
done

python build_lib.py
python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import warp as wp
wp.init()
print(wp.__file__)
print(wp.get_device("cuda:0") if wp.is_cuda_available() else "cpu only")
PY
```
