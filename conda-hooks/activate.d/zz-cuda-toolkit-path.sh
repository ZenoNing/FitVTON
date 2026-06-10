#!/bin/bash
export _SAVED_PATH="$PATH"
export _SAVED_CUDA_HOME="${CUDA_HOME:-}"
export _SAVED_CUDA_PATH="${CUDA_PATH:-}"
export _SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"

# Keep core system utilities available for conda compiler activation scripts.
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
