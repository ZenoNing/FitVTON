#!/bin/bash
if [ -n "${_SAVED_PATH:-}" ]; then
  export PATH="$_SAVED_PATH"
else
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
fi

if [ -n "${_SAVED_CUDA_HOME:-}" ]; then
  export CUDA_HOME="$_SAVED_CUDA_HOME"
else
  unset CUDA_HOME
fi

if [ -n "${_SAVED_CUDA_PATH:-}" ]; then
  export CUDA_PATH="$_SAVED_CUDA_PATH"
else
  unset CUDA_PATH
fi

if [ -n "${_SAVED_LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$_SAVED_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

unset _SAVED_PATH _SAVED_CUDA_HOME _SAVED_CUDA_PATH _SAVED_LD_LIBRARY_PATH
