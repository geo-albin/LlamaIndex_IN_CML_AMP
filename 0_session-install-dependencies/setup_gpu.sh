#!/bin/bash

echo "installing GPU version of llama-cpp"

#pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.82 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.3/lib64 -DCUDACXX=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" FORCE_CMAKE=1 pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.82 --force-reinstall --no-cache-dir

