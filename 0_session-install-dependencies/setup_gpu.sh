#!/bin/bash
# Copyright (c) 2024 Cloudera, Inc.

# This file is part of Chat with your doc AMP.

# Chat with your doc AMP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.

# Chat with your doc AMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with Chat with your doc AMP. If not, see <https://www.gnu.org/licenses/>.
echo "installing GPU version of llama-cpp"

CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.3/lib64 -DCUDACXX=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" FORCE_CMAKE=1 pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.82 --force-reinstall --no-cache-dir

