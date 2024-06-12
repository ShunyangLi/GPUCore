![make](https://img.shields.io/badge/make-4.3-brightgreen.svg)
![cmake](https://img.shields.io/badge/cmake-3.22.1-brightgreen.svg)
![C++](https://img.shields.io/badge/C++-11.4.0-blue.svg)
![nvcc](https://img.shields.io/badge/CUDA-12.2-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)


# Bipartite Core Decomposition On GPU

## Prerequisites
- CMake 3.22 or higher
- NVIDIA CUDA Toolkit 12.2 or higher
- A CUDA-capable GPU device

## Build from Source
To build the GPU core code, execute the following commands in your terminal:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && make
```

## CMake Configuration
For a better performance, you need to change the `set(CMAKE_CUDA_ARCHITECTURES 86)` in `CMakeLists.txt` file.
The value of `CMAKE_CUDA_ARCHITECTURES` can be obtained from the output of GPU device info (Compute Capability):
```bash
./core --device 0 --device_info
----------------------------------------------------------------
|            Property             |            Info            |
----------------------------------------------------------------
...
| Compute Capability              | 86                         |
...
----------------------------------------------------------------
```
Set the appropriate `CMAKE_CUDA_ARCHITECTURES` according to the output, then rebuild the code.
