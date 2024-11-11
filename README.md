# Hash-Join and Hashing with multi GPU systems
[Diploma thesis](diploma_thesis.pdf) which investigated the use of multi GPU systems for joining large database tables using the hash join algoirthm.

## Implementation
- The main hash-join kernels are implemented in [join_provider](src/join/join_provider.cu)
- The main hash kernels are implemented in [hash](src/hash/hash.cu)
- Benchmark results provided in my thesis are located in [benchmark](benchmark/)
- Example usages can be found in [playground](src/playground)

## Build

##### Libs
  - nlohman_json
  - toml11
  - CUDA 10+

##### CMAKE
```cmake
mkdir build
cd build
cmake ..
cmake --build .
```
