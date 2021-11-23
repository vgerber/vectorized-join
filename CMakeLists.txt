cmake_minimum_required(VERSION 3.10)
project("vectorized-join" CUDA CXX)

find_package(toml11 REQUIRED)
find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(build_filter OFF)
set(build_join ON)
set(build_probe ON)
set(build_hash ON)
set(build_hist ON)

if(${build_filter})
add_executable(filter 
    src/playground/filter.cu
)

target_compile_options(filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
>) 

target_include_directories(filter PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)
endif()

if(${build_join})
add_executable(join
    src/join_benchmark.cu
)

set_property(TARGET join PROPERTY CUDA_ARCHITECTURES 75)

target_compile_options(join PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -Xcompiler -rdynamic -lineinfo
>) 

target_include_directories(join PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)
target_link_libraries(join PRIVATE curand)
endif()

if(${build_probe})
add_executable(probe
    src/playground/probe_benchmark.cu
)

set_property(TARGET probe PROPERTY CUDA_ARCHITECTURES 75)

target_compile_options(probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -Xcompiler -rdynamic -lineinfo
>) 

target_include_directories(probe PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)

target_link_libraries(probe PRIVATE curand)
endif()

if(${build_hash})
add_executable(hash
    src/playground/hash_benchmark.cu
)

set_property(TARGET hash PROPERTY CUDA_ARCHITECTURES 75)

target_compile_options(hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -rdynamic -lineinfo
>) 

target_include_directories(hash PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)

target_link_libraries(hash PRIVATE curand)
endif()

if(${build_hist})
add_executable(histogram
    src/playground/histogram_benchmark.cu
)

set_property(TARGET histogram PROPERTY CUDA_ARCHITECTURES 75)

target_compile_options(histogram PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -rdynamic -lineinfo
>) 

target_include_directories(histogram PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)

target_link_libraries(histogram PRIVATE curand)
endif()