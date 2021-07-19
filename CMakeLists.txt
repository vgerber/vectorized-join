cmake_minimum_required(VERSION 3.10)
project("vectorized-join" CUDA CXX)

find_package(toml11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(build_filter OFF)
set(build_join ON)
set(build_probe ON)

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
endif()
