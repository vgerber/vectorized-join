cmake_minimum_required(VERSION 3.10)
project("vectorized-join" CUDA CXX)

find_package(toml11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(build_filter ON)
set(build_partition ON)

if(${build_filter})
add_executable(filter 
    src/playground/filter.cu
    src/benchmark/configuration.hpp
    src/benchmark/log.hpp
    src/config.h
)

target_compile_options(filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
>) 

target_include_directories(filter PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)
endif()

if(${build_partition})
add_executable(partition
    src/playground/partition_test.cu
)

target_compile_options(partition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -Xcompiler -rdynamic -lineinfo
>) 

target_include_directories(partition PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)
endif()
