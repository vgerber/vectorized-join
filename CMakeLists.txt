cmake_minimum_required(VERSION 3.10)
project("vectorized-join" CUDA CXX)

find_package(toml11 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(filter 
    src/playground/filter.cu
    src/benchmark/config.hpp
    src/benchmark/log.hpp
)

target_compile_options(filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
>) 

target_include_directories(filter PRIVATE
    ${TOML11_INCLUDE_DIR}
    src/
)