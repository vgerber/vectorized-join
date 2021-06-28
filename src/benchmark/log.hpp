#pragma once
#include <fstream>

void write_benchmark_header(std::fstream &output) {
    output << "version" << ",";
    output << "gpu_count" << ",";
    output << "mp_count" << ",";
    output << "threads_per_gpu" << ",";
    output << "blocks_per_gpu" << ",";
    output << "element_count" << ",";
    output << "element_size" << ",";
    output << "runtime_ms" << ",";
    output << "throughput_gb" << ",";
    output << "elements_per_thread" << ",";
    output << std::endl;
}

void write_benchmark(std::fstream &output, int filter_version, int gpu_count, int mp_count, int threads_per_gpu, int blocks_per_gpu, int element_count, int element_size, float runtime_ms, float throughput_gb, float elements_per_thread) {
    output << filter_version << ",";
    output << gpu_count << ",";
    output << mp_count << ",";
    output << threads_per_gpu << ",";
    output << blocks_per_gpu << ",";
    output << element_count << ",";
    output << element_size << ",";
    output << runtime_ms << ",";
    output << throughput_gb << ",";
    output << elements_per_thread;
    output << std::endl;
}