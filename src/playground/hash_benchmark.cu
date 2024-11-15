#include <json.hpp>
#include <vector>

#include "benchmark/configuration.hpp"
#include "benchmark/data_generator.cu"
#include "hash/hash.cu"

const int GPU_MODE_SINGLE = 0;
const int GPU_MODE_SHARED = 1;

struct HashBenchmarkResult {
    BenchmarkConfig benchmark_config;
    HashBenchmarkConfig hash_benchmark_config;
    double hash_p_second = -1.0;
    double gb_p_second = -1.0;
    double dist_gb_p_second = -1.0;
    double merge_gb_p_second = -1.0;
    char chunk_size = 0;
    int run_id = -1;
    int blocks = 0;
    int threads = 0;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << config_to_string_header() << ","
                      << "hash_p_second,"
                      << "gb_p_second,"
                      << "dist_gb_p_second,"
                      << "merge_gb_p_second";
        return string_stream.str();
    }

    static std::string config_to_string_header() {
        std::ostringstream string_stream;
        string_stream << "run_id,"
                      << "blocks,"
                      << "threads," << BenchmarkConfig::to_string_header() << "," << HashBenchmarkConfig::to_string_header() << ","
                      << "chunk_size";
        return string_stream.str();
    }

    std::string config_to_string() {
        std::ostringstream string_stream;
        string_stream << run_id << "," << blocks << "," << threads << "," << benchmark_config.to_string() << "," << hash_benchmark_config.to_string() << "," << (int)chunk_size;
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << config_to_string() << "," << hash_p_second << "," << gb_p_second << "," << dist_gb_p_second << "," << merge_gb_p_second;
        return string_stream.str();
    }

    void print() {
        // printf("Runtime %e ms %e H/s %e Gb/s\n", run_time_avg, (config.elements / run_time_avg), (element_buffer_size * element_size / run_time_avg));
    }
};

HashBenchmarkResult hash_single_gpu(BenchmarkConfig benchmark_config, HashBenchmarkConfig hash_benchmark_config, int run_id, bool verbose = false) {
    float elapsed_time_avg = 0.0;
    index_t element_buffer_size = benchmark_config.elements;
    short int element_size = hash_benchmark_config.element_size;
    short int element_chunks = 0;

    int threads = hash_benchmark_config.thread_size;
    int blocks = max(element_buffer_size / threads / hash_benchmark_config.hashes_p_thread, (index_t)1);
    for (int run_index = 0; run_index < benchmark_config.runs; run_index++) {

        /*
         * Allocate
         */
        float elapsed_time_s = 0.0;

        cudaStream_t *device_stream = new cudaStream_t[benchmark_config.gpus];
        const int hash_event_count = 2;
        cudaEvent_t *hash_events = new cudaEvent_t[benchmark_config.gpus * hash_event_count];
        chunk_t **d_element_buffers = new chunk_t *[benchmark_config.gpus];
        hash_t **d_hashed_buffers = new hash_t *[benchmark_config.gpus];
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));
            gpuErrchk(cudaStreamCreate(&device_stream[gpu_index]));
            gpuErrchk(cudaGetLastError());

            for (int hash_event_index = 0; hash_event_index < hash_event_count; hash_event_index++) {
                cudaEventCreate(&hash_events[gpu_index * hash_event_count + hash_event_index]);
            }
            gpuErrchk(cudaMalloc(&d_hashed_buffers[gpu_index], element_buffer_size * sizeof(hash_t)));
            generate_demo_data(element_buffer_size, element_size, &element_chunks, &d_element_buffers[gpu_index]);

            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaGetLastError());
        }

        /*
         *  Run
         */
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));

            HashConfig hash_config;
            hash_config.stream = device_stream[gpu_index];
            hash_config.algorithm = hash_benchmark_config.algorithm;
            hash_config.threads_per_block = hash_benchmark_config.thread_size;
            hash_config.elements_per_thread = hash_benchmark_config.hashes_p_thread;
            hash_config.enable_profile(hash_events[gpu_index * hash_event_count], hash_events[gpu_index * hash_event_count + 1]);
            hash_func(element_buffer_size, 0, element_chunks, d_element_buffers[gpu_index], d_hashed_buffers[gpu_index], hash_config);
        }

        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaGetLastError());
        }
        /*
         *  Measure
         */
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            float tmp_elapsed_time = 0.0f;
            gpuErrchk(cudaEventElapsedTime(&tmp_elapsed_time, hash_events[gpu_index * hash_event_count], hash_events[gpu_index * hash_event_count + 1]));
            elapsed_time_s = max(tmp_elapsed_time / pow(10, 3), elapsed_time_s);
        }

        elapsed_time_avg += elapsed_time_s;

        if (verbose) {
            HashBenchmarkResult tmp_result;
            tmp_result.benchmark_config = benchmark_config;
            tmp_result.hash_benchmark_config = hash_benchmark_config;
            tmp_result.run_id = run_id;
            tmp_result.chunk_size = sizeof(chunk_t);
            std::cout << tmp_result.config_to_string() << std::endl;
        }

        /*
         * Free
         */
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));
            gpuErrchk(cudaStreamDestroy(device_stream[gpu_index]));
            for (int hash_event_index = 0; hash_event_index < hash_event_count; hash_event_index++) {
                cudaEventDestroy(hash_events[gpu_index * hash_event_count + hash_event_index]);
            }

            gpuErrchk(cudaFree(d_hashed_buffers[gpu_index]));
            gpuErrchk(cudaFree(d_element_buffers[gpu_index]));
            gpuErrchk(cudaGetLastError());
        }

        delete[] hash_events;
        delete[] d_hashed_buffers;
        delete[] d_element_buffers;
        delete[] device_stream;
        run_id++;
    }

    HashBenchmarkResult hash_result;
    elapsed_time_avg /= benchmark_config.runs;
    hash_result.run_id = -1;
    hash_result.gb_p_second = ((element_buffer_size * element_size * benchmark_config.gpus) / elapsed_time_avg) / pow(10, 9);
    hash_result.hash_p_second = ((element_buffer_size * benchmark_config.gpus) / elapsed_time_avg);
    hash_result.dist_gb_p_second = 0.0f;
    hash_result.merge_gb_p_second = 0.0f;
    hash_result.benchmark_config = benchmark_config;
    hash_result.hash_benchmark_config = hash_benchmark_config;
    hash_result.chunk_size = sizeof(chunk_t);
    hash_result.blocks = blocks;
    hash_result.threads = threads;
    return hash_result;
}

HashBenchmarkResult hash_shared_gpu(BenchmarkConfig benchmark_config, HashBenchmarkConfig hash_benchmark_config, int run_id, bool verbose = false) {
    float elapsed_time_avg = 0.0;
    float dist_time_avg = 0.0;
    float kernel_time_avg = 0.0;
    float merge_time_avg = 0.0;
    index_t element_buffer_size = benchmark_config.elements;
    short int element_size = hash_benchmark_config.element_size;
    short int element_chunks = 0;

    int threads = hash_benchmark_config.thread_size;
    int blocks = max(element_buffer_size / threads / hash_benchmark_config.hashes_p_thread, (index_t)1);
    for (int run_index = 0; run_index < benchmark_config.runs; run_index++) {

        /*
         * Allocate
         */
        float elapsed_time_s = 0.0;
        float dist_runtime_s = 0.0f;
        float kernel_runtime_s = 0.0f;
        float merge_runtime_s = 0.0f;

        cudaStream_t *device_stream = new cudaStream_t[benchmark_config.gpus];
        const int mem_event_count = 4;
        const int hash_event_count = 2;
        cudaEvent_t *mem_events = new cudaEvent_t[(benchmark_config.gpus - 1) * mem_event_count];
        cudaEvent_t *hash_events = new cudaEvent_t[benchmark_config.gpus * hash_event_count];
        chunk_t **d_element_buffers = new chunk_t *[benchmark_config.gpus];
        hash_t **d_hashed_buffers = new hash_t *[benchmark_config.gpus];
        int *buffer_offsets = new int[benchmark_config.gpus];
        int *buffer_sizes = new int[benchmark_config.gpus];
        int buffer_offset = 0;
        int buffer_offset_inecrement = benchmark_config.elements / benchmark_config.gpus;

        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));

            for (int peer_gpu_index = 0; peer_gpu_index < benchmark_config.gpus; peer_gpu_index++) {
                if (peer_gpu_index != gpu_index) {
                    gpuErrchk(cudaDeviceEnablePeerAccess(peer_gpu_index, 0));
                }
            }

            gpuErrchk(cudaStreamCreate(&device_stream[gpu_index]));
            if (gpu_index > 0) {
                for (int mem_event_index = 0; mem_event_index < mem_event_count; mem_event_index++) {
                    gpuErrchk(cudaEventCreate(&mem_events[(gpu_index - 1) * mem_event_count + mem_event_index]));
                }
            }

            for (int hash_event_index = 0; hash_event_index < hash_event_count; hash_event_index++) {
                gpuErrchk(cudaEventCreate(&hash_events[gpu_index * hash_event_count + hash_event_index]));
            }

            buffer_offsets[gpu_index] = buffer_offset;
            if (gpu_index == benchmark_config.gpus - 1) {
                buffer_sizes[gpu_index] = benchmark_config.elements - buffer_offset;
            } else {
                buffer_sizes[gpu_index] = buffer_offset_inecrement;
            }

            buffer_offset += buffer_sizes[gpu_index];

            if (gpu_index == 0) {
                gpuErrchk(cudaMalloc(&d_hashed_buffers[gpu_index], element_buffer_size * sizeof(hash_t)));
                generate_demo_data(element_buffer_size, element_size, &element_chunks, &d_element_buffers[gpu_index]);
            } else {
                gpuErrchk(cudaMalloc(&d_hashed_buffers[gpu_index], buffer_sizes[gpu_index] * sizeof(hash_t)));
                gpuErrchk(cudaMalloc(&d_element_buffers[gpu_index], buffer_sizes[gpu_index] * element_chunks * sizeof(chunk_t)));
            }
        }

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaGetLastError());

        /*
         * Run
         */
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));

            int mem_event_offset = (gpu_index - 1) * mem_event_count;
            if (gpu_index > 0) {
                gpuErrchk(cudaEventRecord(mem_events[mem_event_offset], device_stream[gpu_index]));
                chunk_t *master_element_buffer = &(d_element_buffers[0][buffer_offsets[gpu_index] * element_chunks]);
                gpuErrchk(cudaMemcpyPeerAsync(d_element_buffers[gpu_index], gpu_index, master_element_buffer, 0, buffer_sizes[gpu_index] * element_chunks * sizeof(chunk_t), device_stream[gpu_index]));
                gpuErrchk(cudaEventRecord(mem_events[mem_event_offset + 1], device_stream[gpu_index]));
            }

            HashConfig hash_config;
            hash_config.stream = device_stream[gpu_index];
            hash_config.algorithm = hash_benchmark_config.algorithm;
            hash_config.threads_per_block = hash_benchmark_config.thread_size;
            hash_config.elements_per_thread = hash_benchmark_config.hashes_p_thread;
            hash_config.enable_profile(hash_events[gpu_index * hash_event_count], hash_events[gpu_index * hash_event_count + 1]);
            hash_func(buffer_sizes[gpu_index], 0, element_chunks, d_element_buffers[gpu_index], d_hashed_buffers[gpu_index], hash_config);

            if (gpu_index > 0) {
                gpuErrchk(cudaEventRecord(mem_events[mem_event_offset + 2], device_stream[gpu_index]));
                hash_t *master_hash_buffer = &(d_hashed_buffers[0][buffer_offsets[gpu_index]]);
                gpuErrchk(cudaMemcpyPeerAsync(master_hash_buffer, 0, d_hashed_buffers[gpu_index], gpu_index, buffer_sizes[gpu_index] * sizeof(hash_t), device_stream[gpu_index]));
                gpuErrchk(cudaEventRecord(mem_events[mem_event_offset + 3], device_stream[gpu_index]));
            }
        }

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaGetLastError());

        /*
         * Measure
         */
        if (benchmark_config.gpus > 1) {
            cudaEventElapsedTime(&elapsed_time_s, mem_events[0], mem_events[(benchmark_config.gpus - 1) * mem_event_count - 1]);
        } else {
            cudaEventElapsedTime(&elapsed_time_s, hash_events[0], hash_events[1]);
        }
        elapsed_time_s /= pow(10, 3);

        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            float tmp_dist_runtime;
            float tmp_kernel_runtime;
            float tmp_merge_runtime;
            if (gpu_index > 0) {
                int event_offset = (gpu_index - 1) * mem_event_count;
                cudaEventElapsedTime(&tmp_dist_runtime, mem_events[event_offset], mem_events[event_offset + 1]);
                cudaEventElapsedTime(&tmp_merge_runtime, mem_events[event_offset + 2], mem_events[event_offset + 3]);
                tmp_dist_runtime /= pow(10, 3);
                tmp_merge_runtime /= pow(10, 3);
                dist_runtime_s += tmp_dist_runtime;
                merge_runtime_s += tmp_merge_runtime;
            }
            int event_offset = gpu_index * hash_event_count;
            cudaEventElapsedTime(&tmp_kernel_runtime, hash_events[event_offset], hash_events[event_offset + 1]);
            tmp_kernel_runtime /= pow(10, 3);
            kernel_runtime_s += tmp_kernel_runtime;
        }

        dist_runtime_s = dist_runtime_s / (benchmark_config.gpus - 1);
        kernel_runtime_s = kernel_runtime_s / benchmark_config.gpus;
        merge_runtime_s = merge_runtime_s / (benchmark_config.gpus - 1);

        elapsed_time_avg += elapsed_time_s;
        dist_time_avg += dist_runtime_s;
        kernel_time_avg += kernel_runtime_s;
        merge_time_avg += merge_runtime_s;

        if (verbose) {
            HashBenchmarkResult tmp_result;
            tmp_result.benchmark_config = benchmark_config;
            tmp_result.hash_benchmark_config = hash_benchmark_config;
            tmp_result.run_id = run_id;
            tmp_result.chunk_size = sizeof(chunk_t);
            std::cout << tmp_result.config_to_string() << std::endl;
        }

        /*
         * Free
         */
        for (int gpu_index = 0; gpu_index < benchmark_config.gpus; gpu_index++) {
            gpuErrchk(cudaSetDevice(gpu_index));
            // gpu 0 is the master
            for (int peer_gpu_index = 0; peer_gpu_index < benchmark_config.gpus; peer_gpu_index++) {
                if (gpu_index != peer_gpu_index) {
                    gpuErrchk(cudaDeviceDisablePeerAccess(peer_gpu_index));
                }
            }

            cudaStreamDestroy(device_stream[gpu_index]);
            if (gpu_index > 0) {
                for (int mem_event_index = 0; mem_event_index < mem_event_count; mem_event_index++) {
                    cudaEventDestroy(mem_events[(gpu_index - 1) * mem_event_count + mem_event_index]);
                }
            }

            for (int hash_event_index = 0; hash_event_index < hash_event_count; hash_event_index++) {
                cudaEventDestroy(hash_events[gpu_index * hash_event_count + hash_event_index]);
            }

            gpuErrchk(cudaFree(d_hashed_buffers[gpu_index]));
            gpuErrchk(cudaFree(d_element_buffers[gpu_index]));
        }

        delete[] mem_events;
        delete[] hash_events;
        delete[] d_hashed_buffers;
        delete[] d_element_buffers;
        delete[] device_stream;
        delete[] buffer_offsets;
        delete[] buffer_sizes;
        run_id++;
    }

    elapsed_time_avg /= benchmark_config.runs;
    dist_time_avg /= benchmark_config.runs;
    kernel_time_avg /= benchmark_config.runs;
    merge_time_avg /= benchmark_config.runs;

    HashBenchmarkResult hash_result;
    hash_result.run_id = -1;
    hash_result.gb_p_second = ((element_buffer_size * element_size) / elapsed_time_avg) / pow(10, 9);
    hash_result.hash_p_second = (element_buffer_size / elapsed_time_avg);
    if (benchmark_config.gpus > 1) {
        hash_result.dist_gb_p_second = ((element_buffer_size * element_size) / benchmark_config.gpus) / dist_time_avg / pow(10, 9);
        hash_result.merge_gb_p_second = (element_buffer_size / benchmark_config.gpus) / merge_time_avg / pow(10, 9);
    }

    hash_result.benchmark_config = benchmark_config;
    hash_result.hash_benchmark_config = hash_benchmark_config;
    hash_result.chunk_size = sizeof(chunk_t);
    hash_result.blocks = blocks;
    hash_result.threads = threads;

    return hash_result;
}

int main(int argc, char *argv[]) {

    std::srand(0);

    // verbose prints every result
    bool verbose = false;
    if (argc >= 4) {
        verbose = std::string(argv[3]) == "-v";
    }

    BenchmarkSetup benchmark_setup;
    if (!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup) && !verbose) {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    std::fstream run_csv_stream;
    if (!verbose) {
        run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
        run_csv_stream << HashBenchmarkResult::to_string_header() << std::endl;
    }
    cudaDeviceReset();

    if (verbose) {
        std::cout << HashBenchmarkResult::config_to_string_header() << std::endl;
    }

    int run_id = 0;

    auto benchmark_configs = get_benchmark_configs(benchmark_setup);
    auto hash_benchmark_configs = get_hash_benchmark_configs(benchmark_setup);
    for (auto benchmark_config : benchmark_configs) {
        for (auto hash_benchmark_config : hash_benchmark_configs) {
            HashBenchmarkResult hash_result;

            if (benchmark_config.gpu_mode == GPU_MODE_SINGLE) {
                hash_result = hash_single_gpu(benchmark_config, hash_benchmark_config, run_id, verbose);
            } else if (benchmark_config.gpu_mode == GPU_MODE_SHARED) {
                hash_result = hash_shared_gpu(benchmark_config, hash_benchmark_config, run_id, verbose);
            } else {
                std::cout << "Invalid gpu mode " << benchmark_config.gpu_mode << std::endl;
                return -1;
            }

            if (!verbose) {
                std::cout << run_id + 1 << "/" << benchmark_configs.size() * hash_benchmark_configs.size() * benchmark_config.runs << std::endl;
            }
            run_id += benchmark_config.runs;

            if (!verbose) {
                run_csv_stream << hash_result.to_string() << std::endl;
            }
        }
    }

    run_csv_stream.close();
}