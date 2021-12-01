#include "benchmark/configuration.hpp"
#include "benchmark/data_generator.cu"
#include "hash/hash.cu"
#include "join/join_provider.cuh"
#include "json.hpp"

using json = nlohmann::json;

struct ProbeBenchmarkResult {
    BenchmarkConfig benchmark_config;
    ProbeBenchmarkConfig probe_config;
    JoinBenchmarkConfig join_config;
    index_t probe_elements;
    double tuples_p_second = 0;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << BenchmarkConfig::to_string_header() << "," << JoinBenchmarkConfig::to_string_header() << "," << ProbeBenchmarkConfig::to_string_header() << ","
                      << "probe_elements,"
                      << "tuples_p_second";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << benchmark_config.to_string() << "," << join_config.to_string() << "," << probe_config.to_string() << "," << probe_elements << "," << tuples_p_second;
        return string_stream.str();
    }
};

void store_probe_summary(json &parent_json, ProbeSummary probe_summary) {
    json probe_summary_json;
    probe_summary_json["k_build_probe_tuples_p_second"] = probe_summary.k_build_probe_tuples_p_second;
    probe_summary_json["k_build_probe_gb_p_second"] = probe_summary.k_build_probe_gb_p_second;
    probe_summary_json["k_extract_tuples_p_second"] = probe_summary.k_extract_tuples_p_second;
    probe_summary_json["k_extract_gb_p_second"] = probe_summary.k_extract_gb_p_second;
    probe_summary_json["r_elements"] = probe_summary.r_elements;
    probe_summary_json["s_elements"] = probe_summary.s_elements;
    probe_summary_json["rs_elements"] = probe_summary.rs_elements;
    probe_summary_json["probe_mode"] = probe_summary.probe_mode;
    parent_json.push_back(probe_summary_json);
}

int main(int argc, char **argv) {

    std::srand(0);

    BenchmarkSetup benchmark_setup;
    if (!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup)) {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    std::fstream run_csv_stream;
    run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
    run_csv_stream << ProbeBenchmarkResult::to_string_header() << std::endl;

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    std::cout << device_properties.asyncEngineCount << "," << device_properties.concurrentKernels << "," << device_properties.multiProcessorCount << std::endl;

    // fix to 1 for probe config preallocated data
    int stream_count = 1; // benchmark_setup.max_streams;
    cudaStream_t *streams = new cudaStream_t[stream_count];
    for (int stream_index = 0; stream_index < stream_count; stream_index++) {
        cudaStreamCreate(&streams[stream_index]);
    }

    std::vector<BenchmarkConfig> benchmark_configs = get_benchmark_configs(benchmark_setup);
    std::vector<ProbeBenchmarkConfig> probe_configs = get_probe_benchmark_configs(benchmark_setup);
    std::vector<JoinBenchmarkConfig> join_configs = get_join_configs(benchmark_setup);
    ProbeBenchmarkResult best_result;

    HashConfig hash_config;
    hash_config.algorithm = "fnv";
    hash_config.profile_enabled = false;

    cudaEvent_t profiling_start, profiling_end;
    cudaEventCreate(&profiling_start);
    cudaEventCreate(&profiling_end);

    json probe_json;

    int config_index = 0;
    for (auto benchmark_config : benchmark_configs) {
        for (auto join_benchmark_config : join_configs) {
            int column_count = join_benchmark_config.rs_columns;
            index_t s_table_size = benchmark_config.elements;
            index_t r_table_size = benchmark_config.elements / join_benchmark_config.rs_scale;

            db_table *r_tables = new db_table[stream_count];
            db_table *s_tables = new db_table[stream_count];
            db_table *rs_tables = new db_table[stream_count];

            db_hash_table *r_hash_tables = new db_hash_table[stream_count];
            db_hash_table *s_hash_tables = new db_hash_table[stream_count];

            size_t table_bytes = (r_table_size + s_table_size) * ((column_count + 1) * sizeof(column_t) + sizeof(hash_t));
            size_t rs_bytes = (r_table_size * s_table_size + 1) * sizeof(index_s_t);
            if (out_of_memory(table_bytes + rs_bytes)) {
                std::cout << "Tables to large" << std::endl;
                continue;
            }

            column_t max_column_value = benchmark_config.max_value;
            for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                generate_tables(r_table_size, s_table_size, column_count, r_tables[stream_index], s_tables[stream_index], max_column_value, benchmark_config.skew, benchmark_config.distribution);

                hash_config.stream = streams[stream_index];
                r_hash_tables[stream_index] = db_hash_table(r_tables[stream_index].size, r_tables[stream_index].gpu);
                s_hash_tables[stream_index] = db_hash_table(s_tables[stream_index].size, s_tables[stream_index].gpu);
                hash_func(r_tables[stream_index].size, 0, sizeof(column_t) / sizeof(chunk_t) * r_tables[stream_index].column_count, (chunk_t *)r_tables[stream_index].column_values, r_hash_tables[stream_index].hashes, hash_config);
                hash_func(s_tables[stream_index].size, 0, sizeof(column_t) / sizeof(chunk_t) * s_tables[stream_index].column_count, (chunk_t *)s_tables[stream_index].column_values, s_hash_tables[stream_index].hashes, hash_config);
            }

            for (auto probe_benchmark_config : probe_configs) {
                // config.print();
                // print_mem();
                config_index++;
                printf("%d/%lu\n", config_index, probe_configs.size() * benchmark_configs.size() * join_configs.size());

                ProbeConfig probe_config;
                probe_config.build_n_per_thread = probe_benchmark_config.build_n_per_thread;
                probe_config.build_table_load = probe_benchmark_config.build_table_load;
                probe_config.build_threads = probe_benchmark_config.build_threads;
                probe_config.extract_threads = probe_benchmark_config.extract_threads;
                probe_config.extract_n_per_thread = probe_benchmark_config.extract_n_per_thread;
                probe_config.probe_mode = probe_benchmark_config.probe_mode;
                if (benchmark_config.profile) {
                    probe_config.enable_profiling(profiling_start, profiling_end);
                }

                if (probe_config.probe_mode == ProbeConfig::MODE_PARTITION_S && probe_config.get_max_table_elements() < r_table_size) {
                    std::cout << "Skipping config due table size" << std::endl;
                    continue;
                }
                double tuples_p_second_avg = 0.0;
                JoinStatus latest_join_status;
                for (int run_index = 0; run_index < benchmark_config.runs; run_index++) {
                    auto probe_start = std::chrono::high_resolution_clock::now();
                    for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                        probe_config.stream = streams[stream_index];
                        latest_join_status = build_and_probe_gpu(r_tables[stream_index], r_hash_tables[stream_index], s_tables[stream_index], s_hash_tables[stream_index], rs_tables[stream_index], 0, probe_config);
                        if (latest_join_status.has_failed()) {
                            for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                                rs_tables[stream_index].free();
                            }
                            std::cout << latest_join_status.message << std::endl;
                            continue;
                        }
                        store_probe_summary(probe_json, probe_config.profiling_summary);
                    }

                    for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                        gpuErrchk(cudaStreamSynchronize(streams[stream_index]));
                        // gpuErrchk(cudaGetLastError());
                    }

                    auto probe_end = std::chrono::high_resolution_clock::now();
                    double probe_duration = std::chrono::duration_cast<std::chrono::microseconds>(probe_end - probe_start).count() / pow(10, 6);
                    tuples_p_second_avg += stream_count * (s_tables[0].size + r_tables[0].size) / probe_duration;
                    // std::cout << " > " << tuples_p_second_avg << std::endl;

                    // rs_tables[0].print();
                    for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                        rs_tables[stream_index].free();
                    }

                    if (latest_join_status.has_failed()) {
                        break;
                    }
                }

                probe_config.free();

                if (latest_join_status.is_successful()) {
                    // max value might be clamped by zipf distribution
                    benchmark_config.max_column_value = max_column_value;

                    best_result.benchmark_config = benchmark_config;
                    best_result.probe_config = probe_benchmark_config;
                    best_result.join_config = join_benchmark_config;

                    tuples_p_second_avg /= benchmark_config.runs;
                    best_result.tuples_p_second = tuples_p_second_avg;
                    best_result.probe_elements = r_table_size;

                    run_csv_stream << best_result.to_string() << std::endl;
                }
            }

            for (int stream_index = 0; stream_index < stream_count; stream_index++) {
                r_tables[stream_index].free();
                s_tables[stream_index].free();
                rs_tables[stream_index].free();

                r_hash_tables[stream_index].free();
                s_hash_tables[stream_index].free();
            }

            delete[] r_tables;
            delete[] s_tables;
            delete[] rs_tables;
            delete[] r_hash_tables;
            delete[] s_hash_tables;
        }
    }

    for (int stream_index = 0; stream_index < stream_count; stream_index++) {
        cudaStreamDestroy(streams[stream_index]);
    }

    cudaEventDestroy(profiling_start);
    cudaEventDestroy(profiling_end);
    if (benchmark_setup.profile) {
        std::fstream run_json_stream;
        run_json_stream.open((benchmark_setup.output_file_path + "/run.json"), std::ios::app);
        run_json_stream << probe_json.dump(1);
        run_json_stream.close();
    }

    // r_table.print();
    // s_table.print();
    // rs_table.print();

    return 0;
}