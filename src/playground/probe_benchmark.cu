#include "join/join_provider.cuh"
#include "benchmark/data_generator.hpp"
#include "benchmark/configuration.hpp"

struct ProbeBenchmarkResult
{
    BenchmarkConfig benchmark_config;
    ProbeBenchmarkConfig probe_config;
    JoinBenchmarkConfig join_config;
    index_t probe_elements;
    double tuples_p_second = 0;

    static std::string to_string_header()
    {
        std::ostringstream string_stream;
        string_stream
            << BenchmarkConfig::to_string_header() << ","
            << JoinBenchmarkConfig::to_string_header() << ","
            << ProbeBenchmarkConfig::to_string_header() << ","
            << "probe_elements,"
            << "tuples_p_second";
        return string_stream.str();
    }

    std::string to_string()
    {
        std::ostringstream string_stream;
        string_stream
            << benchmark_config.to_string() << ","
            << join_config.to_string() << ","
            << probe_config.to_string() << ","
            << probe_elements << ","
            << tuples_p_second;
        return string_stream.str();
    }
};

void hash_table(db_table table, db_hash_table &hash_table)
{
    hash_table.size = table.size;
    hash_table.gpu = table.gpu;
    hash_table.hashes = new hash_t[hash_table.size];

    column_t *column_values = nullptr;
    if (table.gpu)
    {
        column_values = new column_t[table.size * table.column_count];
        gpuErrchk(cudaMemcpy(column_values, table.column_values, table.column_count * table.size * sizeof(column_t), cudaMemcpyDeviceToHost));
    }
    else
    {
        column_values = table.column_values;
    }

    // create hash
    for (index_t hash_index = 0; hash_index < hash_table.size; hash_index++)
    {
        hash_t hash_value = 0;
        for (size_t column_index = 1; column_index < table.column_count; column_index++)
        {
            hash_value += column_values[hash_index * table.column_count + column_index];
        }
        hash_table.hashes[hash_index] = hash_value;
    }

    // create indices
    // copy hashes to gpu if required
    if (table.gpu)
    {
        hash_t *d_hashes = nullptr;
        gpuErrchk(cudaMalloc(&d_hashes, hash_table.size * sizeof(hash_t)));
        gpuErrchk(cudaMemcpy(d_hashes, hash_table.hashes, hash_table.size * sizeof(hash_t), cudaMemcpyHostToDevice));
        delete[] hash_table.hashes;
        hash_table.hashes = d_hashes;
        delete[] column_values;
    }
}

int main(int argc, char **argv)
{

    std::srand(0);

    BenchmarkSetup benchmark_setup;
    if (!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup))
    {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    std::fstream run_csv_stream;
    run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
    run_csv_stream << ProbeBenchmarkResult::to_string_header() << std::endl;

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    std::cout << device_properties.asyncEngineCount << "," << device_properties.concurrentKernels << "," << device_properties.multiProcessorCount << std::endl;

    bool gpu = true;

    // fix to 1 for probe config preallocated data
    int stream_count = 1; //benchmark_setup.max_streams;
    cudaStream_t *streams = new cudaStream_t[stream_count];
    for (int stream_index = 0; stream_index < stream_count; stream_index++)
    {
        cudaStreamCreate(&streams[stream_index]);
    }

    std::vector<BenchmarkConfig> benchmark_configs = get_benchmark_configs(benchmark_setup);
    std::vector<ProbeBenchmarkConfig> probe_configs = get_probe_benchmark_configs(benchmark_setup);
    std::vector<JoinBenchmarkConfig> join_configs = get_join_configs(benchmark_setup);
    ProbeBenchmarkResult best_result;

    int config_index = 0;
    for (auto benchmark_config : benchmark_configs)
    {
        for (auto join_benchmark_config : join_configs)
        {
            for (auto probe_benchmark_config : probe_configs)
            {
                int column_count = join_benchmark_config.rs_columns;
                index_t s_table_size = benchmark_config.elements;
                index_t r_table_size = benchmark_config.elements / join_benchmark_config.rs_scale;
                

                //config.print();
                //print_mem();
                config_index++;
                printf("%d/%lu\n", config_index, probe_configs.size() * benchmark_configs.size() * join_configs.size());

                ProbeConfig probe_config;
                probe_config.build_n_per_thread = probe_benchmark_config.build_n_per_thread;
                probe_config.build_table_load = probe_benchmark_config.build_table_load;
                probe_config.build_threads = probe_benchmark_config.build_threads;
                probe_config.extract_threads = probe_benchmark_config.extract_threads;
                probe_config.extract_n_per_thread = probe_benchmark_config.extract_n_per_thread;
#if PROBE_MODE == 1
                if(probe_config.get_table_size(r_table_size) < device_properties.sharedMemPerBlock)
#endif
                {
                    double tuples_p_second_avg = 0.0;
                    for (int run_index = 0; run_index < benchmark_setup.runs; run_index++)
                    {
                        db_table *r_tables = new db_table[stream_count];
                        db_table *s_tables = new db_table[stream_count];
                        db_table *rs_tables = new db_table[stream_count];

                        db_hash_table *r_hash_tables = new db_hash_table[stream_count];
                        db_hash_table *s_hash_tables = new db_hash_table[stream_count];

                        for (int stream_index = 0; stream_index < stream_count; stream_index++)
                        {
                            generate_table(r_table_size, column_count, benchmark_config.element_max, r_tables[stream_index], gpu);
                            generate_table(s_table_size, column_count, benchmark_setup.element_max, s_tables[stream_index], gpu);

                            hash_table(r_tables[stream_index], r_hash_tables[stream_index]);
                            hash_table(s_tables[stream_index], s_hash_tables[stream_index]);
                        }

                        auto probe_start = std::chrono::high_resolution_clock::now();
                        for (int stream_index = 0; stream_index < stream_count; stream_index++)
                        {
                            probe_config.stream = streams[stream_index];
                            build_and_probe_gpu(r_tables[stream_index], r_hash_tables[stream_index], s_tables[stream_index], s_hash_tables[stream_index], rs_tables[stream_index], 0, probe_config);
                        }

                        for (int stream_index = 0; stream_index < stream_count; stream_index++)
                        {
                            gpuErrchk(cudaStreamSynchronize(streams[stream_index]));
                            //gpuErrchk(cudaGetLastError());
                        }

                        auto probe_end = std::chrono::high_resolution_clock::now();
                        double probe_duration = std::chrono::duration_cast<std::chrono::microseconds>(probe_end - probe_start).count() / pow(10, 6);
                        tuples_p_second_avg += stream_count * (s_tables[0].size + r_tables[0].size) / probe_duration;
                        //std::cout << " > " << tuples_p_second_avg << std::endl;

                        
                    
                        //rs_tables[0].print();

                        for (int stream_index = 0; stream_index < stream_count; stream_index++)
                        {
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

                    probe_config.free();
                    
                    best_result.benchmark_config = benchmark_config;
                    best_result.probe_config = probe_benchmark_config;
                    best_result.join_config = join_benchmark_config;

                    tuples_p_second_avg /= benchmark_setup.runs;
                    best_result.tuples_p_second = tuples_p_second_avg;
                    best_result.probe_elements = r_table_size;
                    

                    run_csv_stream << best_result.to_string() << std::endl;
                }
#if PROBE_MODE == 1
            }
#endif
        }
    }

    for (int stream_index = 0; stream_index < stream_count; stream_index++)
    {
        cudaStreamDestroy(streams[stream_index]);
    }

    //r_table.print();
    //s_table.print();
    //rs_table.print();

    return 0;
}