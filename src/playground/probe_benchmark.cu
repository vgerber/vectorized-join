#include "join/join_provider.cuh"
#include "benchmark/data_generator.hpp"
#include "benchmark/configuration.hpp"

struct ProbeBenchmarkConfig
{
    BenchmarkConfig benchmark_config;
    ProbeConfig config;
    index_t probe_elements;
    double tuples_p_second = 0;

    void print() {
        printf("\nBEST\nT=%E E=%llu\n", tuples_p_second, probe_elements);
        config.print();
        printf("\n");
    }

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream
            << BenchmarkConfig::to_string_header() << ","
            << ProbeConfig::to_string_header() << ","
            << "probe_elements,"
            << "tuples_p_second";
        return string_stream.str(); 
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream 
            << benchmark_config.to_string() << ","
            << config.to_string() << ","
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
        cudaMalloc(&hash_table.indices, hash_table.size * sizeof(index_t));
        hash_t *d_hashes = nullptr;
        cudaMalloc(&d_hashes, hash_table.size * sizeof(hash_t));
        cudaMemcpy(d_hashes, hash_table.hashes, hash_table.size * sizeof(hash_t), cudaMemcpyHostToDevice);
        delete[] hash_table.hashes;
        hash_table.hashes = d_hashes;
        delete[] column_values;
    }
    else
    {
        hash_table.indices = new index_t[hash_table.size];
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
    run_csv_stream << ProbeBenchmarkConfig::to_string_header() << std::endl;


    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    std::cout << device_properties.asyncEngineCount << "," << device_properties.concurrentKernels << "," << device_properties.multiProcessorCount << std::endl;

    bool gpu = true;

    int stream_count = benchmark_setup.max_streams;
    index_t table_size = 800;

    cudaStream_t *streams = new cudaStream_t[stream_count];
    for (int stream_index = 0; stream_index < stream_count; stream_index++)
    {
        cudaStreamCreate(&streams[stream_index]);
    }

    std::vector<BenchmarkConfig> benchmark_configs = get_benchmark_configs(benchmark_setup);
    std::vector<ProbeConfig> probe_configs = get_probe_configs(benchmark_setup);
    ProbeBenchmarkConfig best_config;

        

    int config_index = 0;
    for(auto benchmark_config : benchmark_configs) {
        int column_count = benchmark_config.rs_join_columns;
        for (auto config : probe_configs)
        {
            table_size = 800;
            //config.print();
            //print_mem();
            config_index++;
            printf("[%d|%lu] ", config_index, probe_configs.size() * benchmark_configs.size());
    #if PROBE_MODE == 0
            while (table_size < benchmark_setup.elements)
    #else
            while (config.get_table_size(table_size) < device_properties.sharedMemPerBlock)
    #endif
            {
                double tuples_p_second_avg = 0.0;
                for (int config_index = 0; config_index < benchmark_setup.runs; config_index++)
                {

                    db_table *r_tables = new db_table[stream_count];
                    db_table *s_tables = new db_table[stream_count];
                    db_table *rs_tables = new db_table[stream_count];

                    db_hash_table *r_hash_tables = new db_hash_table[stream_count];
                    db_hash_table *s_hash_tables = new db_hash_table[stream_count];

                    for (int stream_index = 0; stream_index < stream_count; stream_index++)
                    {
                        generate_table(table_size, column_count, benchmark_config.element_max, r_tables[stream_index], gpu);
                        generate_table(r_tables[stream_index].size * benchmark_config.rs_scale, column_count, benchmark_setup.element_max, s_tables[stream_index], gpu);

                        hash_table(r_tables[stream_index], r_hash_tables[stream_index]);
                        hash_table(s_tables[stream_index], s_hash_tables[stream_index]);
                    }

                    auto probe_start = std::chrono::high_resolution_clock::now();
                    for (int stream_index = 0; stream_index < stream_count; stream_index++)
                    {
                        build_and_probe_gpu(r_tables[stream_index], r_hash_tables[stream_index], s_tables[stream_index], s_hash_tables[stream_index], rs_tables[stream_index], 0, streams[stream_index], config);
                    }

                    for (int stream_index = 0; stream_index < stream_count; stream_index++)
                    {
                        cudaStreamSynchronize(streams[stream_index]);
                    }

                    auto probe_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> probe_dur = probe_end - probe_start;
                    tuples_p_second_avg += stream_count * (s_tables[0].size + r_tables[0].size) / (double)probe_dur.count();
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
                }
                tuples_p_second_avg /= benchmark_setup.runs;
                //if(best_config.tuples_p_second < tuples_p_second_avg) {
                    best_config.config = config;
                    best_config.probe_elements = table_size;
                    best_config.tuples_p_second = tuples_p_second_avg;
                    best_config.benchmark_config = benchmark_config;
                //}
                //std::cout << table_size << " \t" << tuples_p_second_avg << "\t\t T/s" << std::endl;

                run_csv_stream << best_config.to_string() << std::endl;
                table_size += 100;
            }
            best_config.print();
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