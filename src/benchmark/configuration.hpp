#pragma once
#include <toml.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <filesystem>

#include "base/types.hpp"

struct HashBenchmarkSetup {
    std::vector<int> hash_thread_sizes;
    std::vector<int> hash_elements_sizes;
    std::vector<std::string> hash_algorithms;
};

struct ProbeBenchmarkSetup {
    // build
    std::vector<float> probe_build_table_loads;
    std::vector<int> probe_build_threads;
    std::vector<int> probe_build_n_per_threads;

    // extract
    std::vector<int> probe_extract_threads;
    std::vector<int> probe_extract_n_per_threads;
};

struct JoinBenchmarkSetup {
    std::vector<float> rs_scales;
    std::vector<int> rs_join_columns;
    std::vector<int> vector_bytes_sizes;
    bool vectorize = false;
};

struct BenchmarkSetup
{
    // launch
    std::vector<int> gpus;
    std::vector<int> runs;
    std::vector<short int> max_streams;
    
    // data gen
    std::vector<long long> elements;
    std::vector<int> max_values;
    std::vector<float> skews;


    // verify algorithm results
    bool verify = false;
    
    // profile algorithm
    bool profile = false;
    
    // results
    std::string output_file_path;
    
    // join
    bool has_join_setup = false;
    JoinBenchmarkSetup join_setup;

    // hash
    bool has_hash_setup = false;
    HashBenchmarkSetup hash_setup;

    // probe
    bool has_probe_setup = false;
    ProbeBenchmarkSetup probe_setup;    
};

struct JoinBenchmarkConfig
{
    int rs_columns = 0;
    float rs_scale = 1.0;
    bool vectorize = false;
    int vector_bytes_size = 0;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "rs_columns,rs_scale";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << rs_columns << "," << rs_scale;
        return string_stream.str();
    }
};

struct HashBenchmarkConfig {
    std::string algorithm;
    int element_size;
    int thread_size;


    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "hash_elements_size,"
            << "hash_algorithm,"
            << "hash_threads";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << element_size << ","
            << algorithm << ","
            << thread_size;
        return string_stream.str();
    }
};

struct ProbeBenchmarkConfig {
    // build
    float build_table_load;
    int build_threads;
    int build_n_per_thread;

    // extract
    int extract_threads;
    int extract_n_per_thread;


    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "probe_build_table_load,"
            << "probe_build_threads,"
            << "probe_build_n_per_thread,"
            << "probe_extract_threads,"
            << "probe_extract_n_per_thread";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << build_table_load << ","
            << build_threads << ","
            << build_n_per_thread << ","
            << extract_threads << ","
            << extract_n_per_thread;
        return string_stream.str();
    }
};

struct BenchmarkConfig {
    int gpus;
    int runs;
    short int max_streams_p_gpu;
    bool verify;
    bool profile;
    
    // data gen
    long long elements;
    int max_value;
    float skew;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "runs,"
            << "gpus,"
            << "elements,"
            << "max_value,"
            << "skew,"
            << "max_streams_p_gpu,"
            << "hash_bytes";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << runs << ","
            << gpus << ","
            << elements << ","
            << max_value << ","
            << skew << ","
            << max_streams_p_gpu << ","
            << sizeof(hash_t);
        return string_stream.str();
    }
};

struct BenchmarkRunConfig
{
    int max_threads_per_gpu = 1024;
    uint max_blocks_per_gpu = std::numeric_limits<uint>::max();
    int max_gpus = 2;
    std::fstream output_file;

    ~BenchmarkRunConfig()
    {
        if (output_file.is_open())
        {
            output_file.close();
        }
    }
};

std::vector<HashBenchmarkConfig> get_hash_benchmark_configs(BenchmarkSetup setup) {
    std::vector<HashBenchmarkConfig> configs;
    if(setup.has_hash_setup) {
        for(auto thread_size : setup.hash_setup.hash_thread_sizes) {
            for(auto element_size : setup.hash_setup.hash_elements_sizes) {
                for(auto algorithm : setup.hash_setup.hash_algorithms) {
                    HashBenchmarkConfig config;
                    config.element_size = element_size;
                    config.algorithm = algorithm;
                    config.thread_size = thread_size;
                    configs.push_back(config);
                }
            }
        }
    }
    return configs;
}

std::vector<ProbeBenchmarkConfig> get_probe_benchmark_configs(BenchmarkSetup setup) {
    std::vector<ProbeBenchmarkConfig> configs;
    if(setup.has_probe_setup) {
        for(auto build_n_per_thread : setup.probe_setup.probe_build_n_per_threads) {
            for(auto build_threads : setup.probe_setup.probe_build_threads) {
                for(auto build_table_load : setup.probe_setup.probe_build_table_loads) {
                    for(auto extract_n_per_thread : setup.probe_setup.probe_extract_n_per_threads) {
                        for(auto extract_threads : setup.probe_setup.probe_extract_threads) {
                            ProbeBenchmarkConfig config;
                            config.build_n_per_thread = build_n_per_thread;
                            config.build_threads = build_threads;
                            config.build_table_load = build_table_load;
                            config.extract_n_per_thread = extract_n_per_thread;
                            config.extract_threads = extract_threads;
                            configs.push_back(config);
                        }
                    }
                }
            }
        }
    }
    return configs;
}


std::vector<BenchmarkConfig> get_benchmark_configs(BenchmarkSetup setup) {
    std::vector<BenchmarkConfig> configs;
    for(auto runs : setup.runs) {
        for(auto max_streams : setup.max_streams) {
            for(auto gpus : setup.gpus) {
                for(auto elements : setup.elements) {
                    for(auto max_value : setup.max_values) {
                        for(auto skew : setup.skews) {
                            BenchmarkConfig config;
                            config.gpus = gpus;
                            config.elements = elements;
                            config.max_value = max_value;
                            config.skew = skew;
                            config.max_streams_p_gpu = max_streams;
                            config.runs = runs;
                            config.verify = setup.verify;
                            config.profile = setup.profile;
                            configs.push_back(config);
                        }
                    }            
                }
            }
        }
    }
    return configs;
}

std::vector<JoinBenchmarkConfig> get_join_configs(BenchmarkSetup setup) {
    std::vector<JoinBenchmarkConfig> configs;
    
    if(setup.has_join_setup) {
        for(auto rs_scale : setup.join_setup.rs_scales) {
            for(auto rs_columns : setup.join_setup.rs_join_columns) {
                JoinBenchmarkConfig config;
                config.vectorize = setup.join_setup.vectorize;
                config.rs_columns = rs_columns;
                config.rs_scale = rs_scale;
                if(setup.join_setup.vectorize) {
                    for(auto vector_bytes_size : setup.join_setup.vector_bytes_sizes) {
                        config.vector_bytes_size = vector_bytes_size;
                        configs.push_back(config);
                    }
                } else {
                    config.vector_bytes_size = 0;
                    configs.push_back(config);
                }
            }
        }
    }
    return configs;
}

bool load_probe_benchmark_setup(toml::value config_file, std::string profile, BenchmarkSetup *setup)
{

    // build
    std::string field = "probe_build_table_loads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->probe_setup.probe_build_table_loads = toml::find<std::vector<float>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "probe_build_n_per_threads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->probe_setup.probe_build_n_per_threads = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "probe_build_threads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->probe_setup.probe_build_threads = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    // extract
    field = "probe_extract_n_per_threads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->probe_setup.probe_extract_n_per_threads = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "probe_extract_threads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->probe_setup.probe_extract_threads = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    return true;
}

bool load_hash_benchmark_setup(toml::value config_file, std::string profile, BenchmarkSetup *setup) {
    std::string field = "hash_elements_sizes";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->hash_setup.hash_elements_sizes = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "hash_algorithms";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->hash_setup.hash_algorithms = toml::find<std::vector<std::string>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "hash_threads";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->hash_setup.hash_thread_sizes = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }
    return true;
}

bool load_join_benchmark_setup(toml::value config_file, std::string profile, BenchmarkSetup *setup) {
    std::string field = "rs_scale";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->join_setup.rs_scales = toml::find<std::vector<float>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "rs_join_columns";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->join_setup.rs_join_columns = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "vectorize";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->join_setup.vectorize = toml::find<bool>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    if(setup->join_setup.vectorize) {
        field = "vector_bytes_sizes";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup->join_setup.vector_bytes_sizes = toml::find<std::vector<int>>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }
    }

    return true;
}

bool load_launch_benchmark_setup(toml::value config_file, std::string profile, BenchmarkSetup *setup)
{

    std::string field = "gpus";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->gpus = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "max_streams";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->max_streams = toml::find<std::vector<short int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "runs";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->runs = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "elements";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->elements = toml::find<std::vector<long long>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "max_values";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->max_values = toml::find<std::vector<int>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "skews";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->skews = toml::find<std::vector<float>>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "verify";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->verify = toml::find<bool>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    field = "profile";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        setup->profile = toml::find<bool>(config_file, profile, field);
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }

    return true;
}

bool load_benchmark_results_location(toml::value config_file, std::string profile, BenchmarkSetup *setup) {
    std::string field = "output";
    std::cout << "Read " << field << std::endl;
    if (config_file.at(profile).contains(field))
    {
        std::string output_file_path = toml::find<std::string>(config_file, profile, field);
        setup->output_file_path = output_file_path;

        // test path
        if(std::filesystem::exists(output_file_path)) {
            std::cout << profile << "." << field << output_file_path << " already exists" << std::endl;
            return false;
        } else {
            std::filesystem::create_directory(output_file_path);
        }
    }
    else
    {
        std::cout << profile << "." << field << " not found" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Loads the setup for the whole benchmark
 * 
 * @param path Path of config file
 * @param profile Profile for benchmark
 * @param setup Parsed config
 * @return true When setup config 
 * @return false When loading failed
 */
bool load_benchmark_setup(std::string path, std::string profile, BenchmarkSetup *setup)
{
    toml::value config_file;
    try
    {
        config_file = toml::parse(path);
    }
    catch (std::runtime_error err)
    {
        std::cout << err.what() << std::endl;
        return false;
    }

    if (config_file.contains(profile))
    {
        setup->has_probe_setup = load_probe_benchmark_setup(config_file, profile, setup);
        setup->has_hash_setup = load_hash_benchmark_setup(config_file, profile, setup);
        setup->has_join_setup = load_join_benchmark_setup(config_file, profile, setup);

        if(!load_launch_benchmark_setup(config_file, profile, setup)) {
            return false;
        }

        if(!load_benchmark_results_location(config_file, profile, setup)) {
            return false;
        }
    }
    else
    {
        std::cout << profile << " not found" << std::endl;
    }
    return true;
}
