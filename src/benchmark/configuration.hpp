#pragma once
#include <toml.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <filesystem>

struct HashBenchmarkSetup {
    std::vector<int> hash_elements_sizes;
    std::vector<std::string> hash_algorithms;

};

struct BenchmarkSetup
{
    std::vector<int> gpus;
    std::vector<long long> elements;
    int runs = 0;
    short int max_streams = 1;
    index_t element_max = 1000;
    std::vector<float> rs_scales;
    std::vector<int> rs_join_columns;

    // hash
    HashBenchmarkSetup hash_setup;

    // probe
    // build
    std::vector<float> probe_build_table_loads;
    std::vector<int> probe_build_threads;
    std::vector<int> probe_build_n_per_threads;

    // extract
    std::vector<int> probe_extract_threads;
    std::vector<int> probe_extract_n_per_threads;

    // launch
    std::string output_file_path;
};

struct HashBenchmarkConfig {
    std::string algorithm;
    int element_size;


    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "elements_size,"
            << "algorithm";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << element_size << ","
            << algorithm;
        return string_stream.str();
    }
};

struct BenchmarkConfig {
    int gpus;
    long long elements;
    int runs;
    short int max_streams_p_gpu;
    hash_t element_max;
    float rs_scale;
    int rs_join_columns;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "runs,"
            << "gpus,"
            << "elements,"
            << "rs_scale,"
            << "rs_join_columns,"
            << "max_streams_p_gpu,"
            << "hash_bytes";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << runs << ","
            << gpus << ","
            << elements << ","
            << rs_scale << ","
            << rs_join_columns << ","
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
    for(auto element_size : setup.hash_setup.hash_elements_sizes) {
        for(auto algorithm : setup.hash_setup.hash_algorithms) {
            HashBenchmarkConfig config;
            config.element_size = element_size;
            config.algorithm = algorithm;
            configs.push_back(config);
        }
    }
    return configs;
}


std::vector<BenchmarkConfig> get_benchmark_configs(BenchmarkSetup setup) {
    std::vector<BenchmarkConfig> configs;
    for(auto gpus : setup.gpus) {
        for(auto rs_scale : setup.rs_scales) {
            for(auto rs_join_columns : setup.rs_join_columns) {
                for(auto elements : setup.elements) {
                    BenchmarkConfig config;
                    config.gpus = gpus;
                    config.element_max = setup.element_max;
                    config.elements = elements;
                    config.max_streams_p_gpu = setup.max_streams;
                    config.rs_scale = rs_scale;
                    config.rs_join_columns = rs_join_columns;
                    config.runs = setup.runs;
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
        setup->probe_build_table_loads = toml::find<std::vector<float>>(config_file, profile, field);
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
        setup->probe_build_n_per_threads = toml::find<std::vector<int>>(config_file, profile, field);
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
        setup->probe_build_threads = toml::find<std::vector<int>>(config_file, profile, field);
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
        setup->probe_extract_n_per_threads = toml::find<std::vector<int>>(config_file, profile, field);
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
        setup->probe_extract_threads = toml::find<std::vector<int>>(config_file, profile, field);
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
        setup->max_streams = toml::find<short int>(config_file, profile, field);
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
        setup->runs = toml::find<int>(config_file, profile, field);
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

        if (!load_probe_benchmark_setup(config_file, profile, setup))
        {
            return false;
        }

        if(!load_launch_benchmark_setup(config_file, profile, setup)) {
            return false;
        }

        if(!load_hash_benchmark_setup(config_file, profile, setup)) {
            return false;
        }

        std::string field = "elements";
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

        field = "element_max";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup->element_max = toml::find<index_t>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "rs_scale";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup->rs_scales = toml::find<std::vector<float>>(config_file, profile, field);
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
            setup->rs_join_columns = toml::find<std::vector<int>>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
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
