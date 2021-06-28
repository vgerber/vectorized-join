#pragma once
#include <toml.hpp>
#include <iostream>
#include <limits>
#include <memory>


struct BenchmarkSetup
{
    std::vector<int> threads;
    std::vector<uint> blocks;
    std::vector<int> gpus;
    long elements = 0;
    std::string output_file_path;
    int runs = 0;
};


struct BenchmarkRunConfig
{
    int max_threads_per_gpu = 1024;
    uint max_blocks_per_gpu = std::numeric_limits<uint>::max();
    int max_gpus = 2;
    std::fstream output_file;

    ~BenchmarkRunConfig() {
        if(output_file.is_open()) {
            output_file.close();
        }
    }
};


/**
 * @brief Loads the setup for the whole benchmark
 * 
 * @param path Path of config file
 * @param profile Profile for benchmark
 * @param setup Parsed config
 * @return true When setup config 
 * @return false When loading failed
 */
bool load_benchmark_setup(std::string path, std::string profile, BenchmarkSetup *setup) {
    toml::value config_file;
    try {
        config_file = toml::parse(path);
    } catch(std::runtime_error err) {
        std::cout << err.what() << std::endl;
        return false;
    }

    if(config_file.contains(profile)) {
        std::string field = "threads";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            setup->threads = toml::find<std::vector<int>>(config_file, profile, field);
        } else {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "blocks";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            setup->blocks = toml::find<std::vector<uint>>(config_file, profile, field);
        } else {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "gpus";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            setup->gpus = toml::find<std::vector<int>>(config_file, profile, field);
        } else {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "elements";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            setup->elements = toml::find<long>(config_file, profile, field);
        } else {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "runs";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            setup->runs = toml::find<int>(config_file, profile, field);
        } else {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "output";
        std::cout << "Read " << field << std::endl;
        if(config_file.at(profile).contains(field)) {
            std::string output_file_path = toml::find<std::string>(config_file, profile, field);
            setup->output_file_path = output_file_path;

            // test path
            std::fstream output_file(output_file_path, std::ios::out);
            if(!output_file.is_open()) {
                std::cout << profile << "." << field << " could not open " << output_file_path << std::endl;
                return false;
            }
            output_file.close();
        } else { 
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

    } else {
        std::cout << profile << " not found" << std::endl;
    }
    return true;
}

