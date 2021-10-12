#include <curand.h>
#include <limits>
#include "benchmark/configuration.hpp"

typedef unsigned int value32_t;
typedef unsigned long long value64_t;

struct HistogramBenchmarkConfig {
    int runs = 1;
    int threads_per_block = 256;
    int elements_per_thread = 1;
    int elements;
    int bins;
    int bin_bytes;
    int element_bytes;

    HistogramBenchmarkConfig() {
        runs = 1;
        threads_per_block = 256;
        elements_per_thread = 1;
        elements = 0;
        bins = 0;
        bin_bytes = 0;
        element_bytes = 0;
    }

    HistogramBenchmarkConfig(int runs, int threads_per_block, int elements_per_thread, int elements, int bins, int bin_bytes, int element_bytes) {
        this->runs = runs;
        this->threads_per_block = threads_per_block;
        this->elements_per_thread = elements_per_thread;
        this->elements = elements;
        this->bins = bins;
        this->bin_bytes = bin_bytes;
        this->element_bytes = element_bytes;
    }

    static std::string to_string_header()
    {
        std::ostringstream string_stream;
        string_stream
            << "threads_per_block,"
            << "elements_per_thread,"
            << "elements,"
            << "bins,"
            << "bin_bytes,"
            << "element_bytes";
        return string_stream.str();
    }

    std::string to_string()
    {
        std::ostringstream string_stream;
        string_stream
            << threads_per_block << ","
            << elements_per_thread << ","
            << elements << ","
            << bins << ","
            << bin_bytes << ","
            << element_bytes;
        return string_stream.str();
    }

};

struct HistogramBenchmarkResult
{
    double elements_p_second = 0;
    double gb_p_second = 0;
    
    HistogramBenchmarkConfig config;

    static std::string to_string_header()
    {
        std::ostringstream string_stream;
        string_stream
            << HistogramBenchmarkConfig::to_string_header() << ","
            << "gb_p_second,"
            << "elements_p_second";
        return string_stream.str();
    }

    std::string to_string()
    {
        std::ostringstream string_stream;
        string_stream
            << config.to_string() << ","
            << gb_p_second << ","
            << elements_p_second;
        return string_stream.str();
    }
};


struct HistogramBenchmarkSetup
{
    std::vector<int> elements;
    std::vector<int> bins;
    std::vector<int> bin_bytes = { 4, 8 };
    std::vector<int> element_bytes = { 4, 8 };
    std::vector<int> threads_per_block;
    std::vector<int> element_per_thread;
    int runs = 0;

    std::string output_file_path;

    std::vector<HistogramBenchmarkConfig> get_configs() {
        std::vector<HistogramBenchmarkConfig> configs;
        for(auto threads_per_block_count : threads_per_block) {
            for(auto elements_per_thread_count : element_per_thread) {
                for(auto element_count : elements) {
                    for(auto bin_count : bins) {
                        for(auto bin_byte_count : bin_bytes) {
                            for(auto element_byte_count : element_bytes) {
                                configs.push_back(HistogramBenchmarkConfig(runs, threads_per_block_count, elements_per_thread_count, element_count, bin_count, bin_byte_count, element_byte_count));
                            }
                        }
                    }
                }
            }
        }
        return configs;
    }
};

bool load_histogram_setup(std::string path, std::string profile, HistogramBenchmarkSetup &setup) {

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

        std::string field = "threads_per_block";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup.threads_per_block = toml::find<std::vector<int>>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "elements_per_thread";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup.element_per_thread = toml::find<std::vector<int>>(config_file, profile, field);
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
            setup.elements = toml::find<std::vector<int>>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "bins";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            setup.bins = toml::find<std::vector<int>>(config_file, profile, field);
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
            setup.runs = toml::find<int>(config_file, profile, field);
        }
        else
        {
            std::cout << profile << "." << field << " not found" << std::endl;
            return false;
        }

        field = "output";
        std::cout << "Read " << field << std::endl;
        if (config_file.at(profile).contains(field))
        {
            std::string output_file_path = toml::find<std::string>(config_file, profile, field);
            setup.output_file_path = output_file_path;

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
    else
    {
        std::cout << profile << " not found" << std::endl;
    }
    return true;
}

template<class B_T, class H_T>
__global__ void histogram_kernel(int buffer_size, B_T *buffer, int bins, H_T *histogram)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int element_index = index; element_index < buffer_size; element_index += stride)
    {
        int key = buffer[element_index] % bins;
        atomicAdd(&histogram[key], (H_T)1);
    }
}


template<class T>
__global__ void scale_demodata_kernel(int buffer_size, T *buffer, float *scale_factor_buffer, T max_value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int element_index = index; element_index < buffer_size; element_index += stride) {
        buffer[element_index] = max_value * scale_factor_buffer[element_index];
    }
}

template<class T>
void generate_demo_data(int buffer_size, T *buffer, curandGenerator_t rand_gen) {
    float *d_distribution = nullptr;
    gpuErrchk(cudaMalloc(&d_distribution, buffer_size * sizeof(float)));
    gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, buffer_size));   
    
    scale_demodata_kernel<<<max(buffer_size/256, 1), 256>>>(buffer_size, buffer, d_distribution, std::numeric_limits<T>::max());
    gpuErrchk(cudaDeviceSynchronize());    
    gpuErrchk(cudaFree(d_distribution));
}

int main(int argc, char **argv)
{
    // verify type size
    // uint64_t is an alias for unsigned long but cuda requires unsigned long long which has the same size on the test machine
    // therefore 8 byte types have to defined by yourself
    if(sizeof(value32_t) != 4) {
        std::cout << "Invalid 4 byte type size" << std::endl;
        return -1;
    }

    if(sizeof(value64_t) != 8) {
        std::cout << "Invalid 8 byte type size" << std::endl;
        return -1;
    }

    std::srand(0);
    curandGenerator_t rand_gen;
    gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));

    HistogramBenchmarkSetup benchmark_setup;
    if (!load_histogram_setup(std::string(argv[1]), std::string(argv[2]), benchmark_setup))
    {
        std::cout << "Failed to load config" << std::endl;
        //return -1;
    }

    std::fstream run_csv_stream;
    run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
    run_csv_stream << HistogramBenchmarkResult::to_string_header() << std::endl;

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);


    auto benchmark_configs = benchmark_setup.get_configs();

    cudaEvent_t profiling_start, profiling_end;
    cudaEventCreate(&profiling_start);
    cudaEventCreate(&profiling_end);

    int config_index = 0;
    for (auto benchmark_config : benchmark_configs)
    {
        float total_runtime_s = 0.0f;
        int elements = benchmark_config.elements;
        int bins = benchmark_config.bins;

        std::cout << (config_index + 1) << "/" << benchmark_configs.size() << std::endl;

        for(int run_index = 0; run_index < benchmark_config.runs; run_index++) {
            if(benchmark_config.bin_bytes == 4) {
                if(benchmark_config.element_bytes == 4) {
                    value32_t * element_buffer = nullptr;
                    cudaMalloc(&element_buffer, elements * sizeof(value32_t));
                    generate_demo_data(elements, element_buffer, rand_gen);
                    value32_t * histogram_buffer = nullptr;
                    cudaMalloc(&histogram_buffer, bins * sizeof(value32_t));
                    cudaMemset(histogram_buffer, 0, bins * sizeof(value32_t));

                    cudaEventRecord(profiling_start);
                    int threads = benchmark_config.threads_per_block;
                    int blocks = max(1, elements / threads / benchmark_config.elements_per_thread);
                    histogram_kernel<<<blocks, threads>>>(elements, element_buffer, bins, element_buffer);
                    cudaEventRecord(profiling_end);                

                    cudaFree(element_buffer);
                    cudaFree(histogram_buffer);
                } else if(benchmark_config.element_bytes == 8) {
                    value64_t * element_buffer = nullptr;
                    cudaMalloc(&element_buffer, elements * sizeof(value64_t));
                    generate_demo_data(elements, element_buffer, rand_gen);
                    value32_t * histogram_buffer = nullptr;
                    cudaMalloc(&histogram_buffer, bins * sizeof(value32_t));
                    cudaMemset(histogram_buffer, 0, bins * sizeof(value32_t));

                    cudaEventRecord(profiling_start);
                    int threads = benchmark_config.threads_per_block;
                    int blocks = max(1, elements / threads / benchmark_config.elements_per_thread);
                    histogram_kernel<<<blocks, threads>>>(elements, element_buffer, bins, element_buffer);
                    cudaEventRecord(profiling_end);                

                    cudaFree(element_buffer);
                    cudaFree(histogram_buffer);
                }
            } else if(benchmark_config.bin_bytes == 8){
                if(benchmark_config.element_bytes == 4) {
                    value32_t * element_buffer = nullptr;
                    cudaMalloc(&element_buffer, elements * sizeof(value32_t));
                    generate_demo_data(elements, element_buffer, rand_gen);
                    value64_t * histogram_buffer = nullptr;
                    cudaMalloc(&histogram_buffer, bins * sizeof(value64_t));
                    cudaMemset(histogram_buffer, 0, bins * sizeof(value64_t));

                    cudaEventRecord(profiling_start);
                    int threads = benchmark_config.threads_per_block;
                    int blocks = max(1, elements / threads / benchmark_config.elements_per_thread);
                    histogram_kernel<<<blocks, threads>>>(elements, element_buffer, bins, element_buffer);
                    cudaEventRecord(profiling_end);                

                    cudaFree(element_buffer);
                    cudaFree(histogram_buffer);
                } else if(benchmark_config.element_bytes == 8) {
                    value64_t * element_buffer = nullptr;
                    cudaMalloc(&element_buffer, elements * sizeof(value64_t));
                    generate_demo_data(elements, element_buffer, rand_gen);
                    value64_t * histogram_buffer = nullptr;
                    cudaMalloc(&histogram_buffer, bins * sizeof(value64_t));
                    cudaMemset(histogram_buffer, 0, bins * sizeof(value64_t));

                    cudaEventRecord(profiling_start);
                    int threads = benchmark_config.threads_per_block;
                    int blocks = max(1, elements / threads / benchmark_config.elements_per_thread);
                    histogram_kernel<<<blocks, threads>>>(elements, element_buffer, bins, element_buffer);
                    cudaEventRecord(profiling_end);                

                    cudaFree(element_buffer);
                    cudaFree(histogram_buffer);
                }
            }
            cudaEventSynchronize(profiling_end);
            float runtime_ms;
            cudaEventElapsedTime(&runtime_ms, profiling_start, profiling_end);
            total_runtime_s += runtime_ms / pow(10, 3);
        }

        total_runtime_s /= benchmark_config.runs;

        HistogramBenchmarkResult result;
        result.config = benchmark_config;
        result.elements_p_second = elements / total_runtime_s;
        result.gb_p_second = ((unsigned long long)elements * benchmark_config.element_bytes) / total_runtime_s / pow(10, 9);
        run_csv_stream << result.to_string() << std::endl;

        config_index++;
    }

    cudaEventDestroy(profiling_start);
    cudaEventDestroy(profiling_end);

    gpuErrchk(curandDestroyGenerator(rand_gen));
    //r_table.print();
    //s_table.print();
    //rs_table.print();

    return 0;
}