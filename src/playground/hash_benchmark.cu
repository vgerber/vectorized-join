#include <nlohmann/json.hpp>
#include <vector>
#include <curand.h>

#include "hash/hash.cu"
#include "benchmark/configuration.hpp"


using json = nlohmann::json;

struct HashBenchmarkResult
{
    BenchmarkConfig benchmark_config;
    HashBenchmarkConfig hash_benchmark_config;
    double hash_p_second = -1.0;
    double gb_p_second = -1.0;
    char chunk_size = 0;
    int run_id = -1;
    int blocks = 0;
    int threads = 0;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream
            << config_to_string_header() << ","
            << "hash_p_second,"
            << "gb_p_second";
        return string_stream.str(); 
    }

    static std::string config_to_string_header() {
        std::ostringstream string_stream;
        string_stream
            << "run_id,"
            << "blocks,"
            << "threads,"
            << BenchmarkConfig::to_string_header() << ","
            << HashBenchmarkConfig::to_string_header() << ","
            << "chunk_size";
        return string_stream.str(); 
    }

    std::string config_to_string() {
        std::ostringstream string_stream;
        string_stream 
            << run_id << ","
            << blocks << ","
            << threads << ","
            << benchmark_config.to_string() << ","
            << hash_benchmark_config.to_string() << ","
            << (int)chunk_size;
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream 
            << config_to_string() << ","
            << hash_p_second << ","
            << gb_p_second;
        return string_stream.str();
    }

    void print() {
        //printf("Runtime %e ms %e H/s %e Gb/s\n", run_time_avg, (config.elements / run_time_avg), (element_buffer_size * element_size / run_time_avg));
    }
};


__global__
void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t* element_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t data_chunk;
            data_chunk.x = element_index;
            /*
            data_chunk.y = element_index;
            data_chunk.z = element_index;
            data_chunk.w = element_index;
            */
            chunked_element_buffer[chunk_index] = data_chunk;
        }
    }
}

__global__
void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t* element_buffer, float * distribution_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    uint32_t* element_buffer32 = reinterpret_cast<uint32_t*>(element_buffer);
    index_t total_elements = element_buffer_size * element_chunks * (sizeof(chunk_t) / sizeof(uint32_t));
    for(index_t element_index = index; element_index < total_elements; element_index += stride) {
        element_buffer32[element_index] = fabsf(distribution_values[element_index]) * UINT32_MAX;
    }
}

__global__
void calculate_distribution_kernel(index_t element_buffer_size, hash_t * hash_buffer, hash_t max_value, hash_t * hash_max, hash_t * hash_min, index_t * hash_collisions, short int sections, unsigned long long * section_buffer, short int buckets, int * bucket_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    hash_t section_size = max_value / sections;
    for(index_t element_index = index; element_index < element_buffer_size * element_buffer_size; element_index += stride) {
        index_t hash_index = element_index / element_buffer_size;
        index_t hash_ref_index = element_index % element_buffer_size;
        hash_t hash = hash_buffer[hash_index];
        
        if(hash_ref_index == 0) {
            
            short int bucket_index =  hash % buckets;
            short int section_index = hash / section_size;
            atomicAdd(&bucket_buffer[bucket_index], 1);
            atomicAdd(&section_buffer[section_index], 1);            
            atomicMin(hash_min, hash);
            atomicMax(hash_max, hash);
        }

        if(hash_index < hash_ref_index) {
            hash_t hash_ref = hash_buffer[hash_ref_index];
            if(hash == hash_ref) {
                atomicAdd(hash_collisions, 1);
            }
        }
    }
}

__global__
void calculate_distribution_base_kernel(index_t element_buffer_size, hash_t * hash_buffer, hash_t max_value, hash_t * hash_max, hash_t * hash_min, index_t * hash_collisions, short int sections, unsigned long long * section_buffer, short int buckets, int * bucket_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    hash_t section_size = max_value / sections;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = hash_buffer[element_index];
        short int bucket_index =  hash % buckets;
        short int section_index = hash / section_size;
        atomicAdd(&bucket_buffer[bucket_index], 1);
        atomicAdd(&section_buffer[section_index], 1);            
        atomicMin(hash_min, hash);
        atomicMax(hash_max, hash);
    }
}


void calculate_distribution(index_t elements, hash_t * hash_buffer, hash_t & hash_min, hash_t & hash_max, index_t &hash_collision, short int buckets, int ** bucket_buffer, short int sections, unsigned long long ** section_buffer) {
    int * d_bucket_buffer = nullptr;
    gpuErrchk(cudaMalloc(&d_bucket_buffer, buckets * sizeof(int)));
    gpuErrchk(cudaMemset(d_bucket_buffer, 0, buckets * sizeof(int)));
    unsigned long long * d_section_buffer = nullptr;
    gpuErrchk(cudaMalloc(&d_section_buffer, sections * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_section_buffer, 0, sections * sizeof(unsigned long long)));

    hash_t * d_hash_max = nullptr;
    hash_t * d_hash_min = nullptr;
    index_t * d_hash_collision = nullptr;
    gpuErrchk(cudaMalloc(&d_hash_max, sizeof(hash_t)));
    gpuErrchk(cudaMalloc(&d_hash_min, sizeof(hash_t)));
    gpuErrchk(cudaMalloc(&d_hash_collision, sizeof(index_t)));
    gpuErrchk(cudaMemset(d_hash_max, 0, sizeof(hash_t)));
    gpuErrchk(cudaMemcpy(d_hash_min, &hash_min, sizeof(hash_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_hash_collision, 0, sizeof(index_t)));


    int threads = 256;
    if(elements <= 10000) {
        int blocks = min(max(elements * elements / threads, (index_t)1), 10000ULL);
        calculate_distribution_kernel<<<blocks, threads>>>(elements, hash_buffer, std::numeric_limits<hash_t>::max(), d_hash_max, d_hash_min, d_hash_collision, sections, d_section_buffer, buckets, d_bucket_buffer);
    } else {
        int blocks = max(elements / threads, (index_t)1);
        calculate_distribution_base_kernel<<<blocks, threads>>>(elements, hash_buffer, std::numeric_limits<hash_t>::max(), d_hash_max, d_hash_min, d_hash_collision, sections, d_section_buffer, buckets, d_bucket_buffer);
    }

    *bucket_buffer = new int[buckets];
    gpuErrchk(cudaMemcpy(*bucket_buffer, d_bucket_buffer, buckets * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_bucket_buffer));
    
    *section_buffer = new unsigned long long[sections];
    gpuErrchk(cudaMemcpy(*section_buffer, d_section_buffer, sections * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_section_buffer));

    gpuErrchk(cudaMemcpy(&hash_max, d_hash_max, sizeof(hash_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&hash_min, d_hash_min, sizeof(hash_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&hash_collision, d_hash_collision, sizeof(index_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_hash_max));
    gpuErrchk(cudaFree(d_hash_min));
    gpuErrchk(cudaFree(d_hash_collision));
}

void generate_demo_data(index_t elements, short int element_size, short int *element_chunks, chunk_t** buffer) {
    *element_chunks = element_size / sizeof(chunk_t) + (element_size % sizeof(chunk_t) > 0);
    cudaMalloc(buffer, elements * *element_chunks * sizeof(chunk_t));

#define USE_CURAND 0

#if USE_CURAND
    float * d_distribution = nullptr;
    index_t distribution_values_count = elements * *element_chunks * (sizeof(chunk_t) / sizeof(uint32_t));
    gpuErrchk(cudaMalloc(&d_distribution, distribution_values_count * sizeof(float)));

    curandGenerator_t rand_gen;
    gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, 101ULL));
    //gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, distribution_values_count));
    //gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.01));
    //gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.3));
    gpuErrchk(curandDestroyGenerator(rand_gen));

    generate_demo_data_kernel<<<max(elements/256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer, d_distribution);
    
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_distribution));
#else    
    
    generate_demo_data_kernel<<<max(elements/256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer);
#endif
}

int main(int argc, char* argv[]) {

    std::srand(0);

    bool verbose = false;
    if(argc >= 4) {
        verbose = std::string(argv[3]) == "-v";
    }

    BenchmarkSetup benchmark_setup;
    if (!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup) && !verbose)
    {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    std::fstream run_csv_stream;
    if(!verbose) {
        run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
        run_csv_stream << HashBenchmarkResult::to_string_header() << std::endl;
    }
    cudaDeviceReset();

    if(verbose) {
        std::cout << HashBenchmarkResult::config_to_string_header() << std::endl;
    }

    json distr_json;

    int run_id = 0;



    auto benchmark_configs = get_benchmark_configs(benchmark_setup);
    auto hash_benchmark_configs = get_hash_benchmark_configs(benchmark_setup);
    for(auto benchmark_config : benchmark_configs) {
        for(auto hash_benchmark_config : hash_benchmark_configs) {
            float run_time_avg = 0.0;
            index_t element_buffer_size = benchmark_config.elements;
            short int element_size = hash_benchmark_config.element_size;
            short int element_chunks = 0;

            int threads = hash_benchmark_config.thread_size;
            int blocks = max(element_buffer_size / threads, (index_t)1);

            for(int run_index = 0; run_index < benchmark_config.runs; run_index++) {
                chunk_t* d_element_buffer = nullptr;
                hash_t * d_hashed_buffer = nullptr;
                
                //print_mem();
                gpuErrchk(cudaMalloc(&d_hashed_buffer, element_buffer_size * sizeof(hash_t)));
                generate_demo_data(element_buffer_size, element_size, &element_chunks, &d_element_buffer);
                //std::cout << d_hashed_buffer << " " << d_element_buffer;

                
                gpuErrchk(cudaGetLastError());
                cudaEvent_t hash_start, hash_end;
                cudaEventCreate(&hash_start);
                cudaEventCreate(&hash_end);

                HashConfig hash_config;
                hash_config.algorithm = hash_benchmark_config.algorithm;
                hash_config.threads_per_block = hash_benchmark_config.thread_size;

                cudaEventRecord(hash_start);
                hash_func(element_buffer_size, 0, element_chunks, d_element_buffer, d_hashed_buffer, hash_config, 0);
                cudaEventRecord(hash_end);

                gpuErrchk(cudaDeviceSynchronize());

                float elapsed_time = 0.0;
                cudaEventElapsedTime(&elapsed_time, hash_start, hash_end);
                run_time_avg += (elapsed_time / 1000.0f);

                if(!verbose) {
                    short int distribution_buckets = 1009;
                    short int sections = 100;
                    int * hash_distribution = nullptr;
                    unsigned long long * hash_sections = nullptr;
                    hash_t hash_max = 0;
                    hash_t hash_min = std::numeric_limits<hash_t>::max();
                    index_t hash_collisions = 0;
                    calculate_distribution(element_buffer_size, d_hashed_buffer, hash_min, hash_max, hash_collisions, distribution_buckets, &hash_distribution, sections, &hash_sections);

                    json run_json;
                    run_json["hash_max"] = hash_max;
                    run_json["hash_min"] = hash_min;
                    run_json["hash_collisions"] = hash_collisions;
                    run_json["hist_part_distribution"] = json(std::vector<int>(hash_distribution, hash_distribution+distribution_buckets));
                    run_json["hist_sections"] = json(std::vector<int>(hash_sections, hash_sections+sections));
                    distr_json[std::to_string(run_id)] = run_json;
                    delete[] hash_distribution;
                    delete[] hash_sections;
                    std::cout << "Run " << run_id << "/" << (benchmark_configs.size() * hash_benchmark_configs.size() * benchmark_config.runs - 1) << std::endl;
                }

                cudaFree(d_element_buffer);
                cudaFree(d_hashed_buffer);
                cudaEventDestroy(hash_start);
                cudaEventDestroy(hash_end);

                if(verbose) {
                    HashBenchmarkResult tmp_result;
                    tmp_result.benchmark_config = benchmark_config;
                    tmp_result.hash_benchmark_config = hash_benchmark_config;
                    tmp_result.run_id = run_id;
                    tmp_result.chunk_size = sizeof(chunk_t);
                    std::cout << tmp_result.config_to_string() << std::endl;
                }
                run_id++;
            }

            if(!verbose) {
                run_time_avg /= benchmark_config.runs;
                HashBenchmarkResult hash_result;
                hash_result.run_id = -1;
                hash_result.gb_p_second = ((element_buffer_size * element_size) / run_time_avg) / pow(10, 9);
                hash_result.hash_p_second = (element_buffer_size / run_time_avg);
                hash_result.benchmark_config = benchmark_config;
                hash_result.hash_benchmark_config = hash_benchmark_config;
                hash_result.chunk_size = sizeof(chunk_t);
                hash_result.blocks = blocks;
                hash_result.threads = threads;

                run_csv_stream << hash_result.to_string() << std::endl;
            }
        }
    }

    if(!verbose) {
        std::fstream run_json_stream;
        run_json_stream.open((benchmark_setup.output_file_path + "/run.json"), std::ios::app);
        run_json_stream << distr_json.dump(-1);
        run_json_stream.close();
    }

    run_csv_stream.close();
}