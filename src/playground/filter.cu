#include <iostream>
#include <fstream>
#include "benchmark/configuration.hpp"
#include "benchmark/log.hpp"
#include "base/types.hpp"


/**
 * @brief Fills a bitmask with the result of the filter predicate
 * 
 * @tparam F Predicate function type
 * @param size Size of input array
 * @param input Input array
 * @param reference Reference value for predicate
 * @param filter_result Resulting bitmask
 * @param f Filter predicate
 */
template <class F>
__global__
void filter_kernel(int size, filter_int *input, filter_int reference, filter_mask *filter_result, F f) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = index; i < size; i += stride) {
        filter_result[i] = f(input[i], reference);
    }
}

/**
 * @brief Evaluates the input values based on the predicated with reference and returns the results in a bitmask.
 * Compiler flag --extended-lambda required
 * @tparam F Predicate function type (must have __device__ declaration)
 * @param input_size Number of values
 * @param input_values Input values
 * @param output_mask Output bitmask
 * @param reference Reference value for filter operation
 * @param predicate Predicate for filter evaluation
 */
template<class F>
void filter(int input_size, filter_int *input_values, filter_mask* output_mask, filter_int reference, F predicate, BenchmarkRunConfig &benchmark_config) {

    // setup device resources
    // get devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    device_count = std::min(device_count, benchmark_config.max_gpus);
    if(device_count == 0) {
        std::cout << "No device found" << std::endl;
        exit(-1);
    }
#if DEBUG_PRINT
    std::cout << device_count << " Device" << (device_count != 1 ? "s" : "") << std::endl;
#endif
    // declare streams, buffers, events
    /* Event setup
     * [0 - 1] Start - Stop before and after memory operations
     * [2 - 3] Start - Stop before and after filte kernel
     */

    cudaStream_t *streams = new cudaStream_t[device_count];
#if BENCHMARK_TIME
    const int events_per_gpu = 4;
    cudaEvent_t *events = new cudaEvent_t[device_count * events_per_gpu];
#endif
    filter_mask **device_d_filter_results = new filter_mask*[device_count];
    filter_int **device_d_inputs = new filter_int*[device_count];
    int *device_input_size = new int[device_count];
    int total_mp_count = 0;
    int threads_per_gpu = 0;
    int blocks_per_gpu = 0;

    // start kernel on each device (async)
    // 1. Declare stream + events
    // 2. Read device prop and define kernel params
    // 3. Assign data chunk
    // 4. Allocate memory on device
    // 5. Transfer input values to device
    // 6. Execute kernel
    // 7. Transfer bitmask to host
    // 8. Start from 1. with next device
    int data_offset = 0;
    for(int device_index = 0; device_index < device_count; device_index++) {
        // setup device resources
        cudaSetDevice(device_index);
        cudaStreamCreate(&streams[device_index]);
#if BENCHMARK_TIME        
        int event_offset = device_index * events_per_gpu;
        cudaEventCreate(&events[event_offset]);
        cudaEventCreate(&events[event_offset + 1]);
        cudaEventCreate(&events[event_offset + 2]);
        cudaEventCreate(&events[event_offset + 3]);
#endif

        // filter kernel settings
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_index);
        dim3 threadsPerBlock(std::min(device_prop.maxThreadsPerBlock, benchmark_config.max_threads_per_gpu));
        dim3 numBlocks(std::min(input_size / threadsPerBlock.x, benchmark_config.max_blocks_per_gpu));

        total_mp_count += device_prop.multiProcessorCount;
        threads_per_gpu = threadsPerBlock.x;
        blocks_per_gpu = numBlocks.x;

        // define data split
        int data_size = 0;
        if(device_index == (device_count -1)) {
            data_size = input_size - data_offset;
        } else {
            data_size = input_size / device_count;
        }

        cudaMalloc(&device_d_inputs[device_index], data_size * sizeof(filter_int));
        cudaMalloc(&device_d_filter_results[device_index], data_size * sizeof(filter_mask)); 
#if DEBUG_PRINT
        std::cout << "GPU=" << device_index << " B=" << numBlocks.x << " T=" << threadsPerBlock.x << " Filter [X=" << reference << "] N=" << data_size << std::endl;
#endif
        // start filter (copy to devie -> run filter -> copy to host)
#if BENCHMARK_TIME
        cudaEventRecord(events[event_offset], streams[device_index]);
#endif
        cudaMemcpyAsync(device_d_inputs[device_index], &input_values[data_offset], data_size * sizeof(filter_int), cudaMemcpyHostToDevice, streams[device_index]);
#if BENCHMARK_TIME
        cudaEventRecord(events[event_offset + 2], streams[device_index]);
#endif
        filter_kernel<<<numBlocks, threadsPerBlock, 0, streams[device_index]>>>(data_size, device_d_inputs[device_index], reference, device_d_filter_results[device_index], predicate);
#if BENCHMARK_TIME
        cudaEventRecord(events[event_offset + 3], streams[device_index]);
#endif
        cudaMemcpyAsync(&output_mask[data_offset], device_d_filter_results[device_index], data_size * sizeof(filter_mask), cudaMemcpyDeviceToHost, streams[device_index]);
#if BENCHMARK_TIME
        cudaEventRecord(events[event_offset + 1], streams[device_index]);
#endif

        device_input_size[device_index] = data_size;
        data_offset += data_size;
    }
    
    // synchronize execution
#if BENCHMARK_TIME
    for(int device_index = 0; device_index < device_count; device_index++) {
        cudaEventSynchronize(events[device_index * events_per_gpu + 1]);
    }
#else
    cudaDeviceSynchronize();
#endif

    // output single gpu runtime
#if DEBUG_PRINT && BENCHMARK_TIME
    for(int device_index = 0; device_index < device_count; device_index++) {
        float filter_runtime_ms = 0;
        cudaEventElapsedTime(&filter_runtime_ms, events[device_index * 2], events[device_index * 2 + 1]);
        std::cout << "GPU=" << device_index << " Filter [Runtime] " << filter_runtime_ms << "ms " << ((float)device_input_size[device_index] / (filter_runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;
    }
#endif

    // calculate throughput + maximum runtime
    float runtime_ms = 0.0f;
    float runtime_no_mem_ms = 0.0f;
#if BENCHMARK_TIME
    for(int d1_index = 0; d1_index < device_count; d1_index++) {
        for(int d2_index = 0; d2_index < device_count; d2_index++) {
            // runtime full (with memory ops)
            float runtime_ms_new = 0.0f;
            cudaEventElapsedTime(&runtime_ms_new, events[d1_index * events_per_gpu], events[d2_index * events_per_gpu + 1]);
            runtime_ms = std::max(runtime_ms, runtime_ms_new);            

            // runtime filter only
            float runtime_no_mem_ms_new = 0.0f;
            cudaEventElapsedTime(&runtime_no_mem_ms_new, events[d1_index * events_per_gpu + 2], events[d2_index * events_per_gpu + 3]);
            runtime_no_mem_ms = std::max(runtime_no_mem_ms, runtime_no_mem_ms_new);
        }
    }
#endif
    float throughput_gb = (input_size * sizeof(filter_int) / std::pow(10, 9)) / (runtime_ms / std::pow(10, 3));
    float throughput_no_mem_gb = (input_size * sizeof(filter_int) / std::pow(10, 9)) / (runtime_no_mem_ms / std::pow(10, 3));

#if DEBUG_PRINT
    std::cout << "Process [Runtime] " << runtime_ms << "ms " << ((float)input_size / (runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;
#endif

#if ERROR_CHECK
    int error_counter = 0;
    for(int i = 0; i < input_size; i++) {
        error_counter += predicate(input_values[i], reference) == output_mask[i] ? 0 : 1;
    }
    std::cout << error_counter << " Error" << (error_counter != 1 ? "s" : "") << std::endl;
#endif

    float elements_per_thread = (float)input_size / (float)(benchmark_config.max_gpus * blocks_per_gpu * threads_per_gpu); 
    write_benchmark(benchmark_config.output_file, FILTER_VERSION, device_count, total_mp_count, threads_per_gpu, blocks_per_gpu, input_size, sizeof(filter_int), runtime_no_mem_ms, throughput_no_mem_gb, elements_per_thread);

    // cleanup
    for(int device_index = 0; device_index < device_count; device_index++) {
        cudaFree(device_d_filter_results[device_index]);
        cudaFree(device_d_inputs[device_index]);
        cudaStreamDestroy(streams[device_index]);
#if BENCHMARK_TIME
        for(int event_index = 0; event_index < events_per_gpu; event_index++) {
            cudaEventDestroy(events[device_index * events_per_gpu + event_index]);    
        }
#endif
    }

    delete[] device_d_filter_results;
    delete[] device_d_inputs;
    delete[] device_input_size;
    
}

int main(int argc, char **argv) {
    srand(time(NULL));

    if(argc != 3) {
        std::cout << "Invalid arguments. Use <app> <config_path> <profile>" << std::endl;
        return -1;
    }

    BenchmarkSetup benchmark_setup;
    if(!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup)) {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }


    const long element_count = benchmark_setup.elements;
    if(element_count < 0) {
        std::cout << "Invalid element count " << element_count << std::endl;
        return -1;
    }
    filter_int *h_input;
    filter_mask *h_filter_result;
    filter_int reference = 20;

    // filter function
    // host is required when ERROR_CHECK == 1
    auto filter_func = [=] 
        __device__ 
#if ERROR_CHECK
        __host__ 
#endif
        (filter_int x, filter_int ref) { return x == ref; };

    // allocate memory
    h_input = new filter_int[element_count];
    h_filter_result = new filter_mask[element_count];

    // init input with numbers from 0 to 200
    for(int i = 0; i < element_count; i++) {
        h_input[i] = rand() % 200;
    }

    // write benchmark header
    {
        std::fstream output_file(benchmark_setup.output_file_path, std::ios::out);
        write_benchmark_header(output_file);
        output_file.close();
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    device_count = std::min(device_count, *std::max_element(std::begin(benchmark_setup.gpus), std::end(benchmark_setup.gpus)));

    int runs = benchmark_setup.runs;
    //int gpu_count[] = { 1, 2 };
    //int thread_count[] = { 32, 64, 128, 256, 512, 1024 };
    //uint block_count[] = { 32, 64, 128, 256, 512, 1024, UINT_MAX };
    /*
    for(int run_index = 0; run_index < runs; run_index++) {
        std::cout << "Run " << (run_index+1) << "/" << runs << std::endl;
        for(auto gpu_count : benchmark_setup.gpus) {
            if(gpu_count > device_count) {
                break;
            }
            for(auto threads : benchmark_setup.threads) {
                for(auto blocks : benchmark_setup.blocks) {
#if DEBUG_PRINT                    
                    std::cout << "Run " << gpu_count << "," << blocks << "," << threads << std::endl;
#endif                
                    BenchmarkRunConfig run_config;
                    run_config.max_gpus = gpu_count;
                    run_config.max_threads_per_gpu = threads;
                    run_config.max_blocks_per_gpu = blocks;
                    run_config.output_file.open(benchmark_setup.output_file_path, std::ios::out | std::ios::app);
                    filter(element_count, h_input, h_filter_result, reference, filter_func, run_config);
                }
            }
        }  
    }
    */

    delete[] h_input;
    delete[] h_filter_result;
}