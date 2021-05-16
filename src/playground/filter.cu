
#include <iostream>
#include <time.h>

#define ERROR_CHECK 1
#define DEBUG_PRINT 1

typedef uint64_t filter_int;
typedef bool filter_mask;

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
void filter(int input_size, filter_int *input_values, filter_mask* output_mask, filter_int reference, F predicate) {

    // recording for whole process
    cudaEvent_t e_process_start, e_process_end;
    cudaEventCreate(&e_process_start);
    cudaEventCreate(&e_process_end);
    cudaEventRecord(e_process_start);

    // setup device resources
    // get devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if(device_count == 0) {
        std::cout << "No device found" << std::endl;
        exit(-1);
    }
#if DEBUG_PRINT
    std::cout << device_count << " Device" << (device_count != 1 ? "s" : "") << std::endl;
#endif
    // declare streams, buffers, events
    cudaStream_t *streams = new cudaStream_t[device_count];
    cudaEvent_t *events = new cudaEvent_t[device_count * 2];
    filter_mask **device_d_filter_results = new filter_mask*[device_count];
    filter_int **device_d_inputs = new filter_int*[device_count];
    int *device_input_size = new int[device_count];

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
        cudaEventCreate(&events[device_index * 2]);
        cudaEventCreate(&events[device_index * 2 + 1]);

        // filter kernel settings
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_index);
        dim3 threadsPerBlock(device_prop.maxThreadsPerBlock);
        dim3 numBlocks(input_size / threadsPerBlock.x);

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
        cudaEventRecord(events[device_index * 2], streams[device_index]);
        cudaMemcpyAsync(device_d_inputs[device_index], &input_values[data_offset], data_size * sizeof(filter_int), cudaMemcpyHostToDevice, streams[device_index]);
        filter_kernel<<<numBlocks, threadsPerBlock, 0, streams[device_index]>>>(data_size, device_d_inputs[device_index], reference, device_d_filter_results[device_index], predicate);
        cudaMemcpyAsync(&output_mask[data_offset], device_d_filter_results[device_index], data_size * sizeof(filter_mask), cudaMemcpyDeviceToHost, streams[device_index]);
        cudaEventRecord(events[device_index * 2 + 1], streams[device_index]);

        device_input_size[device_index] = data_size;
        data_offset += data_size;
    }
    
    // synchronize execution
    for(int device_index = 0; device_index < device_count; device_index++) {
        cudaEventSynchronize(events[device_index * 2 + 1]);
    }
    cudaEventRecord(e_process_end);
    cudaEventSynchronize(e_process_end);

    // output single gpu runtime
#if DEBUG_PRINT
    for(int device_index = 0; device_index < device_count; device_index++) {
        float filter_runtime_ms = 0;
        cudaEventElapsedTime(&filter_runtime_ms, events[device_index * 2], events[device_index * 2 + 1]);
        std::cout << "GPU=" << device_index << " Filter [Runtime] " << filter_runtime_ms << "ms " << ((float)device_input_size[device_index] / (filter_runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;
    }

    // output total runtime
    float filter_runtime_ms = 0;
    cudaEventElapsedTime(&filter_runtime_ms, e_process_start, e_process_end);
    std::cout << "Process [Runtime] " << filter_runtime_ms << "ms " << ((float)input_size / (filter_runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;
#endif

#if ERROR_CHECK
    int error_counter = 0;
    for(int i = 0; i < input_size; i++) {
        error_counter += predicate(input_values[i], reference) == output_mask[i] ? 0 : 1;
    }
    std::cout << error_counter << " Error" << (error_counter != 1 ? "s" : "") << std::endl;
#endif

    // cleanup
    for(int device_index = 0; device_index < device_count; device_index++) {
        cudaFree(device_d_filter_results[device_index]);
        cudaFree(device_d_inputs[device_index]);
        cudaStreamDestroy(streams[device_index]);
        cudaEventDestroy(events[device_index * 2]);
        cudaEventDestroy(events[device_index * 2 + 1]);
    }

    cudaEventDestroy(e_process_start);
    cudaEventDestroy(e_process_end);

    delete[] device_d_filter_results;
    delete[] device_d_inputs;
    delete[] device_input_size;
    
}

int main(int argc, char **argv) {
    srand(time(NULL));

    const int element_count = 1<<20;
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

    filter(element_count, h_input, h_filter_result, reference, filter_func);
    
    delete[] h_input;
    delete[] h_filter_result;
}