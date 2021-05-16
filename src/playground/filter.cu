
#include <iostream>
#include <time.h>

template <class F>
__global__
void filter(int size, int *input, int reference, bool *filter_result, F f) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = index; i < size; i += stride) {
        filter_result[i] = f(input[i], reference);
    }
}



int main(int argc, char **argv) {
    srand(time(NULL));

    const int element_count = 1<<20;
    int *h_input;
    bool *h_filter_result;
    int reference = 20;

    // filter function
    auto filter_func = [=] __device__ __host__ (int x, int ref) { return x == ref; };

    // allocate memory
    h_input = new int[element_count];
    h_filter_result = new bool[element_count];

    // init input with numbers from 0 to 200
    for(int i = 0; i < element_count; i++) {
        h_input[i] = rand() % 200;
    }

    // recording for whole process
    cudaEvent_t e_process_start, e_process_end;
    cudaEventCreate(&e_process_start);
    cudaEventCreate(&e_process_end);
    cudaEventRecord(e_process_start);

    // setup device resources
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << device_count << " Device" << (device_count > 1 ? "s" : "") << std::endl;
    cudaStream_t *streams = new cudaStream_t[device_count];
    cudaEvent_t *events = new cudaEvent_t[device_count * 2];
    bool **device_d_filter_results = new bool*[device_count];
    int **device_d_inputs = new int*[device_count];
    int *device_input_size = new int[device_count];

    

    // start kernel on each device
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
        dim3 numBlocks(element_count / threadsPerBlock.x);

        // define data split
        int data_size = 0;
        if(device_index == (device_count -1)) {
            data_size = element_count - data_offset;
        } else {
            data_size = element_count / device_count;
        }

        cudaMalloc(&device_d_inputs[device_index], data_size * sizeof(int));
        cudaMalloc(&device_d_filter_results[device_index], data_size * sizeof(bool)); 

        std::cout << "GPU=" << device_index << " B=" << numBlocks.x << " T=" << threadsPerBlock.x << " Filter [X=" << reference << "] N=" << data_size << std::endl;
        
        // start filter (copy to devie -> run filter -> copy to host)
        cudaEventRecord(events[device_index * 2], streams[device_index]);
        cudaMemcpyAsync(device_d_inputs[device_index], &h_input[data_offset], data_size * sizeof(int), cudaMemcpyHostToDevice, streams[device_index]);
        filter<<<numBlocks, threadsPerBlock, 0, streams[device_index]>>>(data_size, device_d_inputs[device_index], reference, device_d_filter_results[device_index], filter_func);
        cudaMemcpyAsync(&h_filter_result[data_offset], device_d_filter_results[device_index], data_size * sizeof(bool), cudaMemcpyDeviceToHost, streams[device_index]);
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

    // output gpu runtime
    for(int device_index = 0; device_index < device_count; device_index++) {
        float filter_runtime_ms = 0;
        cudaEventElapsedTime(&filter_runtime_ms, events[device_index * 2], events[device_index * 2 + 1]);
        std::cout << "GPU=" << device_index << " Filter [Runtime] " << filter_runtime_ms << "ms " << ((float)device_input_size[device_index] / (filter_runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;
    }

    // output total runtime
    float filter_runtime_ms = 0;
    cudaEventElapsedTime(&filter_runtime_ms, e_process_start, e_process_end);
    std::cout << "Process [Runtime] " << filter_runtime_ms << "ms " << ((float)element_count / (filter_runtime_ms * std::pow(10, 6))) << " GOP/S" << std::endl;


    int error_counter = 0;
    for(int i = 0; i < element_count; i++) {
        error_counter += filter_func(h_input[i], reference) == h_filter_result[i] ? 0 : 1;
    }
    std::cout << error_counter << " Error" << (error_counter != 1 ? "s" : "") << std::endl;


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
    delete[] h_input;
    delete[] h_filter_result;
}