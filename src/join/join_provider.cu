#pragma once

#include <chrono>
#include <iostream>
#include <math.h>
#include <string.h>
#include <cassert>
#include <sstream>

/*
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
*/

#include "base/types.hpp"

struct PartitionSummary {

    index_t elements = 0;
    float k_histrogram_elements_p_second = 0.0;
    float k_histrogram_gb_p_second = 0.0;
    float k_swap_elements_p_second = 0.0;
    float k_swap_gb_p_second = 0.0;
};

struct ProbeSummary {
    index_t r_elements = 0;
    index_t s_elements = 0;
    index_t rs_elements = 0;
    float k_build_probe_tuples_p_second = 0.0;
    float k_build_probe_gb_p_second = 0.0;
    float k_extract_tuples_p_second = 0.0;
    float k_extract_gb_p_second = 0.0;
};

struct PartitionConfig {
    int histogram_threads = 256;
    int histogram_n_elements_p_thread = 4;

    int swap_threads = 256;
    int swap_n_elements_p_thread = 1;

    int radix_width = 0;
    int bins = 0;
    index_t *d_histogram = nullptr; 
    index_t *d_offsets = nullptr;
    index_t *d_dest_indices = nullptr;

    bool profiling_enabled = false;
    PartitionSummary profiling_summary;
    cudaEvent_t profiling_start, profiling_end;

    cudaStream_t stream = 0;

    void enable_profiling(cudaEvent_t profiling_start, cudaEvent_t profiling_end) {
        profiling_enabled = true;
        this->profiling_start = profiling_start;
        this->profiling_end = profiling_end;
    }

    void disable_profiling() {
        profiling_enabled = false;
    }

    void set_radix_width(int radix_width) {
        this->radix_width = radix_width;
        int new_bins = 1 << radix_width;
        if(new_bins != bins) {
            free();
            bins = new_bins;
            gpuErrchk(cudaMallocAsync(&d_histogram, bins * sizeof(index_t), stream));
            gpuErrchk(cudaMallocAsync(&d_offsets, bins * sizeof(index_t), stream));
            gpuErrchk(cudaMallocAsync(&d_dest_indices, bins * sizeof(index_t), stream));
        }
    }

    void start_profiling() {
        if(profiling_enabled) {
            cudaEventRecord(profiling_start, stream);
        }
    }

    void stop_profiling() {
        if(profiling_enabled) {
            cudaEventRecord(profiling_end, stream);
        }
    }

    float get_elapsed_time_s() {
        if(profiling_enabled) {        
            float runtime_ms = 0.0;
            cudaEventSynchronize(profiling_end);
            cudaEventElapsedTime(&runtime_ms, profiling_start, profiling_end);
            return runtime_ms / pow(10, 3);
        }
        return 0.0f;
    }

    void free() {
        if(d_dest_indices) {
            gpuErrchk(cudaFreeAsync(d_dest_indices, stream));
        }
        if(d_histogram) {
            gpuErrchk(cudaFreeAsync(d_histogram, stream));
        }
        if(d_offsets) {
            gpuErrchk(cudaFreeAsync(d_offsets, stream));
        }

        d_dest_indices = nullptr;
        d_histogram = nullptr;
        d_offsets = nullptr;
        bins = 0;
    }
};

struct ProbeConfig
{
    float build_table_load = 0.75;
    int build_n_per_thread = 1;
    int build_threads;

    int extract_n_per_thread = 1;
    int extract_threads;

    int max_r_elements = 3200;

    index_s_t max_probe_buffer_size = 0;
    index_s_t probe_buffer_size = 0;
    index_s_t *d_probe_buffer = nullptr;
    index_s_t *d_probe_result_size = nullptr;

#if PROBE_MODE == 2
    int max_table_slots = 0;
    int max_table_links = 0;
    index_s_t * d_table_slots = nullptr;
    index_s_t * d_table_links = nullptr;
#endif

    bool profiling_enabled = false;
    ProbeSummary profiling_summary;
    cudaEvent_t profiling_start, profiling_end;

    cudaStream_t stream = 0;

    void print()
    {
        printf("B(%f %d:%d) E(%d:%d)\n", build_table_load, build_n_per_thread, build_threads, extract_n_per_thread, extract_threads);
    }

    int get_table_size(index_s_t elements)
    {
        return get_table_size(elements, get_table_slots(elements));
    }

    int get_table_size(index_s_t elements, index_s_t slots)
    {
        return elements * (sizeof(hash_t) + sizeof(index_s_t)) + slots * sizeof(index_s_t);
    }

    index_s_t get_table_slots(index_s_t elements)
    {
        return build_table_load * elements;
    }

    void enable_profiling(cudaEvent_t profiling_start, cudaEvent_t profiling_end) {
        profiling_enabled = true;
        this->profiling_start = profiling_start;
        this->profiling_end = profiling_end;
    }

    void disable_profiling() {
        profiling_enabled = false;
    }

    void free()
    {
        gpuErrchk(cudaFree(d_probe_buffer));
        gpuErrchk(cudaFree(d_probe_result_size));
        max_probe_buffer_size = 0;
        d_probe_buffer = nullptr;
        d_probe_result_size = nullptr;       
        disable_profiling();

#if PROBE_MODE == 2
        gpuErrchk(cudaFree(d_table_slots));
        gpuErrchk(cudaFree(d_table_links));
        max_table_slots = 0;
        max_table_links = 0;
        
#endif
    }
};

hash_t get_radix_mask(int bins)
{
    return bins - 1;
}

__global__ void histogram_kernel(index_t buffer_size, hash_t *hash_buffer, int bins, index_t *histogram, int radix_shift, hash_t radix_mask)
{
    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;
    
    for (index_t element_index = index; element_index < buffer_size; element_index += stride)
    {
        hash_t key = (hash_buffer[element_index] >> radix_shift) & radix_mask;
        atomicAdd(&histogram[key], 1);
    }
}

__global__ void element_swap_kernel(db_table table, db_hash_table hash_table, db_table table_swap, db_hash_table hash_table_swap, index_t *offsets, index_t *dest_indices, int radix_shift, hash_t radix_mask)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < hash_table.size; element_index += stride)
    {
        // determine bucket by hash
        hash_t element = hash_table.hashes[element_index];
        hash_t key = (element >> radix_shift) & radix_mask;

        // place index in new bucket position
        index_t swap_index = offsets[key] + atomicAdd(&dest_indices[key], 1);
        hash_table_swap.hashes[swap_index] = hash_table.hashes[element_index];
        //hash_table_swap.column_values[swap_index] = hash_table.column_values[element_index];

        table_swap.primary_keys[swap_index] = table.primary_keys[element_index];

        index_t column_value_index = element_index * table.column_count;
        index_t column_value_swap_index = swap_index * table.column_count;

        //memcpy(&table_swap.column_values[column_value_swap_index], &table.column_values[column_value_index], table.column_count * sizeof(column_t));

        for (int column_index = 0; column_index < table.column_count; column_index++)
        {
            table_swap.column_values[column_value_swap_index + column_index] = table.column_values[column_value_index + column_index];
        }
    }
}

__device__ void build_hash_table(db_hash_table r_hash_table, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int slot_index = index; slot_index < slots; slot_index += stride)
    {
        //printf("%d\n", slot_index);
        table_slots[slot_index] = 0;
    }

    __syncthreads();

    for (int r_index = index; r_index < r_hash_table.size; r_index += stride)
    {
        hash_t r_hash = r_hash_table.hashes[r_index];
        hash_t r_key = r_hash >> key_offset;
        int slot = r_key % slots;
        table_links[r_index] = (index_s_t)atomicExch((uint *)&table_slots[slot], (uint)r_index + 1);
        table_hashes[r_index] = r_hash;
    }
}

__device__ void probe_kernel(db_table r_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_sizes_buffer, filter_mask *probe_results, int hash_offset, int hash_full_size, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    // probe values from table s on hash table with values from r
    for (int s_index = index; s_index < s_table.size; s_index += stride)
    {
        //memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        index_s_t probe_size = 0;
        while (table_index)
        {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = table_hashes[table_index];

            index_s_t probe_index = s_index * hash_full_size + hash_offset + table_index;

            if (s_hash == r_hash)
            {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++)
                {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                filter_mask probe_result = column_equal_counter == r_table.column_count;

                probe_results[probe_index] = probe_result;
                probe_size += (index_t)probe_result;
            }
            else
            {
                probe_results[probe_index] = 0;
            }

            table_index = table_links[table_index];
        }
        atomicAdd(&probe_sizes_buffer[s_index], probe_size);
    }
}

__device__ void partial_probe_kernel(db_table r_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // probe values from table s on hash table with values from r
    for (int s_index = index; s_index < s_table.size; s_index += stride)
    {
        //memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        while (table_index)
        {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = table_hashes[table_index];

            index_s_t probe_index = s_index * r_table.size + table_index;

            if (s_hash == r_hash)
            {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++)
                {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                if(column_equal_counter == r_table.column_count) {
                    probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                }
            }
            table_index = table_links[table_index];
        }
    }
}

__global__ void partial_build_and_probe_kernel(db_table r_table, db_hash_table r_hash_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_sizes_buffer, bool *probe_results, int key_offset, int slots, int blocks_per_memory, int block_elements)
{
    int hash_total_size = r_hash_table.size;

    // calculate hash / table offset for r lookup in block
    // global == default for all blocks
    int hash_global_size = block_elements;
    int hash_offset = hash_global_size * blockIdx.x;
    int hash_block_size = hash_global_size;
    if (blockIdx.x == gridDim.x - 1)
    {
        hash_block_size = r_table.size - hash_offset;
    }

    r_table.column_values = &r_table.column_values[r_table.column_count * hash_offset];
    r_hash_table.hashes = &r_hash_table.hashes[hash_offset];
    r_hash_table.size = hash_block_size;
    r_table.size = hash_block_size;

    // hash table
    extern __shared__ hash_t *table[];
    hash_t *table_hashes = (hash_t *)&table[0];
    index_s_t *table_links = (index_s_t *)&table_hashes[r_table.size];
    index_s_t *table_slots = (index_s_t *)&table_links[r_table.size];

    // build phase
    // build hash table from hashes in table r
    build_hash_table(r_hash_table, key_offset, slots, table_hashes, table_links, table_slots);
    __syncthreads();

    // probing
    // probe values from table s on hash table with values from r
    probe_kernel(r_table, s_table, s_hash_table, probe_sizes_buffer, probe_results, hash_offset, hash_total_size, key_offset, slots, table_hashes, table_links, table_slots);
}

__global__ void build_and_partial_probe_kernel(db_table r_table, db_hash_table r_hash_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int key_offset, int slots)
{
    // hash table
    extern __shared__ hash_t *table[];
    hash_t *table_hashes = (hash_t *)&table[0];
    index_s_t *table_links = (index_s_t *)&table_hashes[r_table.size];
    index_s_t *table_slots = (index_s_t *)&table_links[r_table.size];

    // build phase
    // build hash table from hashes in table r
    build_hash_table(r_hash_table, key_offset, slots, table_hashes, table_links, table_slots);
    __syncthreads();

    // probing
    // probe values from table s on hash table with values from  {}
    partial_probe_kernel(r_table, s_table, s_hash_table, probe_results_size, probe_results, key_offset, slots, table_hashes, table_links, table_slots);
}

__global__ void build_global_kernel(db_table r_table, db_hash_table r_hash_table, int key_offset, int slots, index_s_t *table_links, index_s_t *table_slots) {

    // build
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r_index = index; r_index < r_hash_table.size; r_index += stride)
    {
        hash_t r_hash = r_hash_table.hashes[r_index];
        hash_t r_key = r_hash >> key_offset;
        int slot = r_key % slots;
        table_links[r_index] = (index_s_t)atomicExch((uint *)&table_slots[slot], (uint)r_index + 1);
    }
}

__global__ void probe_global_kernel(db_table r_table, db_hash_table r_hash_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int key_offset, int slots, index_s_t *table_links, index_s_t *table_slots) {

    // build
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // probe values from table s on hash table with values from r
    for (int s_index = index; s_index < s_table.size; s_index += stride)
    {
        //memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        while (table_index)
        {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = r_hash_table.hashes[table_index];

            index_s_t probe_index = s_index * r_table.size + table_index;

            if (s_hash == r_hash)
            {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++)
                {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                if(column_equal_counter == r_table.column_count) {
                    probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                }
            }
            table_index = table_links[table_index];
        }
    }
}

__global__ void copy_probe_results_kernel(db_table r_table, db_table s_table, index_s_t *indices, db_table rs_table)
{
    //int rs_half_column_count = rs_table.column_count / 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_s_t buffer_index = index; buffer_index < rs_table.size; buffer_index += stride)
    {
        index_s_t copy_index = indices[buffer_index];
        index_s_t s_index = copy_index / r_table.size;
        index_s_t r_index = copy_index % r_table.size;

        index_s_t rs_offset = buffer_index * rs_table.column_count;
        rs_table.primary_keys[buffer_index] = buffer_index+1;

        /*
        for (int column_index = 0; column_index < rs_half_column_count; column_index++)
        {
            rs_table.column_values[rs_offset + column_index] = r_table.column_values[r_index * r_table.column_count + column_index];
            rs_table.column_values[rs_offset + column_index + rs_half_column_count] = s_table.column_values[s_index * s_table.column_count + column_index];
        }
        */
       
        rs_table.column_values[rs_offset] = r_table.primary_keys[r_index];
        rs_table.column_values[rs_offset + 1] = s_table.primary_keys[s_index];
       
        
    }
}

void partition_gpu(db_table d_table, db_hash_table d_hash_table, db_table d_table_swap, db_hash_table d_hash_table_swap, int radix_shift, index_t *histogram, index_t *offsets, PartitionConfig &partition_config)
{
    assert(d_hash_table.size == d_hash_table_swap.size);

    hash_t radix_mask = get_radix_mask(partition_config.bins);
    int bins = partition_config.bins;
    cudaStream_t stream = partition_config.stream;

    index_t *d_histogram = partition_config.d_histogram;

    gpuErrchk(cudaMemsetAsync(d_histogram, 0, bins * sizeof(index_t), stream));



    int histogram_threads = partition_config.histogram_threads;
    int histrogram_blocks = max(1ULL, d_hash_table.size / histogram_threads / partition_config.histogram_n_elements_p_thread);
    
    partition_config.start_profiling();
    histogram_kernel<<<histrogram_blocks, histogram_threads, 0, stream>>>(d_hash_table.size, d_hash_table.hashes, bins, d_histogram, radix_shift, radix_mask);
    partition_config.stop_profiling();
    if(partition_config.profiling_enabled) {
        float runtime_s = partition_config.get_elapsed_time_s();
        partition_config.profiling_summary.k_histrogram_elements_p_second = d_hash_table.size / runtime_s;
        partition_config.profiling_summary.k_histrogram_gb_p_second = d_hash_table.size * sizeof(hash_t) / runtime_s / pow(10, 9); 
    }

    gpuErrchk(cudaMemcpyAsync(histogram, d_histogram, bins * sizeof(index_t), cudaMemcpyDeviceToHost, stream));

    // calculate offsets
    index_t *d_dest_indices = partition_config.d_dest_indices;
    index_t *d_offsets = partition_config.d_offsets;
    memset(offsets, 0, bins * sizeof(int));
    
    cudaStreamSynchronize(partition_config.stream);
    gpuErrchk(cudaMemsetAsync(d_dest_indices, 0, bins * sizeof(index_t), partition_config.stream));
    index_t offset = 0;
    for (int bin_index = 0; bin_index < bins; bin_index++)
    {
        offsets[bin_index] = offset;
        offset += histogram[bin_index];
    }
    gpuErrchk(cudaMemcpyAsync(d_offsets, offsets, bins * sizeof(index_t), cudaMemcpyHostToDevice, partition_config.stream));


    // swap elments according to bin key
    int swap_threads = partition_config.swap_threads;
    int swap_blocks = max(1ULL, d_table.size / swap_threads / partition_config.swap_n_elements_p_thread);
    partition_config.start_profiling();
    element_swap_kernel<<<swap_blocks, swap_threads, 0, partition_config.stream>>>(d_table, d_hash_table, d_table_swap, d_hash_table_swap, d_offsets, d_dest_indices, radix_shift, radix_mask);
    partition_config.stop_profiling();
    if(partition_config.profiling_enabled) {
        float runtime_s = partition_config.get_elapsed_time_s();
        partition_config.profiling_summary.k_swap_elements_p_second = d_table.size / runtime_s;
        int swap_element_size = (sizeof(hash_t) + d_table.column_count * sizeof(column_t));
        partition_config.profiling_summary.k_swap_gb_p_second = d_table.size * swap_element_size / runtime_s / pow(10, 9);
        partition_config.profiling_summary.elements = d_hash_table.size;
    }    

    gpuErrchk(cudaStreamSynchronize(partition_config.stream));
}

struct is_greater_zero
{
    __host__ __device__ bool operator()(const index_s_t &x)
    {
        return x > 0;
    }
};

void build_and_probe_gpu(db_table d_r_table, db_hash_table d_r_hash_table, db_table d_s_table, db_hash_table d_s_hash_table, db_table &d_joined_rs_table, int key_offset, ProbeConfig &config)
{
    // MAX-Q / Quadro / Turing 7.5
    // 32 Shared Banks
    // 64KB shared mem   / SM
    // 64k 32Bit reg     / SM
    // 255 32Bit reg     / Thread

    // 16 Blocks         / SM (16 x 64 Threads)
    // 32 Warps          / SM
    // 1024 Threads      / SM

    //cudaStreamSynchronize(stream);
    //gpuErrchk(cudaGetLastError());

    /*
    int event_count = 5;
    cudaEvent_t events[2 * event_count];
    for(int event_index = 0;  event_index < event_count; event_index++) {
        cudaEventCreate(&events[event_index*2]);
        cudaEventCreate(&events[event_index*2+1]);
    }
    */
    // allocate probe buffer
    // keep old buffer or extend to bigger buffer
    index_s_t slots = config.get_table_slots(d_r_hash_table.size);
    config.probe_buffer_size = d_r_hash_table.size * d_s_hash_table.size;
    
    // adjust reserved buffers
    if (config.max_probe_buffer_size < config.probe_buffer_size)
    {
        if(config.d_probe_buffer) {
            gpuErrchk(cudaFreeAsync(config.d_probe_buffer, config.stream));
        }
        gpuErrchk(cudaMallocAsync(&config.d_probe_buffer, config.probe_buffer_size * sizeof(index_s_t), config.stream));
        config.max_probe_buffer_size = config.probe_buffer_size;
    }

    if(!config.d_probe_result_size) {
        gpuErrchk(cudaMallocAsync(&config.d_probe_result_size, sizeof(index_s_t), config.stream));
    }
    gpuErrchk(cudaMemsetAsync(config.d_probe_result_size, 0, sizeof(index_s_t), config.stream));

#if PROBE_MODE == 2
    if(slots > config.max_table_slots) {
        config.max_table_slots = slots;
        if(config.d_table_slots) {
            gpuErrchk(cudaFreeAsync(config.d_table_slots, config.stream));
        }
        gpuErrchk(cudaMallocAsync(&config.d_table_slots, slots * sizeof(index_s_t), config.stream));
    }

    if(d_r_hash_table.size > config.max_table_links) {
        config.max_table_links = d_r_hash_table.size;
        if(config.d_table_links) {
            gpuErrchk(cudaFreeAsync(config.d_table_links, config.stream));
        }
        gpuErrchk(cudaMallocAsync(&config.d_table_links, d_r_hash_table.size * sizeof(index_s_t), config.stream));
    }
#endif

    if(config.profiling_enabled) {
        cudaEventRecord(config.profiling_start, config.stream);
    }
#if PROBE_MODE == 0
    assert(1024 % config.build_threads.x == 0);
    int blocks_per_memory = 1024 / config.build_threads.x;
    int block_elements = config.build_threads.x * elements_per_thread;
    int blocks = ceil(d_r_table.size / (float)block_elements);
    slots = config.build_table_load * block_elements;

    int max_block_elements = std::max((index_t)block_elements, (d_r_table.size - (blocks - 1) * block_elements));

    int shared_mem = max_block_elements * (sizeof(hash_t) + sizeof(index_s_t)) + slots * sizeof(index_s_t);
    assert(shared_mem * blocks_per_memory <= 64000);
    partial_build_and_probe_kernel<<<blocks, config.build_threads, shared_mem, stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, d_probe_sizes_buffer, d_probe_results, key_offset, slots, blocks_per_memory, block_elements);
#elif PROBE_MODE == 1
    int blocks = ceil(d_s_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
    int shared_mem = config.get_table_size(d_r_hash_table.size, slots);
    //assert(shared_mem <= 49000);

    assert(blocks> 0);
    assert(config.build_threads > 0);
    build_and_partial_probe_kernel<<<blocks, config.build_threads, shared_mem, config.stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, config.d_probe_result_size, config.d_probe_buffer, key_offset, slots);
#elif PROBE_MODE == 2
    gpuErrchk(cudaMemsetAsync(config.d_table_slots, 0, sizeof(index_s_t) * config.max_table_slots, config.stream));

    int build_blocks = ceil(d_r_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
    int probe_blocks = ceil(d_s_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
    
    //assert(shared_mem <= 49000);

    
    assert(build_blocks > 0);
    assert(probe_blocks > 0);
    assert(config.build_threads > 0);
    build_global_kernel<<<build_blocks, config.build_threads, 0, config.stream>>>(d_r_table, d_r_hash_table, key_offset, slots, config.d_table_links, config.d_table_slots);   
    probe_global_kernel<<<probe_blocks, config.build_threads, 0, config.stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, config.d_probe_result_size, config.d_probe_buffer, key_offset, slots, config.d_table_links, config.d_table_slots);
#endif

    if(config.profiling_enabled) {
        cudaEventRecord(config.profiling_end, config.stream);
        cudaEventSynchronize(config.profiling_end);
        float runtime_ms = 0.0;
        cudaEventElapsedTime(&runtime_ms, config.profiling_start, config.profiling_end);
        float runtime_s = runtime_ms / pow(10, 3);
        config.profiling_summary.k_build_probe_tuples_p_second = (d_r_hash_table.size + d_s_hash_table.size) / runtime_s;
        int tuple_size = sizeof(column_t) * (d_r_table.column_count) + sizeof(hash_t);
        config.profiling_summary.k_build_probe_gb_p_second = (d_r_hash_table.size + d_s_hash_table.size) * tuple_size / runtime_s / pow(10, 9);
    }

    d_joined_rs_table.column_count = 2; // d_r_table.column_count + d_s_table.column_count;
    d_joined_rs_table.gpu = true;
    d_joined_rs_table.data_owner = true;
    gpuErrchk(cudaMemcpyAsync(&d_joined_rs_table.size, config.d_probe_result_size, sizeof(index_s_t), cudaMemcpyDeviceToHost, config.stream));
    gpuErrchk(cudaStreamSynchronize(config.stream));
    gpuErrchk(cudaGetLastError());

    if(d_joined_rs_table.size > 0) {
        gpuErrchk(cudaMallocAsync(&d_joined_rs_table.primary_keys, d_joined_rs_table.size * sizeof(column_t), config.stream));
        gpuErrchk(cudaMallocAsync(&d_joined_rs_table.column_values, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), config.stream));
        //gpuErrchk(cudaMemsetAsync(d_joined_rs_table.column_values, 1, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), config.stream));

        int extract_blocks = max(1ULL, d_joined_rs_table.size / (config.extract_n_per_thread * config.extract_threads));
        int extract_threads_per_block = config.extract_threads;
        assert(extract_blocks > 0);
        assert(extract_threads_per_block > 0);
        
        if(config.profiling_enabled) {
            cudaEventRecord(config.profiling_start, config.stream);
        }
        
        copy_probe_results_kernel<<<extract_blocks, extract_threads_per_block, 0, config.stream>>>(d_r_table, d_s_table, config.d_probe_buffer, d_joined_rs_table);
        
        if(config.profiling_enabled) {
            cudaEventRecord(config.profiling_end, config.stream);
            cudaEventSynchronize(config.profiling_end);
            float runtime_ms = 0.0;
            cudaEventElapsedTime(&runtime_ms, config.profiling_start, config.profiling_end);
            float runtime_s = runtime_ms / pow(10, 3);
            config.profiling_summary.k_extract_tuples_p_second = d_joined_rs_table.size / runtime_s;
            int tuple_size = sizeof(column_t) * d_joined_rs_table.column_count;
            config.profiling_summary.k_extract_gb_p_second = (d_joined_rs_table.size * tuple_size) / runtime_s / pow(10, 9);
        }
    }

    config.profiling_summary.r_elements = d_r_table.size;
    config.profiling_summary.s_elements = d_s_table.size;
    config.profiling_summary.rs_elements = d_joined_rs_table.size;
}