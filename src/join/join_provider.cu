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

struct ProbeConfig
{
    float build_table_load = 0.75;
    int build_n_per_thread = 1;
    int build_threads;

    int extract_n_per_thread = 1;
    int extract_threads;

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

    static std::string to_string_header()
    {
        std::ostringstream string_stream;
        string_stream
            << "probe_version,"
            << "probe_mode,"
            << "build_table_load,"
            << "build_n_per_thread,"
            << "build_threads,"
            << "extract_n_per_thread,"
            << "extract_threads";
        return string_stream.str();
    }

    std::string to_string()
    {
        std::ostringstream string_stream;
        string_stream
            << PROBE_VERSION << ","
            << PROBE_MODE << ","
            << build_table_load << ","
            << build_n_per_thread << ","
            << build_threads << ","
            << extract_n_per_thread << ","
            << extract_threads;
        return string_stream.str();
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

__global__ void element_swap_kernel(db_table table, db_hash_table hash_table, db_table table_swap, db_hash_table hash_table_swap, index_t *offsets, index_t *dest_indices, int radix_shift, hash_t radix_mask, bool index_data = false)
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
        if (index_data)
        {
            hash_table_swap.indices[swap_index] = element_index;
        }
        else
        {
            hash_table_swap.indices[swap_index] = hash_table.indices[element_index];
        }

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
                for (int column_index = 1; column_index < r_table.column_count; column_index++)
                {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                filter_mask probe_result = column_equal_counter == (r_table.column_count - 1);

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
        index_s_t probe_size = 0;
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
                for (int column_index = 1; column_index < r_table.column_count; column_index++)
                {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                filter_mask probe_result = column_equal_counter == (r_table.column_count - 1);
                probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                probe_size += (index_t)probe_result;
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
    // probe values from table s on hash table with values from r
    partial_probe_kernel(r_table, s_table, s_hash_table, probe_results_size, probe_results, key_offset, slots, table_hashes, table_links, table_slots);
}

__global__ void prefix_sum_kernel(index_s_t buffer_size, index_s_t *input_buffer, index_s_t *output_buffer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int stride = gridDim.x * blockDim.x;

    if (threadIdx.x == 0)
    {
        output_buffer[0] = 0;
    }

    for (index_s_t buffer_index = index; buffer_index < buffer_size; buffer_index += stride)
    {
        index_s_t lookup_index = buffer_index - 1;
        index_s_t offset = 0;
        while (true)
        {
            offset += input_buffer[lookup_index];
            if (lookup_index == 0)
            {
                break;
            }
            else
            {
                lookup_index--;
            }
        }
        output_buffer[buffer_index] = offset;
    }
}

__global__ void copy_probe_results_kernel(db_table r_table, db_table s_table, index_s_t *indices, db_table rs_table)
{
    int columns = rs_table.column_count / 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_s_t buffer_index = index; buffer_index < rs_table.size; buffer_index += stride)
    {
        index_s_t copy_index = indices[buffer_index];
        index_s_t s_index = copy_index / r_table.size;
        index_s_t r_index = copy_index % r_table.size;
        
        for (int column_index = 0; column_index < columns; column_index++)
        {
            rs_table.column_values[buffer_index * rs_table.column_count + column_index] = r_table.column_values[r_index * r_table.column_count + column_index];
            rs_table.column_values[buffer_index * rs_table.column_count + column_index + columns] = s_table.column_values[s_index * s_table.column_count + column_index];
        }
    }
}

__global__ void extract_probe_results_kernel(db_table r_table, db_table s_table, filter_mask *probe_buffer, index_s_t *probe_offsets_buffer, index_s_t *probe_sizes_buffer, db_table rs_table)
{
    typedef int32_t l_mask_t;
#if EXTRACT_MODE == 0
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = gridDim.x * blockDim.x;
    for (index_s_t s_buffer_index = x_index; s_buffer_index < s_table.size; s_buffer_index += x_stride)
    {
        //printf("s %d\n", s_buffer_index);
        index_s_t probe_size = probe_sizes_buffer[s_buffer_index];
        index_s_t probe_offset = s_buffer_index * r_table.size;
        l_mask_t *s_probe_buffer = (l_mask_t *)&probe_buffer[probe_offset];

        //const index_s_t s_column_value_index = s_table.offset + s_buffer_index;
        if (probe_size)
        {
            int elements_p_thread = sizeof(l_mask_t);
            int probe_element_index = blockIdx.y * blockDim.y + threadIdx.y;
            int probe_index = probe_element_index * elements_p_thread;
            int probe_element_stride = gridDim.y * blockDim.y;
            int probe_stride = probe_element_stride * elements_p_thread;
            for (; probe_index < r_table.size; probe_index += probe_stride)
            {
                l_mask_t l_mask = s_probe_buffer[probe_element_index];
                probe_element_index += probe_element_stride;
                if (l_mask > 0)
                {
                    for (int mask_index = 0; mask_index < elements_p_thread; mask_index++)
                    {
                        l_mask = l_mask >> mask_index * sizeof(filter_mask);
                        filter_mask mask = l_mask & 0x0001;
                        if (mask)
                        {
                            index_s_t pair_offset = atomicAdd(&probe_offsets_buffer[s_buffer_index], 1);

                            for (int column_index = 0; column_index < r_table.column_count; column_index++)
                            {
                                rs_table.column_values[pair_offset * rs_table.column_count + column_index] = r_table.column_values[(probe_index + mask_index) * r_table.column_count + column_index];
                                rs_table.column_values[pair_offset * rs_table.column_count + column_index + r_table.column_count] = s_table.column_values[s_buffer_index * s_table.column_count + column_index];
                            }
                        }
                    }
                    // column 0 = r primary keys
                    // column 1 = s primary keys
                    //printf("s %d p %d\n", s_buffer_index, probe_index);
                }
            }
        }
    }
#elif EXTRACT_MODE == 1

    extern __shared__ index_s_t *extract_buffer[];
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = gridDim.x * blockDim.x;
    for (index_s_t s_buffer_index = x_index; s_buffer_index < s_table.size; s_buffer_index += x_stride)
    {
        //printf("s %d\n", s_buffer_index);
        index_s_t probe_size = probe_sizes_buffer[s_buffer_index];
        index_s_t probe_offset = s_buffer_index * r_table.size;

        //const index_s_t s_column_value_index = s_table.offset + s_buffer_index;
        if (probe_size)
        {
            int probe_index = blockIdx.y * blockDim.y + threadIdx.y;
            int probe_stride = gridDim.y * blockDim.y;
            for (; probe_index < r_table.size; probe_index += probe_stride)
            {
                if (probe_buffer[probe_offset + probe_index])
                {

                    // column 0 = r primary keys
                    // column 1 = s primary keys
                    //printf("s %d p %d\n", s_buffer_index, probe_index);

                    index_s_t pair_offset = atomicAdd(&probe_offsets_buffer[s_buffer_index], 1);

                    for (int column_index = 0; column_index < r_table.column_count; column_index++)
                    {
                        rs_table.column_values[pair_offset * rs_table.column_count + column_index] = r_table.column_values[probe_index * r_table.column_count + column_index];
                        rs_table.column_values[pair_offset * rs_table.column_count + column_index + r_table.column_count] = s_table.column_values[s_buffer_index * s_table.column_count + column_index];
                    }
                }
            }
        }
    }
#endif
}

void partition_gpu(db_table d_table, db_hash_table d_hash_table, db_table d_table_swap, db_hash_table d_hash_table_swap, int radix_width, int radix_shift, int bins, index_t *histogram, index_t *offsets, bool index_data, cudaStream_t stream)
{
    assert(d_hash_table.size == d_hash_table_swap.size);

    hash_t radix_mask = get_radix_mask(bins);

    index_t *d_histogram = nullptr;

#if BENCHMARK_PART
    cudaEvent_t hist_start, hist_end, swap_start, swap_end;
    cudaEventCreate(&hist_start);
    cudaEventCreate(&hist_end);
    cudaEventCreate(&swap_start);
    cudaEventCreate(&swap_end);
#endif

    //gpuErrchk(cudaStreamSynchronize(stream));
    //size_t free_mem, total_mem;
    //cudaMemGetInfo(&free_mem, &total_mem);
    //std::cout << "Mem Free=" << free_mem / std::pow(10, 9) << "GiB Mem Total=" << total_mem / std::pow(10, 9) << "GiB " << std::endl;
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMallocAsync(&d_histogram, bins * sizeof(index_t), stream));
    gpuErrchk(cudaMemsetAsync(d_histogram, 0, bins * sizeof(index_t), stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaGetLastError());
#if BENCHMARK_PART
    cudaEventRecord(hist_start, stream);
#endif
    histogram_kernel<<<1, 1, 0, stream>>>(d_hash_table.size, d_hash_table.hashes, bins, d_histogram, radix_shift, radix_mask);
#if BENCHMARK_PART
    cudaEventRecord(hist_end, stream);
#endif
    gpuErrchk(cudaMemcpyAsync(histogram, d_histogram, bins * sizeof(index_t), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaGetLastError());

    // calculate offsets
    index_t *d_dest_indices = nullptr;
    index_t *d_offsets = nullptr;
    memset(offsets, 0, bins * sizeof(int));
    gpuErrchk(cudaMallocAsync(&d_dest_indices, bins * sizeof(index_t), stream));
    gpuErrchk(cudaMemsetAsync(d_dest_indices, 0, bins * sizeof(index_t), stream));
    gpuErrchk(cudaMallocAsync(&d_offsets, bins * sizeof(index_t), stream));
    index_t offset = 0;
    cudaStreamSynchronize(stream);
    for (int bin_index = 0; bin_index < bins; bin_index++)
    {
        offsets[bin_index] = offset;
        offset += histogram[bin_index];
    }
    gpuErrchk(cudaMemcpyAsync(d_offsets, offsets, bins * sizeof(index_t), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    // swap elments according to bin key
#if BENCHMARK_PART
    cudaEventRecord(swap_start, stream);
#endif
    int threads = 128;
    int blocks = ceil(d_table.size / (float)threads);
    element_swap_kernel<<<blocks, threads, 0, stream>>>(d_table, d_hash_table, d_table_swap, d_hash_table_swap, d_offsets, d_dest_indices, radix_shift, radix_mask, index_data);
#if BENCHMARK_PART
    cudaEventRecord(swap_end, stream);
#endif
    std::cout << d_histogram << std::endl;
    gpuErrchk(cudaFreeAsync(d_histogram, stream));
    gpuErrchk(cudaFreeAsync(d_offsets, stream));
    gpuErrchk(cudaFreeAsync(d_dest_indices, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

#if BENCHMARK_PART
    float hist_runtime = 0.0f;
    float swap_runtime = 0.0f;
    cudaEventElapsedTime(&hist_runtime, hist_start, hist_end);
    cudaEventElapsedTime(&swap_runtime, swap_start, swap_end);
    std::cout << "S C=" << d_table.size << " Hist=" << hist_runtime << " Swap=" << swap_runtime << std::endl;

    cudaEventDestroy(hist_start);
    cudaEventDestroy(hist_end);
    cudaEventDestroy(swap_start);
    cudaEventDestroy(swap_end);
#endif
}

struct is_greater_zero
{
    __host__ __device__ bool operator()(const index_s_t &x)
    {
        return x > 0;
    }
};

void build_and_probe_gpu(db_table d_r_table, db_hash_table d_r_hash_table, db_table d_s_table, db_hash_table d_s_hash_table, db_table &d_joined_rs_table, int key_offset, cudaStream_t stream, ProbeConfig config)
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

    index_s_t *d_probe_results_size = nullptr;
    index_s_t *d_probe_results = nullptr;
    gpuErrchk(cudaMallocAsync(&d_probe_results, d_r_hash_table.size * d_s_hash_table.size * sizeof(index_s_t), stream));
    gpuErrchk(cudaMallocAsync(&d_probe_results_size, sizeof(index_s_t), stream));
    gpuErrchk(cudaMemsetAsync(d_probe_results_size, 0, sizeof(index_s_t), stream));
#if PROBE_MODE == 0

    assert(1024 % config.build_threads.x == 0);
    int blocks_per_memory = 1024 / config.build_threads.x;
    int block_elements = config.build_threads.x * elements_per_thread;
    int blocks = ceil(d_r_table.size / (float)block_elements);
    int slots = config.build_table_load * block_elements;

    int max_block_elements = std::max((index_t)block_elements, (d_r_table.size - (blocks - 1) * block_elements));

    int shared_mem = max_block_elements * (sizeof(hash_t) + sizeof(index_s_t)) + slots * sizeof(index_s_t);
    assert(shared_mem * blocks_per_memory <= 64000);
    partial_build_and_probe_kernel<<<blocks, config.build_threads, shared_mem, stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, d_probe_sizes_buffer, d_probe_results, key_offset, slots, blocks_per_memory, block_elements);
#elif PROBE_MODE == 1
    index_s_t slots = config.get_table_slots(d_r_hash_table.size);
    int blocks = ceil(d_s_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
    int shared_mem = config.get_table_size(d_r_hash_table.size, slots);
    //assert(shared_mem <= 49000);
    build_and_partial_probe_kernel<<<blocks, config.build_threads, shared_mem, stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, d_probe_results_size, d_probe_results, key_offset, slots);
#endif
    /*
    index_s_t *d_probe_offsets_buffer = nullptr;
    gpuErrchk(cudaMallocAsync(&d_probe_offsets_buffer, d_s_hash_table.size * sizeof(index_s_t), stream));

    thrust::device_ptr<index_s_t> td_probe_sizes(d_probe_sizes_buffer);
    thrust::device_ptr<index_s_t> td_probe_offsets(d_probe_offsets_buffer);  
    thrust::exclusive_scan(thrust::cuda::par.on(stream), td_probe_sizes, td_probe_sizes + d_s_table.size, td_probe_offsets);

    index_s_t result_last_offset = 0;
    index_s_t result_last_size = 0;
    gpuErrchk(cudaMemcpyAsync(&result_last_offset, &d_probe_offsets_buffer[d_s_hash_table.size - 1], sizeof(index_s_t), cudaMemcpyDeviceToHost, stream));
    cudaMemcpyAsync(&result_last_size, &d_probe_sizes_buffer[d_s_hash_table.size - 1], sizeof(index_s_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    d_joined_rs_table.size = result_last_offset + result_last_size;
    //std::cout << d_joined_rs_table.size << std::endl;
    // 2 = r + s primary key columns
    d_joined_rs_table.column_count = 2; // * d_r_table.column_count;
    //std::cout << d_joined_rs_table.column_values << std::endl;
    //std::cout << d_joined_rs_table.size << std::endl;
    */

    /*
    thrust::device_ptr<index_s_t> td_probe_results(d_probe_results);
    d_joined_rs_table.column_count = 2; // * d_r_table.column_count;
    d_joined_rs_table.size = thrust::count_if(thrust::cuda::par.on(stream), td_probe_results, td_probe_results + (d_s_hash_table.size * d_r_table.size), is_greater_zero());    
    
    index_s_t * d_probe_result_reduced = nullptr;
    cudaMallocAsync(&d_probe_result_reduced, d_joined_rs_table.size * sizeof(index_s_t), stream);
    thrust::device_ptr<index_s_t> probe_result_reduced(d_probe_result_reduced);
    thrust::copy_if(thrust::cuda::par.on(stream), td_probe_results, td_probe_results + (d_s_hash_table.size * d_r_hash_table.size), td_probe_results, probe_result_reduced, is_greater_zero());
    */

    d_joined_rs_table.column_count = 2;
    gpuErrchk(cudaMemcpyAsync(&d_joined_rs_table.size, d_probe_results_size, sizeof(index_s_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMallocAsync(&d_joined_rs_table.column_values, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), stream));
    gpuErrchk(cudaMemsetAsync(d_joined_rs_table.column_values, 1, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), stream));
    d_joined_rs_table.gpu = true;
    d_joined_rs_table.data_owner = true;

    int extract_blocks = ceil(d_joined_rs_table.size / (float)(config.extract_n_per_thread * (config.extract_threads)));
    int extract_threads_per_block = config.extract_threads;
    copy_probe_results_kernel<<<extract_blocks, extract_threads_per_block, 0, stream>>>(d_r_table, d_s_table, d_probe_results, d_joined_rs_table);
    //extract_probe_results_kernel<<<extract_blocks, config.extract_threads, 0, stream>>>(d_r_table, d_s_table, d_probe_results, d_probe_offsets_buffer, d_probe_sizes_buffer, d_joined_rs_table);

    //cudaFreeAsync(d_probe_offsets_buffer, stream);
    //cudaFreeAsync(d_probe_result_reduced, stream);
    cudaFreeAsync(d_probe_results_size, stream);
    cudaFreeAsync(d_probe_results, stream);

    /*
    cudaEventSynchronize(events[9]);
    for(int e_index = 0; e_index < event_count; e_index++) {
        float ms;
        cudaEventElapsedTime(&ms, events[e_index*2], events[e_index*2+1]);
        cudaEventDestroy(events[e_index*2]);
        cudaEventDestroy(events[e_index*2+1]);
        printf("%d-%d %F\n", e_index*2, e_index*2+1, ms);
    }
    */
}   