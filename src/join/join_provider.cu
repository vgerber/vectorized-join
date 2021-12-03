#pragma once

#include "base/types.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string.h>

struct JoinStatus {
    std::string message = "Unknown";
    bool successful = false;

    JoinStatus(bool successful, std::string message) {
        this->successful = successful;
        this->message = message;
    }

    JoinStatus(bool successful) {
        this->successful = successful;
        this->message = successful ? "Successful" : "Failed";
    }

    JoinStatus() : JoinStatus(false, "Unknown") {
    }

    bool is_successful() {
        return successful;
    }

    bool has_failed() {
        return !successful;
    }
};

struct PartitionSummary {
    int device_index = 0;
    int stream_index = 0;
    int max_streams = 0;
    int buckets = 0;
    int depth = 0;

    size_t vector_bytes = 0;
    index_t elements = 0;

    std::string hash_function = "";

    float k_histogram_second = 0.0;
    float k_prefix_second = 0.0;
    float k_swap_second = 0.0;
    float k_histogram_elements_p_second = 0.0;
    float k_histogram_gb_p_second = 0.0;
    float k_swap_elements_p_second = 0.0;
    float k_swap_gb_p_second = 0.0;

    static std::string to_string_header() {
        std::ostringstream string_stream;
        string_stream << "device_index,"
                      << "max_streams,"
                      << "hash,"
                      << "stream,"
                      << "depth,"
                      << "buckets,"
                      << "elements,"
                      << "vector_bytes,"
                      << "kernel,"
                      << "runtime,"
                      << "gb_p_s,"
                      << "tuples_p_s";
        return string_stream.str();
    }

    std::string to_string() {
        std::ostringstream string_stream;
        string_stream << device_index << "," << max_streams << "," << hash_function << "," << stream_index << "," << depth << "," << buckets << "," << elements << "," << vector_bytes << ",histogram," << k_histogram_second << "," << k_histogram_gb_p_second << "," << k_histogram_elements_p_second
                      << std::endl;
        string_stream << device_index << "," << max_streams << "," << hash_function << "," << stream_index << "," << depth << "," << buckets << "," << elements << "," << vector_bytes << ",prefix," << k_prefix_second << ",0,0" << std::endl;
        string_stream << device_index << "," << max_streams << "," << hash_function << "," << stream_index << "," << depth << "," << buckets << "," << elements << "," << vector_bytes << ",swap," << k_swap_second << "," << k_swap_gb_p_second << "," << k_swap_elements_p_second;
        return string_stream.str();
    }
};

struct ProbeSummary {
    int probe_mode = 0;
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
        if (new_bins != bins) {
            free();
            bins = new_bins;
            gpuErrchk(cudaMallocAsync(&d_histogram, bins * sizeof(index_t), stream));
            gpuErrchk(cudaMallocAsync(&d_offsets, bins * sizeof(index_t), stream));
            gpuErrchk(cudaMallocAsync(&d_dest_indices, bins * sizeof(index_t), stream));
        }
    }

    void start_profiling() {
        if (profiling_enabled) {
            cudaEventRecord(profiling_start, stream);
        }
    }

    void stop_profiling() {
        if (profiling_enabled) {
            cudaEventRecord(profiling_end, stream);
        }
    }

    float get_elapsed_time_s() {
        if (profiling_enabled) {
            float runtime_ms = 0.0;
            cudaEventSynchronize(profiling_end);
            cudaEventElapsedTime(&runtime_ms, profiling_start, profiling_end);
            runtime_ms = max(0.0f, runtime_ms);
            return runtime_ms / pow(10, 3);
        }
        return 0.0f;
    }

    void free() {
        if (d_dest_indices) {
            gpuErrchk(cudaFreeAsync(d_dest_indices, stream));
        }
        if (d_histogram) {
            gpuErrchk(cudaFreeAsync(d_histogram, stream));
        }
        if (d_offsets) {
            gpuErrchk(cudaFreeAsync(d_offsets, stream));
        }

        d_dest_indices = nullptr;
        d_histogram = nullptr;
        d_offsets = nullptr;
        bins = 0;
    }
};

struct ProbeConfig {
    static const int MODE_PARTITION_R = 0;
    static const int MODE_PARTITION_S = 1;
    static const int MODE_GLOBAL_R = 2;

    int probe_mode = 0;

    float build_table_load = 0.75;
    int build_n_per_thread = 1;
    int build_threads;

    int extract_n_per_thread = 1;
    int extract_threads;

    int max_r_bytes = 48000;

    index_s_t max_probe_buffer_size = 0;
    index_s_t probe_buffer_size = 0;
    index_s_t *d_probe_buffer = nullptr;
    index_s_t *d_probe_result_size = nullptr;

    // probe mode 2 only
    int max_table_slots = 0;
    int max_table_links = 0;
    index_s_t *d_table_slots = nullptr;
    index_s_t *d_table_links = nullptr;

    bool profiling_enabled = false;
    ProbeSummary profiling_summary;
    cudaEvent_t profiling_start, profiling_end;

    cudaStream_t stream = 0;

    void print() {
        printf("B(%f %d:%d) E(%d:%d)\n", build_table_load, build_n_per_thread, build_threads, extract_n_per_thread, extract_threads);
    }

    int get_table_size(index_s_t elements) {
        return get_table_size(elements, get_table_slots(elements));
    }

    int get_table_size(index_s_t elements, index_s_t slots) {
        return elements * (sizeof(hash_t) + sizeof(index_s_t)) + slots * sizeof(index_s_t);
    }

    int get_max_table_elements() {
        return get_max_table_elements(max_r_bytes);
    }

    int get_max_table_elements(size_t bytes) {
        return floor(bytes / (sizeof(hash_t) + sizeof(index_s_t) + sizeof(index_s_t) * build_table_load));
    }

    int get_max_vector_elements(size_t bytes, int columns) {
        return floor(bytes / (sizeof(index_s_t) * (1 + build_table_load) + sizeof(column_t) * (1 + columns) + sizeof(hash_t)));
    }

    static int get_max_s_vector_elements(size_t bytes, db_table table) {
        return floor(bytes / (sizeof(hash_t) + (1 + sizeof(column_t)) * table.column_count));
    }

    index_s_t get_table_slots(index_s_t elements) {
        return max(1.f, build_table_load * elements);
    }

    void enable_profiling(cudaEvent_t profiling_start, cudaEvent_t profiling_end) {
        profiling_enabled = true;
        this->profiling_start = profiling_start;
        this->profiling_end = profiling_end;
    }

    void disable_profiling() {
        profiling_enabled = false;
    }

    void free() {
        if (d_probe_buffer) {
            gpuErrchk(cudaFree(d_probe_buffer));
            d_probe_buffer = nullptr;
        }
        if (d_probe_result_size) {
            gpuErrchk(cudaFree(d_probe_result_size));
            d_probe_result_size = nullptr;
        }
        max_probe_buffer_size = 0;

        disable_profiling();

        if (probe_mode == MODE_GLOBAL_R) {
            if (d_table_slots) {
                gpuErrchk(cudaFree(d_table_slots));
                d_table_slots = nullptr;
            }
            if (d_table_links) {
                gpuErrchk(cudaFree(d_table_links));
                d_table_links = nullptr;
            }
            max_table_slots = 0;
            max_table_links = 0;
        }
    }

    int get_allocated_memory() {
        int total_size = 0;
        if (probe_mode == ProbeConfig::MODE_GLOBAL_R) {
            total_size += (max_table_links + max_table_slots) * sizeof(index_s_t);
        }
        total_size += (probe_buffer_size + 1) * sizeof(index_s_t);
        return total_size;
    }
};

std::pair<size_t, size_t> get_memory_left() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return std::make_pair(free_mem, total_mem);
}

bool out_of_memory(size_t required_memory) {
    size_t memory_left = get_memory_left().first;
    if (memory_left < required_memory) {
        return false;
    }
    return (memory_left - required_memory) < MEMORY_TOLERANCE;
}

hash_t get_radix_mask(int bins) {
    return bins - 1;
}

__global__ void generate_primary_key_kernel(db_table table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int element_index = index; element_index < table.size; element_index += stride) {
        table.primary_keys[element_index] = element_index + 1;
    }
}

__global__ void histogram_kernel(index_t buffer_size, hash_t *hash_buffer, int bins, index_t *histogram, int radix_shift, hash_t radix_mask) {
    extern __shared__ index_t block_histogram[];
    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;

    for (index_t bin_index = threadIdx.x; bin_index < bins; bin_index++) {
        block_histogram[bin_index] = 0;
    }

    // calculate histogram for each block
    for (index_t element_index = index; element_index < buffer_size; element_index += stride) {
        hash_t key = (hash_buffer[element_index] >> radix_shift) & radix_mask;
        atomicAdd(&block_histogram[key], 1);
    }

    // sum block histograms
    __syncthreads();
    for (int bin_index = threadIdx.x; bin_index < bins; bin_index += blockDim.x) {
        atomicAdd(&histogram[bin_index], block_histogram[bin_index]);
    }
}

__global__ void prefix_sum_kernel(int bins, index_t *histogram, index_t *offsets) {
    extern __shared__ index_t shared_offsets[];
    index_t index = threadIdx.x;
    index_t stride = blockDim.x;

    // copy histogram into shared memory
    for (index_t element_index = index; element_index < bins - 1; element_index += stride) {
        shared_offsets[element_index] = histogram[element_index];
    }
    __syncthreads();

    index_t offset = 0;
    for (int offset_index = threadIdx.x - 1; offset_index >= 0; offset_index--) {
        offset += shared_offsets[offset_index];
    }

    offsets[threadIdx.x] = offset;
}

__global__ void element_swap_kernel(db_table table, db_hash_table hash_table, db_table table_swap, db_hash_table hash_table_swap, index_t *offsets, index_t *dest_indices, int radix_shift, hash_t radix_mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < hash_table.size; element_index += stride) {
        // determine bucket by hash
        hash_t element = hash_table.hashes[element_index];
        hash_t key = (element >> radix_shift) & radix_mask;

        // place index in new bucket position
        index_t swap_index = offsets[key] + atomicAdd(&dest_indices[key], 1);
        hash_table_swap.hashes[swap_index] = hash_table.hashes[element_index];
        // hash_table_swap.column_values[swap_index] = hash_table.column_values[element_index];

        table_swap.primary_keys[swap_index] = table.primary_keys[element_index];

        index_t column_value_index = element_index * table.column_count;
        index_t column_value_swap_index = swap_index * table.column_count;

        // memcpy(&table_swap.column_values[column_value_swap_index], &table.column_values[column_value_index], table.column_count * sizeof(column_t));

        for (int column_index = 0; column_index < table.column_count; column_index++) {
            table_swap.column_values[column_value_swap_index + column_index] = table.column_values[column_value_index + column_index];
        }
    }
}

__device__ void build_hash_table(db_hash_table r_hash_table, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int slot_index = index; slot_index < slots; slot_index += stride) {
        // printf("%d\n", slot_index);
        table_slots[slot_index] = 0;
    }

    __syncthreads();

    for (int r_index = index; r_index < r_hash_table.size; r_index += stride) {
        hash_t r_hash = r_hash_table.hashes[r_index];
        hash_t r_key = r_hash >> key_offset;
        int slot = r_key % slots;
        table_links[r_index] = (index_s_t)atomicExch((uint *)&table_slots[slot], (uint)r_index + 1);
        table_hashes[r_index] = r_hash;
    }
}

__device__ void probe_kernel(db_table r_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int table_offset, int table_full_size, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    // probe values from table s on hash table with values from r
    for (int s_index = index; s_index < s_table.size; s_index += stride) {
        // memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        while (table_index) {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = table_hashes[table_index];

            int global_r_index = table_index + table_offset;
            index_s_t probe_index = s_index * table_full_size + global_r_index;

            if (s_hash == r_hash) {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++) {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                if (column_equal_counter == r_table.column_count) {
                    probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                }
            }
            table_index = table_links[table_index];
        }
    }
}

__device__ void partial_probe_kernel(db_table r_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int key_offset, int slots, hash_t *table_hashes, index_s_t *table_links, index_s_t *table_slots) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // probe values from table s on hash table with values from r
    for (int s_index = index; s_index < s_table.size; s_index += stride) {
        // memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        while (table_index) {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = table_hashes[table_index];

            index_s_t probe_index = s_index * r_table.size + table_index;

            if (s_hash == r_hash) {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++) {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                if (column_equal_counter == r_table.column_count) {
                    probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                }
            }
            table_index = table_links[table_index];
        }
    }
}

__global__ void partial_build_and_probe_kernel(db_table r_table, db_hash_table r_hash_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_sizes_buffer, index_s_t *probe_results, int key_offset, int slots, int block_elements) {
    int table_full_size = r_hash_table.size;

    // calculate hash / table offset for r lookup in block
    // global == default for all blocks
    int hash_global_size = block_elements;
    int table_offset = hash_global_size * blockIdx.x;
    int hash_block_size = hash_global_size;
    if (blockIdx.x == gridDim.x - 1) {
        hash_block_size = r_table.size - table_offset;
    }

    r_table.column_values = &r_table.column_values[r_table.column_count * table_offset];
    r_hash_table.hashes = &r_hash_table.hashes[table_offset];
    r_hash_table.size = hash_block_size;
    r_table.size = hash_block_size;

    // hash table
    extern __shared__ hash_t *table[];
    hash_t *table_hashes = (hash_t *)&table[0];
    index_s_t *table_links = (index_s_t *)&table_hashes[hash_block_size];
    index_s_t *table_slots = (index_s_t *)&table_links[hash_block_size];

    // build phase
    // build hash table from hashes in table r
    build_hash_table(r_hash_table, key_offset, slots, table_hashes, table_links, table_slots);
    __syncthreads();

    // probing
    // probe values from table s on hash table with values from r
    probe_kernel(r_table, s_table, s_hash_table, probe_sizes_buffer, probe_results, table_offset, table_full_size, key_offset, slots, table_hashes, table_links, table_slots);
}

__global__ void build_and_partial_probe_kernel(db_table r_table, db_hash_table r_hash_table, db_table s_table, db_hash_table s_hash_table, index_s_t *probe_results_size, index_s_t *probe_results, int key_offset, int slots) {
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

__global__ void build_global_kernel(db_hash_table r_hash_table, int key_offset, int slots, index_s_t *table_links, index_s_t *table_slots) {

    // build
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r_index = index; r_index < r_hash_table.size; r_index += stride) {
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
    for (int s_index = index; s_index < s_table.size; s_index += stride) {
        // memset(&probe_results[r_table.size * s_index], 0, r_table.size * sizeof(filter_mask));
        hash_t s_hash = s_hash_table.hashes[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_s_t s_slot = s_key % slots;

        index_s_t table_index = table_slots[s_slot];
        while (table_index) {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = r_hash_table.hashes[table_index];

            index_s_t probe_index = s_index * r_table.size + table_index;

            if (s_hash == r_hash) {
                int column_equal_counter = 0;
                const index_t r_column_value_index = table_index * r_table.column_count;
                const index_t s_column_value_index = s_index * s_table.column_count;

                // probe every key column but ignore the primary key column
                for (int column_index = 0; column_index < r_table.column_count; column_index++) {
                    column_t r_column_value = r_table.column_values[r_column_value_index + column_index];
                    column_t s_column_value = s_table.column_values[s_column_value_index + column_index];
                    column_equal_counter += (r_column_value == s_column_value);
                }

                // table entries do match if both table entries have the same column values
                if (column_equal_counter == r_table.column_count) {
                    probe_results[atomicAdd(probe_results_size, 1)] = probe_index;
                }
            }
            table_index = table_links[table_index];
        }
    }
}

__global__ void copy_probe_results_kernel(db_table r_table, db_table s_table, index_s_t *indices, db_table rs_table) {
    // int rs_half_column_count = rs_table.column_count / 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_s_t buffer_index = index; buffer_index < rs_table.size; buffer_index += stride) {
        index_s_t copy_index = indices[buffer_index];
        index_s_t s_index = copy_index / r_table.size;
        index_s_t r_index = copy_index % r_table.size;

        index_s_t rs_offset = buffer_index * rs_table.column_count;
        // rs_table.primary_keys[buffer_index] = buffer_index + 1;

        rs_table.column_values[rs_offset] = r_table.primary_keys[r_index];
        rs_table.column_values[rs_offset + 1] = s_table.primary_keys[s_index];
    }
}

void partition_gpu(db_table d_table, db_hash_table d_hash_table, db_table d_table_swap, db_hash_table d_hash_table_swap, int radix_shift, index_t *histogram, index_t *offsets, PartitionConfig &partition_config) {
    assert(d_hash_table.size == d_hash_table_swap.size);
    partition_config.profiling_summary.elements = d_table.size;

    hash_t radix_mask = get_radix_mask(partition_config.bins);
    int bins = partition_config.bins;
    cudaStream_t stream = partition_config.stream;

    index_t *d_histogram = partition_config.d_histogram;
    index_t *d_offsets = partition_config.d_offsets;
    index_t *d_dest_indices = partition_config.d_dest_indices;

    gpuErrchk(cudaMemsetAsync(d_histogram, 0, bins * sizeof(index_t), stream));
    gpuErrchk(cudaMemsetAsync(d_dest_indices, 0, bins * sizeof(index_t), stream));

    int histogram_threads = partition_config.histogram_threads;
    int histogram_blocks = max(1ULL, d_hash_table.size / histogram_threads / partition_config.histogram_n_elements_p_thread);

    partition_config.start_profiling();
    histogram_kernel<<<histogram_blocks, histogram_threads, bins * sizeof(index_t), stream>>>(d_hash_table.size, d_hash_table.hashes, bins, d_histogram, radix_shift, radix_mask);
    partition_config.stop_profiling();
    if (partition_config.profiling_enabled) {
        float runtime_s = partition_config.get_elapsed_time_s();
        partition_config.profiling_summary.k_histogram_second = runtime_s;
        partition_config.profiling_summary.k_histogram_elements_p_second = d_hash_table.size / runtime_s;
        partition_config.profiling_summary.k_histogram_gb_p_second = d_hash_table.size * sizeof(hash_t) / runtime_s / pow(10, 9);
    }

    partition_config.start_profiling();
    prefix_sum_kernel<<<1, max(bins, 32), sizeof(index_t) * bins, stream>>>(bins, d_histogram, d_offsets);
    if (partition_config.profiling_enabled) {
        float runtime_s = partition_config.get_elapsed_time_s();
        partition_config.profiling_summary.k_prefix_second = runtime_s;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpyAsync(histogram, d_histogram, bins * sizeof(index_t), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(offsets, d_offsets, bins * sizeof(index_t), cudaMemcpyDeviceToHost, stream));

    // swap elments according to bin key
    int swap_threads = partition_config.swap_threads;
    int swap_blocks = max(1ULL, d_table.size / swap_threads / partition_config.swap_n_elements_p_thread);
    partition_config.start_profiling();
    element_swap_kernel<<<swap_blocks, swap_threads, 0, partition_config.stream>>>(d_table, d_hash_table, d_table_swap, d_hash_table_swap, d_offsets, d_dest_indices, radix_shift, radix_mask);
    partition_config.stop_profiling();
    if (partition_config.profiling_enabled) {
        float runtime_s = partition_config.get_elapsed_time_s();
        partition_config.profiling_summary.k_swap_second = runtime_s;
        partition_config.profiling_summary.k_swap_elements_p_second = d_table.size / runtime_s;
        int swap_element_size = (sizeof(hash_t) + d_table.column_count * sizeof(column_t));
        partition_config.profiling_summary.k_swap_gb_p_second = d_table.size * swap_element_size / runtime_s / pow(10, 9);
    }
}

struct is_greater_zero {
    __host__ __device__ bool operator()(const index_s_t &x) {
        return x > 0;
    }
};

JoinStatus build_and_probe_gpu(db_table d_r_table, db_hash_table d_r_hash_table, db_table d_s_table, db_hash_table d_s_hash_table, db_table &d_joined_rs_table, int key_offset, ProbeConfig &config) {
    // MAX-Q / Quadro / Turing 7.5
    // 32 Shared Banks
    // 64KB shared mem   / SM
    // 64k 32Bit reg     / SM
    // 255 32Bit reg     / Thread

    // 16 Blocks         / SM (16 x 64 Threads)
    // 32 Warps          / SM
    // 1024 Threads      / SM

    // cudaStreamSynchronize(stream);
    // gpuErrchk(cudaGetLastError());

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
    if (config.max_probe_buffer_size < config.probe_buffer_size) {
        if (config.d_probe_buffer) {
            gpuErrchk(cudaFreeAsync(config.d_probe_buffer, config.stream));
        }
        gpuErrchk(cudaMallocAsync(&config.d_probe_buffer, config.probe_buffer_size * sizeof(index_s_t), config.stream));
        config.max_probe_buffer_size = config.probe_buffer_size;
    }

    bool reallocate = !config.d_probe_result_size;
    if (reallocate) {
        gpuErrchk(cudaMallocAsync(&config.d_probe_result_size, sizeof(index_s_t), config.stream));
    }

    gpuErrchk(cudaMemsetAsync(config.d_probe_result_size, 0, sizeof(index_s_t), config.stream));

    if (config.probe_mode == ProbeConfig::MODE_GLOBAL_R) {
        if (slots > config.max_table_slots) {
            config.max_table_slots = slots;
            if (config.d_table_slots) {
                gpuErrchk(cudaFreeAsync(config.d_table_slots, config.stream));
            }
            gpuErrchk(cudaMallocAsync(&config.d_table_slots, slots * sizeof(index_s_t), config.stream));
        }

        if (d_r_hash_table.size > config.max_table_links) {
            config.max_table_links = d_r_hash_table.size;
            if (config.d_table_links) {
                gpuErrchk(cudaFreeAsync(config.d_table_links, config.stream));
            }
            gpuErrchk(cudaMallocAsync(&config.d_table_links, d_r_hash_table.size * sizeof(index_s_t), config.stream));
        }
    }

    if (config.profiling_enabled) {
        cudaEventRecord(config.profiling_start, config.stream);
    }

    assert(d_r_hash_table.size > 0);
    assert(d_s_hash_table.size > 0);

    /*
     * Build + Probe Kernels
     * Selects one by probe_mode
     */
    if (config.probe_mode == ProbeConfig::MODE_PARTITION_R) {
        int max_hash_table_size = min(config.get_max_table_elements(), (int)d_r_hash_table.size);
        int max_block_elements = min(config.build_threads * config.build_n_per_thread, max_hash_table_size);

        int blocks = max(1.0f, ceil((float)d_r_hash_table.size / max_block_elements));
        slots = config.get_table_slots(max_block_elements);
        int shared_mem = config.get_table_size(max_block_elements, slots);
        assert(blocks > 0);
        assert(shared_mem > 0);
        assert(config.build_threads > 0);
        assert(slots > 0);
        assert(max_block_elements > 0);
        assert(shared_mem < 48000);
        partial_build_and_probe_kernel<<<blocks, config.build_threads, shared_mem, config.stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, config.d_probe_result_size, config.d_probe_buffer, key_offset, slots, max_block_elements);
    } else if (config.probe_mode == ProbeConfig::MODE_PARTITION_S) {
        int blocks = ceil(d_s_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
        int shared_mem = config.get_table_size(d_r_hash_table.size, slots);
        // assert(shared_mem <= 49000);

        assert(blocks > 0);
        assert(config.build_threads > 0);
        assert(shared_mem > 0);
        assert(config.build_threads > 0);
        assert(shared_mem < 48000);
        build_and_partial_probe_kernel<<<blocks, config.build_threads, shared_mem, config.stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, config.d_probe_result_size, config.d_probe_buffer, key_offset, slots);
    } else if (config.probe_mode == ProbeConfig::MODE_GLOBAL_R) {
        gpuErrchk(cudaMemsetAsync(config.d_table_slots, 0, sizeof(index_s_t) * config.max_table_slots, config.stream));

        int build_blocks = ceil(d_r_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));
        int probe_blocks = ceil(d_s_hash_table.size / (float)(config.build_threads * config.build_n_per_thread));

        // assert(shared_mem <= 49000);

        assert(build_blocks > 0);
        assert(probe_blocks > 0);
        assert(config.build_threads > 0);
        build_global_kernel<<<build_blocks, config.build_threads, 0, config.stream>>>(d_r_hash_table, key_offset, slots, config.d_table_links, config.d_table_slots);
        probe_global_kernel<<<probe_blocks, config.build_threads, 0, config.stream>>>(d_r_table, d_r_hash_table, d_s_table, d_s_hash_table, config.d_probe_result_size, config.d_probe_buffer, key_offset, slots, config.d_table_links, config.d_table_slots);
    } else {
        return JoinStatus(false, "Unknown probing mode");
    }

    // profile build+probe
    if (config.profiling_enabled) {
        cudaEventRecord(config.profiling_end, config.stream);
        cudaEventSynchronize(config.profiling_end);
        float runtime_ms = 0.0;
        cudaEventElapsedTime(&runtime_ms, config.profiling_start, config.profiling_end);
        float runtime_s = runtime_ms / pow(10, 3);
        config.profiling_summary.k_build_probe_tuples_p_second = (d_r_hash_table.size + d_s_hash_table.size) / runtime_s;
        int tuple_size = sizeof(column_t) * (d_r_table.column_count) + sizeof(hash_t);
        config.profiling_summary.k_build_probe_gb_p_second = (d_r_hash_table.size + d_s_hash_table.size) * tuple_size / runtime_s / pow(10, 9);
    }

    /*
     * Fetch probe results
     * Merges matches in rs table
     */
    d_joined_rs_table.column_count = 2; // d_r_table.column_count + d_s_table.column_count;
    d_joined_rs_table.gpu = true;
    d_joined_rs_table.data_owner = true;
    gpuErrchk(cudaMemcpyAsync(&d_joined_rs_table.size, config.d_probe_result_size, sizeof(index_s_t), cudaMemcpyDeviceToHost, config.stream));
    gpuErrchk(cudaStreamSynchronize(config.stream));
    // gpuErrchk(cudaGetLastError());

    if (d_joined_rs_table.size > 0) {
        /*
        auto rs_table_memory = (d_joined_rs_table.column_count + 1) * d_joined_rs_table.size * sizeof(column_t);
        if (out_of_memory(rs_table_memory)) {
            return JoinStatus(false, "Not enough memory for rs table (" + std::to_string(d_joined_rs_table.size) + " Rows)");
        }
        */

        gpuErrchk(cudaMallocAsync(&d_joined_rs_table.primary_keys, d_joined_rs_table.size * sizeof(column_t), config.stream));
        gpuErrchk(cudaMallocAsync(&d_joined_rs_table.column_values, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), config.stream));
        gpuErrchk(cudaMemsetAsync(d_joined_rs_table.column_values, 0, d_joined_rs_table.column_count * d_joined_rs_table.size * sizeof(column_t), config.stream));

        int extract_blocks = max(1ULL, d_joined_rs_table.size / (config.extract_n_per_thread * config.extract_threads));
        int extract_threads_per_block = config.extract_threads;
        assert(extract_blocks > 0);
        assert(extract_threads_per_block > 0);

        if (config.profiling_enabled) {
            cudaEventRecord(config.profiling_start, config.stream);
        }

        copy_probe_results_kernel<<<extract_blocks, extract_threads_per_block, 0, config.stream>>>(d_r_table, d_s_table, config.d_probe_buffer, d_joined_rs_table);

        if (config.profiling_enabled) {
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

    if (config.profiling_enabled) {
        config.profiling_summary.r_elements = d_r_table.size;
        config.profiling_summary.s_elements = d_s_table.size;
        config.profiling_summary.rs_elements = d_joined_rs_table.size;
    }
    return JoinStatus(true);
}