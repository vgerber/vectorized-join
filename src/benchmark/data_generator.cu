#pragma once
#include "base/debug.cuh"
#include "base/types.hpp"
#include <curand.h>
#include <iostream>
#include <random>

#define USE_CURAND 1

#if USE_CURAND
// oe instance for each device
static std::vector<curandGenerator_t> rand_gens;
#endif

__global__ void zipf_constant_kernel(int max_rank, float skew, float *zipf_constant) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float tmp_zipf_constant = 0.f;
    for (int rank_index = index; rank_index < max_rank; rank_index += stride) {
        tmp_zipf_constant += (1.f / pow(rank_index + 1, skew));
        // printf("ZC %f\n", tmp_zipf_constant);
    }
    atomicAdd(zipf_constant, tmp_zipf_constant);
}

__global__ void zipf_distribution_kernel(int max_rank, float skew, float *zipf_constant, float *zipf_distribution) {
    float tmp_zipf_constant = 1.f / *zipf_constant;
    // printf("C %f %f\n", tmp_zipf_constant, *zipf_constant);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int rank_index = index; rank_index < max_rank; rank_index += stride) {
        float zipf_sum_prob = 0.f;
        for (int rank_iter_index = 0; rank_iter_index < rank_index; rank_iter_index++) {
            zipf_sum_prob += tmp_zipf_constant / pow(rank_iter_index + 1, skew);
        }

        zipf_distribution[rank_index] = zipf_sum_prob;
    }
}

__global__ void zipf_kernel(int elements, column_t *element_buffer, float *uniform_values, float *zipf_distribution, int max_rank, float skew) {
    float margin = 0.00001f;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int value_index = index; value_index < elements; value_index += stride) {
        float uniform_value = max(margin, min(uniform_values[value_index], 1.f - margin));
        for (int rank_index = 1; rank_index <= max_rank; rank_index++) {
            if (zipf_distribution[rank_index - 1] >= uniform_value) {
                element_buffer[value_index] = rank_index;
                break;
            }
        }
    }
}

__global__ void uniform_kernel(int elements, column_t *element_buffer, float *uniform_values, column_t max_value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int value_index = index; value_index < elements; value_index += stride) {
        element_buffer[value_index] = uniform_values[value_index] * max_value;
    }
}

__global__ void primary_key_kernel(db_table table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int element_index = index; element_index < table.size; element_index += stride) {
        table.primary_keys[element_index] = element_index + 1;
    }
}

__global__ void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t *element_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        index_t buffer_index = element_index * element_chunks;
        for (short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t data_chunk;
            data_chunk.x = element_index;
            /*
            data_chunk.y = element_index;
            data_chunk.z = element_index;
            data_chunk.w = element_index;
            */
            element_buffer[buffer_index + chunk_index] = data_chunk;
        }
    }
}

__global__ void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t *element_buffer, float *distribution_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        index_t buffer_index = element_index * element_chunks;
        for (int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {

#if HASH_CHUNK_BITS == 8
            element_buffer[buffer_index + chunk_index] = fabsf(distribution_values[buffer_index + chunk_index]) * CHAR_MAX;
#endif
#if HASH_CHUNK_BITS >= 32
            element_buffer[buffer_index + chunk_index].x = fabsf(distribution_values[buffer_index + chunk_index]) * UINT32_MAX;
#endif
#if HASH_CHUNK_BITS >= 64
            element_buffer[buffer_index + chunk_index].y = fabsf(distribution_values[buffer_index + chunk_index]) * UINT32_MAX;
#endif
#if HASH_CHUNK_BITS >= 128
            element_buffer[buffer_index + chunk_index].z = fabsf(distribution_values[buffer_index + chunk_index]) * UINT32_MAX;
            element_buffer[buffer_index + chunk_index].w = fabsf(distribution_values[buffer_index + chunk_index]) * UINT32_MAX;
#endif
        }
    }
}

__global__ void generate_demo_data_kernel(db_table table, float *distribution_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (index_t element_index = index; element_index < table.size; element_index += stride) {
        for (int column_index = 0; column_index < table.column_count; column_index++) {
            table.column_values[element_index * table.column_count + column_index] = fabsf(distribution_values[element_index]) * UINT64_MAX;
        }
        table.primary_keys[element_index] = element_index + 1;
    }
}

void generate_demo_data(index_t elements, short int element_size, short int *element_chunks, chunk_t **buffer) {
    *element_chunks = element_size / sizeof(chunk_t) + (element_size % sizeof(chunk_t) > 0);
    gpuErrchk(cudaMalloc(buffer, elements * *element_chunks * sizeof(chunk_t)));

#if USE_CURAND
    float *d_distribution = nullptr;
    index_t distribution_values_count = elements * *element_chunks;
    gpuErrchk(cudaMalloc(&d_distribution, distribution_values_count * sizeof(float)));

    if (rand_gens.size() == 0) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        rand_gens = std::vector<curandGenerator_t>(device_count);
    }

    int device_index = 0;
    gpuErrchk(cudaGetDevice(&device_index));
    curandGenerator_t rand_gen = rand_gens[device_index];
    if (!rand_gen) {
        gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
        rand_gens[device_index] = rand_gen;
    }
    gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, distribution_values_count));

    generate_demo_data_kernel<<<max(elements / 256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer, d_distribution);

    gpuErrchk(cudaFree(d_distribution));
#else

    generate_demo_data_kernel<<<max(elements / 256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer);
#endif
}

void generate_table_data(db_table &table, int max_value, float skew) {
    gpuErrchk(cudaMalloc(&table.primary_keys, table.size * sizeof(column_t)));
    gpuErrchk(cudaMalloc(&table.column_values, table.size * table.column_count * sizeof(column_t)));
    gpuErrchk(cudaGetLastError());

    float *d_distribution = nullptr;
    index_t distribution_values_count = table.size * table.column_count;
    gpuErrchk(cudaMalloc(&d_distribution, distribution_values_count * sizeof(float)));

    if (rand_gens.size() == 0) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        rand_gens = std::vector<curandGenerator_t>(device_count);
    }

    int device_index = 0;
    gpuErrchk(cudaGetDevice(&device_index));
    curandGenerator_t rand_gen = rand_gens[device_index];
    if (!rand_gen) {
        gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        // gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
        gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
        rand_gens[device_index] = rand_gen;
    }

    gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, distribution_values_count));
    // gpuErrchk(curandDestroyGenerator(rand_gen));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());

    if (skew > 0.0f) {
        float *d_zipf_constant = nullptr;
        float *d_zipf_distribution = nullptr;
        gpuErrchk(cudaMalloc(&d_zipf_constant, sizeof(float)));
        gpuErrchk(cudaMemset(d_zipf_constant, 0, sizeof(float)));
        gpuErrchk(cudaMalloc(&d_zipf_distribution, max_value * sizeof(float)));

        int threads = 256;
        zipf_constant_kernel<<<max(1, max_value / threads), threads>>>(max_value, skew, d_zipf_constant);
        zipf_distribution_kernel<<<max(1, max_value / threads), threads>>>(max_value, skew, d_zipf_constant, d_zipf_distribution);
        zipf_kernel<<<max(1ULL, table.size * max_value / threads), threads>>>(table.size, table.column_values, d_distribution, d_zipf_distribution, max_value, skew);

        gpuErrchk(cudaFree(d_zipf_constant));
        gpuErrchk(cudaFree(d_zipf_distribution));
    } else {
        uniform_kernel<<<max(1ULL, table.size / 256), 256>>>(table.size, table.column_values, d_distribution, max_value);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaGetLastError());
    }
    gpuErrchk(cudaFree(d_distribution));

    primary_key_kernel<<<max(table.size / 256, 1ULL), 256>>>(table);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());
}

void generate_table(index_t table_size, int column_count, db_table &table_data, int max_value, float skew) {
    // +1 = primary key column
    table_data.column_count = column_count;
    table_data.gpu = true;
    table_data.size = table_size;
    table_data.data_owner = true;

    generate_table_data(table_data, max_value, skew);
    gpuErrchk(cudaDeviceSynchronize());
}