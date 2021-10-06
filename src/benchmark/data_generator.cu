#pragma once
#include <iostream>
#include <random>
#include <curand.h>
#include "base/types.hpp"
#include "base/debug.cuh"

#define USE_CURAND 1

__global__
void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t* element_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        index_t buffer_index = element_index * element_chunks;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
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

__global__
void generate_demo_data_kernel(index_t element_buffer_size, short int element_size, short int element_chunks, chunk_t* element_buffer, float * distribution_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    index_t total_elements = element_buffer_size * element_chunks;
    for(index_t element_index = index; element_index < total_elements; element_index += stride) {
        element_buffer[element_index].x = fabsf(distribution_values[element_index]) * UINT32_MAX;
    }
}

__global__
void generate_demo_data_kernel(db_table table, float * distribution_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(index_t element_index = index; element_index < table.size; element_index += stride) {
        for(int column_index = 0; column_index < table.column_count; column_index++) {
            table.column_values[element_index * table.column_count + column_index] = fabsf(distribution_values[element_index]) * UINT64_MAX;
        }        
        table.primary_keys[element_index] = element_index + 1;
    }
}

void generate_demo_data(index_t elements, short int element_size, short int *element_chunks, chunk_t** buffer) {
    *element_chunks = element_size / sizeof(chunk_t) + (element_size % sizeof(chunk_t) > 0);
    gpuErrchk(cudaMalloc(buffer, elements * *element_chunks * sizeof(chunk_t)));

#if USE_CURAND
    float * d_distribution = nullptr;
    index_t distribution_values_count = elements * *element_chunks * (sizeof(chunk_t) / sizeof(uint32_t));
    gpuErrchk(cudaMalloc(&d_distribution, distribution_values_count * sizeof(float)));

    curandGenerator_t rand_gen;
    gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
    //gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, distribution_values_count));
    gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.01));
    //gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.3));
    gpuErrchk(curandDestroyGenerator(rand_gen));

    generate_demo_data_kernel<<<max(elements/256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer, d_distribution);
    
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_distribution));
#else    
    
    generate_demo_data_kernel<<<max(elements/256, 1ULL), 256>>>(elements, element_size, *element_chunks, *buffer);
#endif
}


void generate_table_data(db_table &table) {
    gpuErrchk(cudaMalloc(&table.primary_keys, table.size * sizeof(column_t)));
    gpuErrchk(cudaMalloc(&table.column_values, table.size * table.column_count * sizeof(column_t)));

#if USE_CURAND
    float * d_distribution = nullptr;
    index_t distribution_values_count = table.size * table.column_count;
    gpuErrchk(cudaMalloc(&d_distribution, distribution_values_count * sizeof(float)));

    curandGenerator_t rand_gen;
    gpuErrchk(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    gpuErrchk(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
    //gpuErrchk(curandGenerateUniform(rand_gen, d_distribution, distribution_values_count));
    gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.01));
    //gpuErrchk(curandGenerateNormal(rand_gen, d_distribution, distribution_values_count, 0.5f, 0.3));
    gpuErrchk(curandDestroyGenerator(rand_gen));

    generate_demo_data_kernel<<<max(table.size/256, 1ULL), 256>>>(table, d_distribution);
    
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_distribution));
#else    
    
    //generate_demo_data_kernel<<<max(elements/256, 1ULL), 256>>>(table_data);
#endif
}


void generate_table(index_t table_size, int column_count, db_table &table_data) {
    // +1 = primary key column
    table_data.column_count = column_count;
    table_data.gpu = true;
    table_data.column_values = new column_t[table_data.column_count * table_size];
    table_data.primary_keys = new column_t[table_size];
    table_data.size = table_size;
    table_data.data_owner = true;
    
    generate_table_data(table_data);
    gpuErrchk(cudaDeviceSynchronize());
}