#pragma once
#include <iostream>
//#include <<experimental/random>
#include <random>
#include "base/types.hpp"

template<class V>
void generate_random_numbers(size_t size, V max_int, V *buffer) {
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_int_distribution<V> distr(0, max_int);

    for(size_t x_index = 0; x_index < size; x_index++) {
        buffer[x_index] = std::rand() % (max_int + (V)1); //distr(gen);
    }
}

void generate_table(index_t table_size, int column_count, column_t max_int, db_table &table_data, bool gpu=true) {
    // +1 = primary key column
    table_data.column_count = column_count;
    table_data.gpu = gpu;
    table_data.column_values = new column_t[table_data.column_count * table_size];
    table_data.primary_keys = new column_t[table_size];
    table_data.size = table_size;
    table_data.data_owner = true;

    for(int table_index = 0; table_index < table_size; table_index++) {
        table_data.primary_keys[table_index] = table_index+1;
        generate_random_numbers(table_data.column_count, max_int, &table_data.column_values[table_index * table_data.column_count]);
    }

    if(gpu) {
        column_t * d_column_values = nullptr;
        gpuErrchk(cudaMalloc(&d_column_values, table_size *  table_data.column_count * sizeof(column_t)));
        gpuErrchk(cudaMemcpy(d_column_values, table_data.column_values, table_size * table_data.column_count *  sizeof(column_t), cudaMemcpyHostToDevice));
        delete[] table_data.column_values;
        table_data.column_values = d_column_values;
    }    

        column_t * d_primary_keys = nullptr;
        gpuErrchk(cudaMalloc(&d_primary_keys, table_size * sizeof(column_t)));
        gpuErrchk(cudaMemcpy(d_primary_keys, table_data.primary_keys, table_size * sizeof(column_t), cudaMemcpyHostToDevice));
        delete[] table_data.primary_keys;
        table_data.primary_keys = d_primary_keys;
}