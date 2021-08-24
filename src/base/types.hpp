#pragma once
#include "config.h"
#include "debug.cuh"

typedef uint32_t hash_t;
typedef unsigned long long index_t;
typedef unsigned int index_s_t;
typedef unsigned long long column_t;

struct db_hash_table {
    hash_t * hashes = nullptr;
    index_t * indices = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;

    db_hash_table() {
        hashes = nullptr;
        indices = nullptr;
        size = 0;
        data_owner = true;
    }

    db_hash_table(index_t table_size, bool gpu=true) {
        this->size = table_size;
        this->gpu = gpu;
        this->data_owner = true;

        if(gpu) {
            gpuErrchk(cudaMalloc(&hashes, size * sizeof(hash_t)));
            gpuErrchk(cudaMalloc(&indices, size * sizeof(index_t)));
        } else {
            hashes = new hash_t[size];
            indices = new index_t[size];
        }
    }

    db_hash_table(index_t offset, index_t size, db_hash_table table_source) {
        this->size = size;
        this->gpu = table_source.gpu;
        this->hashes = &table_source.hashes[offset];
        this->indices = &table_source.indices[offset];
        this->data_owner = false;
    }

    void print() const {
        for(index_t hash_index = 0; hash_index < size; hash_index++) {
            std::cout << std::bitset<sizeof(hash_t) * 8>(hashes[hash_index]) << "::" << hashes[hash_index] << std::endl;
        }
    } 

    void free() {
        if(data_owner) {
            if(gpu) {
                cudaFree(hashes);
                cudaFree(indices);
            } else {
                delete[] hashes;
                delete[] indices;
            }
        }
    }
};

struct db_table {
    size_t column_count = 0;
    column_t * column_values = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;

    db_table() {
        column_count = 0;
        column_values = nullptr;
        size = 0;
        data_owner = true;
    }

    db_table(size_t column_count, index_t table_size, bool gpu=true) {
        this->column_count = column_count;
        this->size = table_size;
        this->gpu = gpu;
        this->data_owner = true;

        if(gpu) {
            gpuErrchk(cudaMalloc(&column_values, column_count * table_size * sizeof(column_t)));
        } else {
            column_values = new column_t[column_count * table_size];
        }
    }

    db_table(index_t offset, index_t size, db_table table_source) {
        this->size = size;
        this->gpu = table_source.gpu;
        this->column_count = table_source.column_count;
        this->data_owner = false;
        this->column_values = &table_source.column_values[offset * column_count];

    }

    db_table copy(bool to_gpu) {
        db_table table_copy;
        table_copy.column_count = column_count;
        table_copy.gpu = to_gpu;
        table_copy.data_owner = true;
        table_copy.size = size;
        if(to_gpu) {

        } else {
            if(gpu) {
                table_copy.column_values = new column_t[table_copy.size * table_copy.column_count];
                gpuErrchk(cudaMemcpy(table_copy.column_values, column_values, table_copy.size * table_copy.column_count * sizeof(column_t), cudaMemcpyDeviceToHost));
            }
        }
        return table_copy;
    }

    void print() {
        db_table h_table;
        if(gpu) {
            h_table = copy(false);
        } else {
            h_table = *this;
        }

        for(index_t value_index = 0; value_index < h_table.size; value_index++) {
            std::cout << value_index << "\t\t";
            for(int column_index = 0; column_index < h_table.column_count; column_index++) {
                std::cout << h_table.column_values[value_index * column_count + column_index] << "\t\t";
            }
            std::cout << std::endl;
        }
        h_table.free();
    }

    void free() {
        if(data_owner) {
            if(gpu) {
                gpuErrchk(cudaFree(column_values));
            } else {
                delete[] column_values;
            }
        }
        column_values = nullptr;
    }
};

typedef unsigned long long filter_int;
typedef bool filter_mask;