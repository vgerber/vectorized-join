#pragma once
#include "config.h"
#include "debug.cuh"

typedef unsigned long long hash_t;
typedef unsigned long long index_t;
typedef unsigned int index_s_t;
typedef unsigned long long column_t;
typedef uint1 chunk_t;

struct db_hash_table {
    hash_t * hashes = nullptr;
    //index_t * indices = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;

    db_hash_table() {
        hashes = nullptr;
        //indices = nullptr;
        size = 0;
        data_owner = true;
    }

    db_hash_table(index_t table_size, bool gpu=true) {
        this->size = table_size;
        this->gpu = gpu;
        this->data_owner = true;

        if(gpu) {
            gpuErrchk(cudaMalloc(&hashes, size * sizeof(hash_t)));
            //gpuErrchk(cudaMalloc(&indices, size * sizeof(index_t)));
        } else {
            hashes = new hash_t[size];
            //indices = new index_t[size];
        }
    }

    db_hash_table(index_t offset, index_t size, db_hash_table table_source) {
        this->size = size;
        this->gpu = table_source.gpu;
        this->hashes = &table_source.hashes[offset];
        //this->indices = &table_source.indices[offset];
        this->data_owner = false;
    }

    void print() const {
        hash_t *h_hashes;
        if(gpu) {
            h_hashes = new hash_t[size];
            gpuErrchk(cudaMemcpy(h_hashes, hashes, size * sizeof(hash_t), cudaMemcpyDeviceToHost));
            
        } else {
            h_hashes = hashes;
        }

        for(index_t hash_index = 0; hash_index < size; hash_index++) {
            std::cout << std::bitset<sizeof(hash_t) * 8>(h_hashes[hash_index]) << "::" << h_hashes[hash_index] << std::endl;
        }

        if(gpu) {
            delete[] h_hashes;
        }
    } 

    int get_bytes() {
        return size * sizeof(hash_t);
    }

    void free() {
        if(data_owner) {
            if(gpu) {
                gpuErrchk(cudaFree(hashes));
                //gpuErrchk(cudaFree(indices));
            } else {
                delete[] hashes;
                //delete[] indices;
            }
        }
        size = 0;
        hashes = nullptr;
        //indices = nullptr;
    }
};

struct db_table {
    size_t column_count = 0;
    column_t * primary_keys = nullptr;
    column_t * column_values = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;

    db_table() {
        column_count = 0;
        column_values = nullptr;
        primary_keys = nullptr;
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
            gpuErrchk(cudaMalloc(&primary_keys, table_size * sizeof(column_t)));
        } else {
            column_values = new column_t[column_count * table_size];
            primary_keys = new column_t[table_size];
        }
    }

    db_table(index_t offset, index_t size, db_table table_source) {
        this->size = size;
        this->gpu = table_source.gpu;
        this->column_count = table_source.column_count;
        this->data_owner = false;
        this->column_values = &table_source.column_values[offset * column_count];
        this->primary_keys = &table_source.primary_keys[offset];

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
                table_copy.primary_keys = new column_t[table_copy.size];
                gpuErrchk(cudaMemcpy(table_copy.primary_keys, primary_keys, table_copy.size * sizeof(column_t), cudaMemcpyDeviceToHost));
            } else {
                table_copy.column_values = new column_t[table_copy.size * table_copy.column_count];
                memcpy(table_copy.column_values, column_values, table_copy.size * table_copy.column_count * sizeof(column_t));
                table_copy.primary_keys = new column_t[table_copy.size];
                memcpy(table_copy.primary_keys, primary_keys, table_copy.size * sizeof(column_t));
            }
        }
        return table_copy;
    }

    int get_bytes() {
        return size * (column_count + 1) * sizeof(column_t);
    }

    void print() {
        db_table h_table;
        if(gpu) {
            h_table = copy(false);
        } else {
            h_table = *this;
        }

        for(index_t value_index = 0; value_index < h_table.size; value_index++) {
            std::cout << value_index << "\t\t" << h_table.primary_keys[value_index] << "\t\t";
            for(int column_index = 0; column_index < h_table.column_count; column_index++) {
                std::cout << h_table.column_values[value_index * column_count + column_index] << "\t\t";
            }
            std::cout << std::endl;
        }

        if(gpu) {
            h_table.free();
        }
    }

    void free() {
        free(0);
    }

    void free(cudaStream_t stream) {
        if(size > 0) {
            if(data_owner) {
                if(gpu) {
                    if(stream) {
                        gpuErrchk(cudaFreeAsync(column_values, stream));
                        gpuErrchk(cudaFreeAsync(primary_keys, stream));
                    } else {
                        gpuErrchk(cudaFree(column_values));
                        gpuErrchk(cudaFree(primary_keys));
                    }
                    
                } else {
                    delete[] column_values;
                    delete[] primary_keys;
                }
            }
            column_count = 0;
            size = 0;
            column_values = nullptr;
            primary_keys = nullptr;
        }
    }
};

typedef unsigned long long filter_int;
typedef bool filter_mask;