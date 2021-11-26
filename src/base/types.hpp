#pragma once
#include "config.h"
#include "debug.cuh"

typedef unsigned long long index_t;
typedef unsigned int index_s_t;
typedef unsigned long long column_t;

#if HASH_BITS == 32
typedef unsigned int hash_t;
#elif HASH_BITS == 64
typedef unsigned long long hash_t;
#endif

#if HASH_CHUNK_BITS == 8
typedef char chunk_t;
#elif HASH_CHUNK_BITS == 32
typedef uint1 chunk_t;
#elif HASH_CHUNK_BITS == 64
typedef uint2 chunk_t;
#elif HASH_CHUNK_BITS == 128
typedef uint4 chunk_t;
#endif

struct db_hash_table {
    hash_t *hashes = nullptr;
    // index_t * indices = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;

    // owner of data
    int device_index = 0;

    db_hash_table() {
        hashes = nullptr;
        // indices = nullptr;
        size = 0;
        data_owner = true;
    }

    db_hash_table(index_t table_size, bool gpu = true, int device_index = 0) {
        this->size = table_size;
        this->gpu = gpu;
        this->data_owner = true;
        this->device_index = device_index;

        if (gpu) {
            gpuErrchk(cudaSetDevice(device_index));
            gpuErrchk(cudaMalloc(&hashes, size * sizeof(hash_t)));
        } else {
            hashes = new hash_t[size];
        }
    }

    db_hash_table(index_t offset, index_t size, db_hash_table table_source) {
        this->size = size;
        this->gpu = table_source.gpu;
        this->hashes = &table_source.hashes[offset];
        this->data_owner = false;
        this->device_index = table_source.device_index;
    }

    __device__ __host__ db_hash_table(const db_hash_table &table) {
        this->data_owner = table.data_owner;
        this->gpu = table.gpu;
        this->device_index = table.device_index;
        this->size = table.size;
        this->hashes = table.hashes;
    }

    void print() const {
        hash_t *h_hashes;
        if (gpu) {
            h_hashes = new hash_t[size];
            gpuErrchk(cudaMemcpy(h_hashes, hashes, size * sizeof(hash_t), cudaMemcpyDeviceToHost));
        } else {
            h_hashes = hashes;
        }

        for (index_t hash_index = 0; hash_index < size; hash_index++) {
            std::cout << std::bitset<sizeof(hash_t) * 8>(h_hashes[hash_index]) << "::" << h_hashes[hash_index] << std::endl;
        }

        if (gpu) {
            delete[] h_hashes;
        }
    }

    int get_bytes() {
        return db_hash_table::get_bytes(size);
    }

    static int get_bytes(index_t size) {
        return sizeof(hash_t) * size;
    }

    db_hash_table copyAsync(int device_index = 0, cudaStream_t stream = 0) {
        db_hash_table copy_table;
        copy_table.device_index = device_index;
        gpuErrchk(cudaSetDevice(device_index));
        gpuErrchk(cudaMallocAsync(&copy_table.hashes, size * sizeof(hash_t), stream));
        gpuErrchk(cudaMemcpyAsync(copy_table.hashes, hashes, size * sizeof(hash_t), cudaMemcpyDeviceToDevice, stream));
        copy_table.data_owner = true;
        copy_table.gpu = true;
        copy_table.size = size;
        return copy_table;
    }

    void free() {
        if (data_owner) {
            if (gpu) {
                gpuErrchk(cudaSetDevice(device_index));
                gpuErrchk(cudaFree(hashes));
                // gpuErrchk(cudaFree(indices));
            } else {
                delete[] hashes;
                // delete[] indices;
            }
        }
        size = 0;
        hashes = nullptr;
        // indices = nullptr;
    }
};

struct db_table {
    size_t column_count = 0;
    column_t *primary_keys = nullptr;
    column_t *column_values = nullptr;
    index_t size;
    bool gpu = false;
    bool data_owner = false;
    int device_index = 0;

    db_table() {
        column_count = 0;
        column_values = nullptr;
        primary_keys = nullptr;
        size = 0;
        data_owner = true;
        device_index = 0;
    }

    db_table(size_t column_count, index_t table_size, bool gpu = true) {
        this->column_count = column_count;
        this->size = table_size;
        this->gpu = gpu;
        this->data_owner = true;

        if (gpu) {
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

    db_table copyAsync(int device_index, cudaStream_t stream = 0) {
        if (gpu) {
            db_table table_copy;
            table_copy.column_count = column_count;
            table_copy.gpu = true;
            table_copy.data_owner = true;
            table_copy.size = size;
            table_copy.device_index = device_index;
            gpuErrchk(cudaSetDevice(device_index));
            gpuErrchk(cudaMallocAsync(&table_copy.column_values, table_copy.size * table_copy.column_count * sizeof(column_t), stream));
            gpuErrchk(cudaMallocAsync(&table_copy.primary_keys, table_copy.size * sizeof(column_t), stream));
            gpuErrchk(cudaMemcpyAsync(table_copy.column_values, column_values, table_copy.size * table_copy.column_count * sizeof(column_t), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpyAsync(table_copy.primary_keys, primary_keys, table_copy.size * sizeof(column_t), cudaMemcpyDeviceToDevice));
            return table_copy;
        } else {
            return copy(false);
        }
    }

    db_table copy(bool to_gpu) {
        db_table table_copy;
        table_copy.column_count = column_count;
        table_copy.gpu = to_gpu;
        table_copy.data_owner = true;
        table_copy.size = size;
        if (to_gpu) {
        } else {
            if (gpu) {
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
        return db_table::get_bytes(size, column_count);
    }

    void print() {
        db_table h_table;
        if (gpu) {
            h_table = copy(false);
        } else {
            h_table = *this;
        }

        for (index_t value_index = 0; value_index < h_table.size; value_index++) {
            std::cout << value_index << "\t\t" << h_table.primary_keys[value_index] << "\t\t";
            for (int column_index = 0; column_index < h_table.column_count; column_index++) {
                std::cout << h_table.column_values[value_index * column_count + column_index] << "\t\t";
            }
            std::cout << std::endl;
        }

        if (gpu) {
            h_table.free();
        }
    }

    void free() {
        free(0);
    }

    void free(cudaStream_t stream) {
        if (size > 0) {
            if (data_owner) {
                if (gpu) {
                    gpuErrchk(cudaSetDevice(device_index));
                    if (stream) {
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

    static size_t get_bytes(index_t elements, int columns) {
        return sizeof(column_t) * elements * (columns + 1);
    }
};

typedef unsigned long long filter_int;
typedef bool filter_mask;