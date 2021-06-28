#pragma once

#include <iostream>
#include <bitset>

#define ERROR_CHECK 0
#define DEBUG_PRINT 0
#define FILTER_VERSION 0

#ifndef BENCHMARK_TIME
    #define BENCHMARK_TIME 0
#endif

typedef unsigned long long hash_t;
typedef unsigned long long index_t;

struct db_entry {
    hash_t * hashes = nullptr;
    index_t * indices = nullptr;
    index_t size;

    void print() const {
        for(index_t hash_index = 0; hash_index < size; hash_index++) {
            std::cout << std::bitset<sizeof(hash_t) * 8>(hashes[hash_index]) << "::" << hashes[hash_index] << std::endl;
        }
    } 
};

struct db_column_data {
    size_t column_count = 0;
    int * column_data = nullptr;
};

typedef unsigned long long filter_int;
typedef bool filter_mask;
