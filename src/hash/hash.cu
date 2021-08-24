#pragma once
#include "base/types.hpp"

typedef char data_t;
typedef uint4 chunk_t;


__host__ __device__
constexpr hash_t get_fnv_prime() {
    if(sizeof(hash_t) <= 4) {
        return (uint32_t)(1 << 24) + (uint32_t)(1 << 8) + 0x93;
    } else {
        return (uint64_t)(1 << 40) + (uint64_t)(1 << 8) + 0xb3;
    }
}

__host__ __device__
constexpr hash_t get_fnv_offset() {
    if(sizeof(hash_t) <= 4) {
        return 2166136261;
    } else {
        return 14695981039346656037;
    }
}


__host__ __device__
constexpr unsigned short int get_max_shift() {
    short int shifts = sizeof(chunk_t) / sizeof(uint);
    short int chunk_element_size = sizeof(uint) * 8;;
    short int hash_size = sizeof(hash_t) * 8;
    short int max_shift = hash_size - (chunk_element_size + shifts);
    return max_shift > 0 ? max_shift : 1;
}

__host__ __device__
constexpr unsigned short int get_shift_increment() {
    return sizeof(hash_t) > sizeof(uint);
}

__host__ __device__
constexpr int64_t get_max_value(unsigned short int bytes) {
    return 1 << bytes * 8 - 1;
}

__global__
void hash_fnv(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    const hash_t FNV_prime = get_fnv_prime();
    const hash_t FNV_offste_bias = get_fnv_offset();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = FNV_offste_bias;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash ^= chunk.x;
            hash *= FNV_prime;
            hash ^= chunk.y;
            hash *= FNV_prime;
            hash ^= chunk.z;
            hash *= FNV_prime;
            hash ^= chunk.w;
            hash *= FNV_prime;
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_xor(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash_offset = hash_offset % get_max_shift();
            hash ^= ((hash_t) chunk.x) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.y) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.z) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.w) << hash_offset;
            hash_offset += get_shift_increment();
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_add(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash += ((hash_t) chunk.x);
            hash += ((hash_t) chunk.y);
            hash += ((hash_t) chunk.z);
            hash += ((hash_t) chunk.w);
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_mult(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    std::is_same<bool, chunk_t>();
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash = (hash * ((hash_t) chunk.x+1)); //% get_max_value(sizeof(hash_t));
            hash = (hash * ((hash_t) chunk.y+1)); //% get_max_value(sizeof(hash_t));
            hash = (hash * ((hash_t) chunk.z+1)); //% get_max_value(sizeof(hash_t));
            hash = (hash * ((hash_t) chunk.w+1)); //% get_max_value(sizeof(hash_t));
        }
        
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_xor_shift(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash_offset = (chunk.x + hash_offset) % get_max_shift();
            hash ^= ((hash_t) chunk.x) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.y) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.z) << hash_offset;
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.w) << hash_offset;
            hash_offset += get_shift_increment();
            
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_mult_xor_shift(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash_offset = (chunk.x + hash_offset) % get_max_shift();
            hash = (hash * ((hash_t) chunk.x << hash_offset)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.y) << hash_offset;
            hash_offset += get_shift_increment();
            hash = (hash * ((hash_t) chunk.z << hash_offset)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
            hash ^= ((hash_t) chunk.w) << hash_offset;
            hash_offset += get_shift_increment();
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_n_mult_xor_shift(index_t element_buffer_size, short int element_size, short int element_chunks, short int rounds, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            for(short int round_index = 0; round_index < rounds; round_index++) {
                hash_offset = (hash_offset + chunk.x) % get_max_shift();
                hash = (hash * ((hash_t) chunk.x << hash_offset)) ; //% get_max_value(sizeof(hash_t));
                hash_offset += get_shift_increment();
                hash ^= ((hash_t) chunk.y) << hash_offset;
                hash_offset += get_shift_increment();
                hash = (hash * ((hash_t) chunk.z << hash_offset)) ; //% get_max_value(sizeof(hash_t));
                hash_offset += get_shift_increment();
                hash ^= ((hash_t) chunk.w) << hash_offset;
                hash_offset += get_shift_increment();
            }
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_mult_shift(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    std::is_same<bool, chunk_t>();
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash_offset = chunk.x % get_max_shift();
            hash = (hash * ((hash_t) chunk.x+1)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
            hash = (hash * ((hash_t) chunk.y+1)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
            hash = (hash * ((hash_t) chunk.z+1)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
            hash = (hash * ((hash_t) chunk.w+1)) ; //% get_max_value(sizeof(hash_t));
            hash_offset += get_shift_increment();
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__
void hash_custom_add_shift(index_t element_buffer_size, short int element_size, short int element_chunks, data_t* element_buffer, hash_t * hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks * sizeof(chunk_t);
        chunk_t * chunked_element_buffer = reinterpret_cast<chunk_t*>(&element_buffer[buffer_index]);
        short int hash_offset = 0;
        for(short int chunk_index = 0; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = chunked_element_buffer[chunk_index];
            hash_offset = chunk.x % get_max_shift();
            hash += ((hash_t) chunk.x) << hash_offset;
            hash_offset += get_shift_increment();
            hash += ((hash_t) chunk.y) << hash_offset;
            hash_offset += get_shift_increment();
            hash += ((hash_t) chunk.z) << hash_offset;
            hash_offset += get_shift_increment();
            hash += ((hash_t) chunk.w) << hash_offset;
            hash_offset += get_shift_increment();
        }
        hashed_buffer[element_index] = hash;
    }
}