#pragma once
#include "base/types.hpp"

struct HashSummary {
    std::string algorithm = "fnv";
    index_t elements = 0;
    int element_bytes = 0;
    int element_offset = 0;
    float k_tuples_p_seconds = 0.0;
    float k_gb_p_seconds = 0.0;

    void reset() {
        algorithm = "fnv";
        elements = 0;
        element_offset = 0;
        element_bytes = 0;
        k_tuples_p_seconds = 0.0;
        k_gb_p_seconds = 0.0;
    }
};

struct HashConfig {
    int threads_per_block = 256;
    int elements_per_thread = 1;
    std::string algorithm = "fnv";

    cudaStream_t stream = 0;

    bool profile_enabled = false;
    cudaEvent_t profile_start, profile_end;
    HashSummary profile_summary;

    void enable_profile(cudaEvent_t profile_start, cudaEvent_t profile_end) {
        this->profile_start = profile_start;
        this->profile_end = profile_end;
        profile_enabled = true;
    }

    void disbale_profile() {
        profile_enabled = false;
    }
};

__host__ __device__ constexpr hash_t get_fnv_prime() {
#if HASH_BITS == 32
    return (uint32_t)(1 << 24) + (uint32_t)(1 << 8) + 0x93;
#elif HASH_BITS == 64
    return (unsigned long long)(1ULL << 40) + (unsigned long long)(1 << 8) + 0xb3;
#endif
}

__host__ __device__ constexpr hash_t get_fnv_offset() {
#if HASH_BITS == 32
    return 2166136261;
#elif HASH_BITS == 64
    return 14695981039346656037;
#endif
}

__host__ __device__ constexpr unsigned short int get_max_shift() {
    return (unsigned short int)(sizeof(hash_t) - sizeof(chunk_t) * 8) + 1;
}

__device__ inline hash_t reverse(hash_t hash) {
#if HASH_BITS == 64
    return __brevll((unsigned long long)hash);
#elif HASH_BITS == 32
    return __brev((uint)hash);
#endif
}

__device__ inline unsigned int permutate(unsigned int value, int shift) {
    return __funnelshift_l(value, value, shift);
}

__host__ __device__ constexpr unsigned short int get_shift_increment() {
    return sizeof(hash_t) > sizeof(uint);
}

__host__ __device__ constexpr int64_t get_max_value(unsigned short int bytes) {
    return 1 << bytes * 8 - 1;
}

__global__ void hash_fnv(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    const hash_t FNV_prime = get_fnv_prime();
    const hash_t FNV_offste_bias = get_fnv_offset();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = FNV_offste_bias;
        index_t buffer_index = element_index * element_chunks;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash ^= chunk;
            hash *= FNV_prime;
#endif
#if HASH_CHUNK_BITS >= 32
            hash ^= chunk.x;
            hash *= FNV_prime;
#endif
#if HASH_CHUNK_BITS >= 64
            hash ^= chunk.y;
            hash *= FNV_prime;
#endif
#if HASH_CHUNK_BITS == 128
            hash ^= chunk.z;
            hash *= FNV_prime;
            hash ^= chunk.w;
            hash *= FNV_prime;
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_add(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;

        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash += ((hash_t)chunk);
#endif
#if HASH_CHUNK_BITS >= 32
            hash += ((hash_t)chunk.x);
#endif
#if HASH_CHUNK_BITS >= 64
            hash += ((hash_t)chunk.y);
#endif
#if HASH_CHUNK_BITS == 128
            hash += ((hash_t)chunk.z);
            hash += ((hash_t)chunk.w);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_add_shift(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;
        short int hash_offset = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash_offset = (hash_offset + chunk) % get_max_shift();
            hash += ((hash_t)chunk) << hash_offset;
#endif
#if HASH_CHUNK_BITS >= 32
            hash_offset = (hash_offset + chunk.x) % get_max_shift();
            hash += ((hash_t)chunk.x) << hash_offset;
#endif
#if HASH_CHUNK_BITS >= 64
            hash_offset = (hash_offset + chunk.y) % get_max_shift();
            hash += ((hash_t)chunk.y) << hash_offset;
#endif
#if HASH_CHUNK_BITS == 128
            hash_offset = (hash_offset + chunk.z) % get_max_shift();
            hash += ((hash_t)chunk.z) << hash_offset;
            hash_offset = (hash_offset + chunk.w) % get_max_shift();
            hash += ((hash_t)chunk.w) << hash_offset;
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_add_hw(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;
        int shift = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash += (hash_t)permutate(chunk, shift++);
#endif
#if HASH_CHUNK_BITS >= 32
            hash += (hash_t)permutate(chunk.x, shift++);
#endif
#if HASH_CHUNK_BITS >= 64
            hash += (hash_t)permutate(chunk.y, shift++);
#endif
#if HASH_CHUNK_BITS == 128
            hash += (hash_t)permutate(chunk.z, shift++);
            hash += (hash_t)permutate(chunk.w, shift++);
#endif
            hash = reverse(hash);
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_xor(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;

        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash ^= ((hash_t)chunk);
#endif
#if HASH_CHUNK_BITS >= 32
            hash ^= ((hash_t)chunk.x);
#endif
#if HASH_CHUNK_BITS >= 64
            hash ^= ((hash_t)chunk.y);
#endif
#if HASH_CHUNK_BITS == 128
            hash ^= ((hash_t)chunk.z);
            hash ^= ((hash_t)chunk.w);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_xor_shift(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;
        short int hash_offset = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash_offset = (hash_offset + chunk) % get_max_shift();
            hash ^= ((hash_t)chunk) << hash_offset;
#endif
#if HASH_CHUNK_BITS >= 32
            hash_offset = (hash_offset + chunk.x) % get_max_shift();
            hash ^= ((hash_t)chunk.x) << hash_offset;
#endif
#if HASH_CHUNK_BITS >= 64
            hash_offset = (hash_offset + chunk.y) % get_max_shift();
            hash ^= ((hash_t)chunk.y) << hash_offset;
#endif
#if HASH_CHUNK_BITS == 128
            hash_offset = (hash_offset + chunk.z) % get_max_shift();
            hash ^= ((hash_t)chunk.z) << hash_offset;
            hash_offset = (hash_offset + chunk.w) % get_max_shift();
            hash ^= ((hash_t)chunk.w) << hash_offset;
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_xor_hw(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;
        uint32_t shift = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            shift += chunk;
            hash ^= (hash_t)permutate(chunk, shift);
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS >= 32
            shift += chunk.x;
            hash ^= (hash_t)permutate(chunk.x, shift);
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS >= 64
            shift += chunk.y;
            hash ^= (hash_t)permutate(chunk.y, shift);
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS == 128
            shift += chunk.z;
            hash ^= (hash_t)permutate(chunk.z, shift);
            hash = reverse(hash);
            shift += chunk.w;
            hash ^= (hash_t)permutate(chunk.w, shift);
            hash = reverse(hash);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_mult(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks;

        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash = 1 + (hash * ((hash_t)chunk + 1));
#endif
#if HASH_CHUNK_BITS >= 32
            hash = 1 + (hash * ((hash_t)chunk.x + 1));
#endif
#if HASH_CHUNK_BITS >= 64
            hash = 1 + (hash * ((hash_t)chunk.y + 1));
#endif
#if HASH_CHUNK_BITS == 128
            hash = 1 + (hash * ((hash_t)chunk.z + 1));
            hash = 1 + (hash * ((hash_t)chunk.w + 1));
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_mult_shift(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks;

        short int hash_offset = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            hash_offset = (hash_offset + chunk) % get_max_shift();
            hash = 1 + (hash * ((hash_t)chunk + 1) << hash_offset);
#endif
#if HASH_CHUNK_BITS >= 32
            hash_offset = (hash_offset + chunk.x) % get_max_shift();
            hash = 1 + (hash * ((hash_t)chunk.x + 1) << hash_offset);
#endif
#if HASH_CHUNK_BITS >= 64
            hash_offset = (hash_offset + chunk.y) % get_max_shift();
            hash = 1 + (hash * ((hash_t)chunk.y + 1) << hash_offset);
#endif
#if HASH_CHUNK_BITS == 128
            hash_offset = (hash_offset + chunk.z) % get_max_shift();
            hash = 1 + (hash * ((hash_t)chunk.z + 1) << hash_offset);
            hash_offset = (hash_offset + chunk.w) % get_max_shift();
            hash = 1 + (hash * ((hash_t)chunk.w + 1) << hash_offset);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_mult_hw(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 1;
        index_t buffer_index = element_index * element_chunks;
        int shift = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            shift += chunk;
            hash = 1 + (hash * ((hash_t)permutate(chunk, shift) + 1));
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS >= 32
            shift += chunk.x;
            hash = 1 + (hash * ((hash_t)permutate(chunk.x, shift) + 1));
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS >= 64
            shift += chunk.y;
            hash = 1 + (hash * ((hash_t)permutate(chunk.y, shift) + 1));
            hash = reverse(hash);
#endif
#if HASH_CHUNK_BITS == 128
            shift += chunk.z;
            hash = 1 + (hash * ((hash_t)permutate(chunk.z, shift) + 1));
            hash = reverse(hash);
            shift += chunk.w;
            hash = 1 + (hash * ((hash_t)permutate(chunk.w, shift) + 1));
            hash = reverse(hash);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__device__ inline void mult_xor_shift(hash_t &hash, hash_t chunk, short int &operation, short int &hash_offset) {
    hash_offset = (hash_offset + chunk) % get_max_shift();
    if (operation & 1) {
        hash = 1 + (hash * ((hash_t)(chunk + 1) << hash_offset));
    } else {
        hash = (hash ^ ((hash_t)chunk << hash_offset));
    }
    operation++;
}

__device__ inline void mult_xor_hw(hash_t &hash, hash_t chunk, short int &operation, short int &hash_offset) {
    if (operation & 1) {
        hash = 1 + (hash * ((hash_t)permutate(chunk + 1, hash_offset)));
    } else {
        hash = (hash ^ ((hash_t)permutate(chunk, hash_offset)));
    }
    operation++;
    hash_offset += chunk;
    hash = reverse(hash);
}

__global__ void hash_custom_mult_xor_shift(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;

        short int hash_offset = 0;
        short int operation = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            mult_xor_shift(hash, chunk, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 32
            mult_xor_shift(hash, chunk.x, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 64
            mult_xor_shift(hash, chunk.y, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS == 128
            mult_xor_shift(hash, chunk.z, operation, hash_offset);
            mult_xor_shift(hash, chunk.w, operation, hash_offset);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_n_mult_xor_shift(index_t element_buffer_size, short int chunk_offset, short int element_chunks, short int rounds, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;
        short int hash_offset = 0;
        short int operation = 0;
        for (short int round_index = 0; round_index < rounds; round_index++) {
            for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
                chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
                mult_xor_shift(hash, chunk, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 32
                mult_xor_shift(hash, chunk.x, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 64
                mult_xor_shift(hash, chunk.y, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS == 128
                mult_xor_shift(hash, chunk.z, operation, hash_offset);
                mult_xor_shift(hash, chunk.w, operation, hash_offset);
#endif
            }
        }
        hashed_buffer[element_index] = hash;
    }
}

__global__ void hash_custom_mult_xor_hw(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (index_t element_index = index; element_index < element_buffer_size; element_index += stride) {
        hash_t hash = 0;
        index_t buffer_index = element_index * element_chunks;

        short int hash_offset = 0;
        short int operation = 0;
        for (short int chunk_index = chunk_offset; chunk_index < element_chunks; chunk_index++) {
            chunk_t chunk = element_buffer[buffer_index + chunk_index];
#if HASH_CHUNK_BITS == 8
            mult_xor_hw(hash, chunk, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 32
            mult_xor_hw(hash, chunk.x, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS >= 64
            mult_xor_hw(hash, chunk.y, operation, hash_offset);
#endif
#if HASH_CHUNK_BITS == 128
            mult_xor_hw(hash, chunk.z, operation, hash_offset);
            mult_xor_hw(hash, chunk.w, operation, hash_offset);
#endif
        }
        hashed_buffer[element_index] = hash;
    }
}

void hash_func(index_t element_buffer_size, short int chunk_offset, short int element_chunks, chunk_t *element_buffer, hash_t *hashed_buffer, HashConfig &hash_config) {
    int threads = hash_config.threads_per_block;
    int blocks = max((element_buffer_size / hash_config.elements_per_thread) / threads, (index_t)1);

    if (hash_config.profile_enabled) {
        cudaEventRecord(hash_config.profile_start, hash_config.stream);
    }

    if (hash_config.algorithm == "fnv") {
        hash_fnv<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_xor") {
        hash_custom_xor<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_xor_shift") {
        hash_custom_xor_shift<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_xor_hw") {
        hash_custom_xor_hw<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_mult_xor_shift") {
        hash_custom_mult_xor_shift<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_n_mult_xor_shift") {
        hash_custom_n_mult_xor_shift<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, 2, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_mult_xor_hw") {
        hash_custom_mult_xor_hw<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_mult") {
        hash_custom_mult<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_mult_shift") {
        hash_custom_mult_shift<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_mult_hw") {
        hash_custom_mult_hw<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_add") {
        hash_custom_add<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_add_shift") {
        hash_custom_add_shift<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    } else if (hash_config.algorithm == "custom_add_hw") {
        hash_custom_add_hw<<<blocks, threads, 0, hash_config.stream>>>(element_buffer_size, chunk_offset, element_chunks, element_buffer, hashed_buffer);
    }

    if (hash_config.profile_enabled) {
        cudaEventRecord(hash_config.profile_end, hash_config.stream);
        float runtime_ms = 0.0;
        cudaEventSynchronize(hash_config.profile_end);
        cudaEventElapsedTime(&runtime_ms, hash_config.profile_start, hash_config.profile_end);
        float runtime_s = runtime_ms / pow(10, 3);
        hash_config.profile_summary.k_tuples_p_seconds = element_buffer_size / runtime_s;
        int element_size = (element_chunks - chunk_offset) * sizeof(chunk_t);
        hash_config.profile_summary.k_gb_p_seconds = element_size * element_buffer_size / runtime_s / pow(10, 9);
        hash_config.profile_summary.algorithm = hash_config.algorithm;
        hash_config.profile_summary.elements = element_buffer_size;
        hash_config.profile_summary.element_bytes = element_chunks * sizeof(chunk_t);
        hash_config.profile_summary.element_offset = chunk_offset * sizeof(chunk_t);
    }
}