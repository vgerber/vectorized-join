#pragma once

#include <iostream>
#include <math.h>
#include <string.h>
#include <cassert>
#include "config.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
struct HashTable {
    index_t * table = nullptr;
    index_t * links = nullptr;

    int slots = 0;
    index_t size;

    HashTable(int slots, index_t size) {
        this->slots = slots;
        this->size = size;
    }

    void print() {
        for(int slot_index = 0; slot_index < slots; slot_index++) {
            index_t entry_index = table[slot_index];
            if(entry_index) {
                std::cout << "[#S=" << slot_index << " #H=" << entry_index  << " ";
                while (entry_index) {
                    index_t link_value = links[entry_index-1];
                    std::cout << link_value << ",";
                    entry_index = link_value;
                }
                std::cout << "] ";
            }
        }
        std::cout << std::endl;
    }
};

hash_t get_radix_mask(int bins) {
    return bins - 1;
}

__global__
void histogram_kernel(index_t buffer_size, hash_t * hash_buffer, int bins, index_t *histogram, int radix_shift, hash_t radix_mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int element_index = index; element_index < buffer_size; element_index += stride) {
        hash_t key = (hash_buffer[element_index] >> radix_shift) & radix_mask;
        atomicAdd(&histogram[key], 1);
    }
}

__global__
void element_swap_kernel(db_entry entry, db_entry entry_swap, index_t * offsets, index_t * dest_indices, int radix_shift, hash_t radix_mask, bool index_data=false) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < entry.size; element_index += stride) {
        // determine bucket by hash
        hash_t element = entry.hashes[element_index];
        hash_t key = (element >> radix_shift) & radix_mask;

        // place index in new bucket position
        index_t swap_index = offsets[key] + atomicAdd(&dest_indices[key], 1);
        entry_swap.hashes[swap_index] = entry.hashes[element_index];
        if(index_data) {
            entry_swap.indices[swap_index] = element_index;
        } else {
            entry_swap.indices[swap_index] = entry.indices[element_index];  
        }
    }
}

__global__
void create_hash_table_kernel(hash_t * hash_buffer, HashTable table, int key_offset) {
    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;
    for(index_t element_index = index; element_index < table.size; element_index += stride) {
        hash_t element = hash_buffer[element_index];
        int slot = (element >> key_offset) % table.slots;
        table.links[element_index] = atomicExch(&table.table[slot], element_index+1);
    }
}

__global__
void probe_hash_tabel_kernel(index_t r_buffer_size, hash_t * r_hash_buffer, index_t s_hash_buffer_size, hash_t * s_hash_buffer, HashTable table, bool * probe_results, int key_offset) {
    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;
    for(index_t s_index = index; s_index < s_hash_buffer_size; s_index += stride) {
        memset(&probe_results[table.size * s_index], 0, table.size * sizeof(bool));
        hash_t s_hash = s_hash_buffer[s_index];
        hash_t probe_key = s_hash >> key_offset;

        index_t table_index = table.table[probe_key % table.slots];        
        while (table_index)
        {
            table_index--;
            const hash_t r_hash = r_hash_buffer[table_index];
            bool probe_result = (s_hash == r_hash);
            probe_results[s_index * table.size + table_index] = probe_result;
            table_index = table.links[table_index];
        }
    }
}

__global__
void build_and_probe_kernel(index_t r_hash_buffer_size, hash_t * r_hash_buffer, index_t s_hash_buffer_size, hash_t * s_hash_buffer, bool * probe_results, int key_offset, int slots) {
    
    // hash table
    extern __shared__ hash_t table[];
    hash_t * hash_buffer = &table[0];
    index_t * table_links = (index_t*)&table[r_hash_buffer_size];
    index_t * table_slots = (index_t*)&table_links[r_hash_buffer_size];

    if(threadIdx.x == 0) {
        memset(table_slots, 0, slots * sizeof(index_t));
    }

    __syncthreads();
    
    // build phase
    // build hash table from hasehs in table r
    index_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = gridDim.x * blockDim.x;
    for(index_t r_index = index; r_index < r_hash_buffer_size; r_index += stride) {
        hash_t r_hash = r_hash_buffer[r_index];
        hash_t r_key = r_hash >> key_offset;
        int slot = r_key % slots;
        table_links[r_index] = atomicExch(&table_slots[slot], r_index+1);
        hash_buffer[r_index] = r_hash;
    }

    __syncthreads();

    // probing
    // probe values from table s on hash table with values from r
    for(index_t s_index = index; s_index < s_hash_buffer_size; s_index += stride) {
        memset(&probe_results[r_hash_buffer_size * s_index], 0, r_hash_buffer_size * sizeof(bool));
        hash_t s_hash = s_hash_buffer[s_index];
        hash_t s_key = s_hash >> key_offset;
        index_t table_index = table_slots[s_key % slots];        
        while (table_index)
        {
            // compare hash of table r (build) and s (probe)
            // indices in link table have an offset by one (linke value of 0 -> last node)
            table_index--;
            const hash_t r_hash = hash_buffer[table_index];
            bool probe_result = (s_hash == r_hash);
            probe_results[s_index * r_hash_buffer_size + table_index] = probe_result;
            table_index = table_links[table_index];
        }
    }
}

void partition_gpu(db_entry d_entry, db_entry d_entry_swap, int radix_width, int radix_shift, int bins, index_t * histogram, index_t * offsets, bool index_data, cudaStream_t stream) {
    assert(d_entry.size == d_entry_swap.size);

    hash_t radix_mask = get_radix_mask(bins);

    index_t * d_histogram = nullptr;
    gpuErrchk(cudaMallocAsync(&d_histogram, bins * sizeof(index_t), stream));
    gpuErrchk(cudaMemsetAsync(d_histogram, 0, bins * sizeof(index_t), stream));
    histogram_kernel<<<256, 512, 0, stream>>>(d_entry.size, d_entry.hashes, (index_t)bins, d_histogram, radix_shift, radix_mask);
    gpuErrchk(cudaMemcpyAsync(histogram, d_histogram, bins * sizeof(index_t), cudaMemcpyDeviceToHost, stream));
    

    // calculate offsets
    index_t * d_dest_indices = nullptr;
    index_t * d_offsets = nullptr;
    memset(offsets, 0, bins * sizeof(int));
    cudaMallocAsync(&d_dest_indices, bins * sizeof(index_t), stream);
    cudaMemsetAsync(d_dest_indices, 0, bins * sizeof(index_t), stream);    
    cudaMallocAsync(&d_offsets, bins * sizeof(index_t), stream);
    index_t offset = 0;
    cudaStreamSynchronize(stream);
    for(int bin_index = 0; bin_index < bins; bin_index++) {
        offsets[bin_index] = offset;
        offset += histogram[bin_index];
    }
    gpuErrchk(cudaMemcpyAsync(d_offsets, offsets, bins * sizeof(index_t), cudaMemcpyHostToDevice, stream));


    // swap elments according to bin key
    element_swap_kernel<<<256, 512, 0, stream>>>(d_entry, d_entry_swap, d_offsets, d_dest_indices, radix_shift, radix_mask, index_data);
    
    cudaFreeAsync(d_histogram, stream);
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_dest_indices, stream);
}

void partition_cpu(db_entry entry, db_entry entry_swap, int radix_width, int radix_shift, index_t * histogram, index_t * offsets, int bins) {
    assert(entry.size == entry_swap.size);

    // cpu    
    bins = (1 << radix_width);
    int radix_mask = (bins - 1);


    // gpu
    memset(histogram, 0, bins * sizeof(index_t));
    for(int element_index = 0; element_index < entry.size; element_index++) {
        int key = (entry.hashes[element_index] >> radix_shift) & radix_mask;
        histogram[key]++;
    }

    // cpu
    int offset = 0;
    index_t * dest_indices = new index_t[bins];
    memset(dest_indices, 0, bins * sizeof(index_t));
    for(int bin_index = 0; bin_index < bins; bin_index++) {
        offsets[bin_index] = offset;
        offset += histogram[bin_index];
    }

    // gpu
    for(int element_index = 0; element_index < entry.size; element_index++) {
        hash_t element = entry.hashes[element_index];
        int key = (element >> radix_shift) & radix_mask;
        index_t offset = offsets[key];
        index_t dest_index = dest_indices[key]++;
        entry_swap.hashes[offset + dest_index] = element;     
        entry_swap.indices[offset + dest_index] = entry.indices[element_index];   
    }

    delete[] dest_indices;
}

void partition(db_entry entry, db_entry entry_swap, int radix_width, int radix_shift, int bins, index_t * histogram, index_t * offsets, bool index_data, bool gpu=true) {
    assert(entry.size == entry_swap.size);
    if(gpu) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        db_entry d_entry, d_entry_swap;
        d_entry.size = entry.size;
        d_entry_swap.size = entry_swap.size;
        gpuErrchk(cudaMallocAsync(&d_entry.hashes, entry.size * sizeof(hash_t), stream));
        gpuErrchk(cudaMallocAsync(&d_entry.indices, entry.size * sizeof(hash_t), stream));
        gpuErrchk(cudaMallocAsync(&d_entry_swap.hashes, entry.size * sizeof(hash_t), stream));
        gpuErrchk(cudaMallocAsync(&d_entry_swap.indices, entry.size * sizeof(index_t), stream));

        gpuErrchk(cudaMemcpyAsync(d_entry.hashes, entry.hashes, entry.size * sizeof(hash_t), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(d_entry.indices, entry.indices, entry.size * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        partition_gpu(d_entry, d_entry_swap, radix_width, radix_shift, bins, histogram, offsets, index_data,  stream);
        gpuErrchk(cudaMemcpyAsync(entry_swap.hashes, d_entry_swap.hashes, entry.size * sizeof(hash_t), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(entry_swap.indices, d_entry_swap.indices, entry.size * sizeof(index_t), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        gpuErrchk(cudaGetLastError());

        cudaStreamDestroy(stream);
        cudaFree(d_entry.hashes);
        cudaFree(d_entry.indices);
        cudaFree(d_entry_swap.hashes);
        cudaFree(d_entry_swap.indices);
    } else {
        partition_cpu(entry, entry_swap, radix_width, radix_shift, histogram, offsets, bins);
    }
}

void create_hash_table_cpu(index_t buffer_size, hash_t * hash_buffer, HashTable table, int key_offset) {
    memset(table.links, 0, buffer_size * sizeof(index_t));
    memset(table.table, 0, table.slots * sizeof(index_t));

    for(int buffer_index = 0; buffer_index < buffer_size; buffer_index++) {
        hash_t element = hash_buffer[buffer_index];
        int slot = (element >> key_offset) % table.slots;
        table.links[buffer_index] = table.table[slot];
        table.table[slot] = buffer_index+1;
    }
}

void create_hash_table_gpu(index_t buffer_size, hash_t * d_hash_buffer, HashTable & d_table, int key_offset, cudaStream_t & stream) {
    create_hash_table_kernel<<<1, 256, 0, stream>>>(d_hash_buffer, d_table, key_offset);
}

void create_hash_table(index_t buffer_size, hash_t * hash_buffer, HashTable table, int key_offset, bool gpu=true) {
    if(gpu) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        HashTable d_table(table.slots, table.size);
        hash_t * d_hash_buffer = nullptr;
        cudaMallocAsync(&d_table.links, d_table.size * sizeof(index_t), stream);
        cudaMallocAsync(&d_table.table, table.slots * sizeof(index_t), stream);
        cudaMemsetAsync(d_table.table, 0, table.slots * sizeof(index_t), stream);
        cudaMallocAsync(&d_hash_buffer, buffer_size * sizeof(hash_t), stream);
        cudaMemcpyAsync(d_hash_buffer, hash_buffer, buffer_size * sizeof(hash_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(table.table, d_table.table, table.slots * sizeof(index_t), cudaMemcpyDeviceToHost, stream);

        create_hash_table_gpu(buffer_size, d_hash_buffer, d_table, key_offset, stream);

        cudaMemcpyAsync(table.links, d_table.links, d_table.size * sizeof(index_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(table.table, d_table.table, d_table.slots * sizeof(index_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_table.links);
        cudaFree(d_table.table);
        cudaFree(d_hash_buffer);
    } else {
        create_hash_table_cpu(buffer_size, hash_buffer, table, key_offset);
    }
}

void probe_hash_table_cpu(HashTable table, index_t r_buffer_size, hash_t * r_hash_buffer, index_t s_buffer_size, hash_t * s_hash_buffer, bool * probe_results, int key_offset) {
    for(index_t s_index = 0; s_index < s_buffer_size; s_index++) {
        memset(&probe_results[r_buffer_size * s_index], 0, r_buffer_size * sizeof(filter_mask));
        hash_t s_hash = s_hash_buffer[s_index];
        hash_t s_key = s_hash >> key_offset;

        index_t table_index = table.table[s_key % table.slots];
        

        while (table_index)
        {
            table_index--;
            const hash_t r_hash = r_hash_buffer[table_index];
            bool probe_result = (s_hash == r_hash);
            probe_results[s_index * r_buffer_size + table_index] = probe_result;
            table_index = table.links[table_index];
        }        
        
    }
}

void probe_hash_table_gpu(const HashTable d_table, index_t r_buffer_size, hash_t * r_hash_buffer, index_t s_buffer_size, hash_t * d_s_hash_buffer, bool * d_probe_results, int key_offset, cudaStream_t &stream) {
    probe_hash_tabel_kernel<<<256, 256, 0, stream>>>(r_buffer_size, r_hash_buffer, s_buffer_size, d_s_hash_buffer, d_table, d_probe_results, key_offset);
}


void probe_hash_table(HashTable table, index_t r_buffer_size, hash_t * r_hash_buffer, index_t s_buffer_size, hash_t * s_hash_buffer, filter_mask * probe_results, int key_offset, bool gpu=true) {
    if(gpu) {
        hash_t * d_r_hash_buffer = nullptr;
        hash_t * d_s_hash_buffer = nullptr;
        filter_mask * d_probe_results = nullptr;
        HashTable d_table(table.slots, table.size);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMallocAsync(&d_table.table, d_table.slots * sizeof(index_t), stream);
        cudaMemcpyAsync(d_table.table, table.table, d_table.slots * sizeof(index_t), cudaMemcpyHostToDevice, stream);
        cudaMallocAsync(&d_table.links, d_table.size * sizeof(index_t), stream);
        cudaMemcpyAsync(d_table.links, table.links, d_table.size * sizeof(index_t), cudaMemcpyHostToDevice, stream);
        cudaMallocAsync(&d_s_hash_buffer, s_buffer_size * sizeof(hash_t), stream);
        cudaMemcpyAsync(d_s_hash_buffer, s_hash_buffer, s_buffer_size * sizeof(hash_t), cudaMemcpyHostToDevice, stream);
        cudaMallocAsync(&d_r_hash_buffer, r_buffer_size * sizeof(hash_t), stream);
        cudaMemcpyAsync(d_r_hash_buffer, r_hash_buffer, r_buffer_size * sizeof(hash_t), cudaMemcpyHostToDevice, stream);
        cudaMallocAsync(&d_probe_results, d_table.size * s_buffer_size * sizeof(filter_mask), stream);
        probe_hash_table_gpu(d_table, r_buffer_size, d_r_hash_buffer, s_buffer_size, d_s_hash_buffer, d_probe_results, key_offset, stream);
        cudaMemcpyAsync(probe_results, d_probe_results, d_table.size * s_buffer_size * sizeof(filter_mask), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        cudaFree(d_s_hash_buffer);
        cudaFree(d_probe_results);
        cudaFree(d_table.table);
        cudaFree(d_table.links);
        cudaStreamDestroy(stream);
    } else {
        probe_hash_table_cpu(table, r_buffer_size, r_hash_buffer, s_buffer_size, s_hash_buffer, probe_results, key_offset);
    }
}

void build_and_probe_gpu(index_t buffer_size, hash_t * r_hash_buffer, index_t s_buffer_size, hash_t * s_hash_buffer, filter_mask * probe_results, int key_offset, int slots, cudaStream_t stream) {
    int shared_mem = buffer_size * sizeof(hash_t) + buffer_size * sizeof(index_t) + slots * sizeof(index_t);
    build_and_probe_kernel<<<1, 1024, shared_mem, stream>>>(buffer_size, r_hash_buffer, s_buffer_size, s_hash_buffer, probe_results, key_offset, slots);
}

void build_and_probe(index_t r_buffer_size, hash_t * r_hash_buffer, index_t s_buffer_size, hash_t * s_hash_buffer, filter_mask * probe_results, int key_offset, int slots, bool gpu=true) {
    if(gpu) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        hash_t * d_r_hash_buffer = nullptr;
        hash_t * d_s_hash_buffer = nullptr;
        filter_mask * d_probe_results = nullptr;
        cudaMallocAsync(&d_r_hash_buffer, r_buffer_size * sizeof(hash_t), stream);
        cudaMemcpyAsync(d_r_hash_buffer, r_hash_buffer, r_buffer_size * sizeof(hash_t), cudaMemcpyHostToDevice, stream);
        cudaMallocAsync(&d_s_hash_buffer, s_buffer_size * sizeof(hash_t), stream);
        cudaMemcpyAsync(d_s_hash_buffer, s_hash_buffer, s_buffer_size * sizeof(hash_t), cudaMemcpyHostToDevice, stream);
        gpuErrchk(cudaMallocAsync(&d_probe_results, r_buffer_size * s_buffer_size * sizeof(filter_mask), stream));
        build_and_probe_gpu(r_buffer_size, d_r_hash_buffer, s_buffer_size, d_s_hash_buffer, d_probe_results, key_offset, slots, stream);
        gpuErrchk(cudaMemcpyAsync(probe_results, d_probe_results, r_buffer_size * s_buffer_size * sizeof(filter_mask), cudaMemcpyDeviceToHost, stream));
        
        cudaFreeAsync(d_r_hash_buffer, stream);
        cudaFreeAsync(d_s_hash_buffer, stream);
        cudaFreeAsync(d_probe_results, stream);
        cudaStreamSynchronize(stream);        
    } else {
        HashTable table(100, r_buffer_size);
        table.table = new index_t[table.slots];
        table.links = new index_t[table.size];
        create_hash_table(r_buffer_size, r_hash_buffer, table, key_offset, false);
        table.print();
        probe_hash_table_cpu(table, r_buffer_size, r_hash_buffer, s_buffer_size, s_hash_buffer, probe_results, key_offset);

        delete[] table.table;
        delete[] table.links;
    }
}