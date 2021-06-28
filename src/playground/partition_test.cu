#include "join/join_provider.cuh"
#include "benchmark/data_generator.hpp"


int main(int argc, char** argv) {

    bool gpu = true;
    
    db_entry r_table;
    r_table.size = 1000;
    if(gpu) {
        gpuErrchk(cudaMalloc(&r_table.hashes, r_table.size * sizeof(hash_t)));
        gpuErrchk(cudaMalloc(&r_table.indices, r_table.size * sizeof(index_t)));
    } else {
        r_table.hashes = new hash_t[r_table.size];
        r_table.indices = new index_t[r_table.size];
    }

    db_entry s_table;
    s_table.size = r_table.size * 2;
    if(gpu) {
        gpuErrchk(cudaMalloc(&s_table.hashes, s_table.size * sizeof(hash_t)));
        gpuErrchk(cudaMalloc(&s_table.indices, s_table.size * sizeof(index_t)));
    } else {
        s_table.hashes = new hash_t[s_table.size];
        s_table.indices = new index_t[s_table.size];
    }

    if(gpu) {
        db_entry h_r_table;
        h_r_table.size = r_table.size;
        h_r_table.hashes = new hash_t[h_r_table.size];
        generate_random_numbers(h_r_table.size, (filter_int)1000, h_r_table.hashes);
        cudaMemcpy(r_table.hashes, h_r_table.hashes, h_r_table.size * sizeof(hash_t), cudaMemcpyHostToDevice);
        delete[] h_r_table.hashes;

        db_entry h_s_table;
        h_s_table.size = s_table.size;
        h_s_table.hashes = new hash_t[h_s_table.size];
        generate_random_numbers(h_s_table.size, (filter_int)1000, h_s_table.hashes);
        cudaMemcpy(s_table.hashes, h_s_table.hashes, h_s_table.size * sizeof(hash_t), cudaMemcpyHostToDevice);
        delete[] h_s_table.hashes;
    } else {
        generate_random_numbers(r_table.size, (filter_int)1000, r_table.hashes);
        generate_random_numbers(s_table.size, (filter_int)1000, s_table.hashes);
    }

    


    JoinProvider join_provider;
    if(gpu) {
        join_provider.join_gpu(r_table, s_table);
    } else {
        join_provider.join_cpu(r_table, s_table);
    }

    if(!gpu) {
        delete[] r_table.hashes;
        delete[] r_table.indices;
        delete[] s_table.hashes;
        delete[] s_table.indices;
    }
    return 0;
}