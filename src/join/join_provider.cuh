#pragma once
#include <vector>
#include "partition.cu"

struct BucketConfig {
    int buckets = 0;
    int bucket_depth = 0;
    BucketConfig * sub_buckets = nullptr;
    index_t * histogram = nullptr;
    index_t * offsets = nullptr;
    db_entry data;
    HashTable * table = nullptr;

    ~BucketConfig() {
        delete[] histogram;
        delete[] offsets;
        if(sub_buckets) {
            for(int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                sub_buckets[bucket_index].~BucketConfig();
            }  
        }
    }

    void print() {
        std::cout << std::string(bucket_depth, '-') << data.size << std::endl;
        if(sub_buckets) {
            for(int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                sub_buckets[bucket_index].print();
            }  
        }
    }

    void print_compare(const BucketConfig * compare_config) {
        std::cout << std::string(bucket_depth, '-') << data.size << ":" << compare_config->data.size << std::endl;
        if(sub_buckets) {
            for(int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                sub_buckets[bucket_index].print_compare(&compare_config->sub_buckets[bucket_index]);
            }  
        }
    }
};

struct DeviceConfig {
    int device_id = 0;
    cudaStream_t stream;
};

class JoinProvider {

public:
    void join_cpu(db_entry r_table, db_entry s_table) {

        int buckets = 2;
        int radix_width = std::ceil(std::log2(buckets));
        int bins = (1 << radix_width);
        
        int max_radix_steps = (8 * sizeof(index_t)) / radix_width;
        int max_bucket_elements = 1000 / sizeof(db_entry);

        db_entry r_table_swap;
        r_table_swap.hashes = new hash_t[r_table.size];
        r_table_swap.indices = new index_t[r_table.size];
        r_table_swap.size = r_table.size;
        db_entry s_table_swap;
        s_table_swap.hashes = new hash_t[s_table.size];
        s_table_swap.indices = new index_t[s_table.size];
        s_table_swap.size = s_table.size;

        
        BucketConfig r_bucket_config;
        r_bucket_config.buckets = bins;
        r_bucket_config.data = r_table;
        BucketConfig s_bucket_config;
        s_bucket_config.buckets = bins;
        s_bucket_config.data = s_table;

        devices.push_back(DeviceConfig());
        partition_r_recursive(r_table_swap, radix_width, &r_bucket_config, max_bucket_elements, max_radix_steps, devices[0]);
        partition_s_recursive(s_table_swap, radix_width, &s_bucket_config, &r_bucket_config, max_radix_steps, devices[0]);
        r_bucket_config.print_compare(&s_bucket_config);


        probe_recursive(&r_bucket_config, &s_bucket_config, radix_width, max_bucket_elements, devices[0], true);
        

        delete[] r_table_swap.hashes;
        delete[] r_table_swap.indices;
        delete[] s_table_swap.hashes;
        delete[] s_table_swap.indices;
    } 


    void join_gpu(db_entry d_r_table, db_entry d_s_table) {
        configure_devices();

        int buckets = 2;
        int radix_width = std::ceil(std::log2(buckets));
        int bins = (1 << radix_width);
        
        int max_radix_steps = (8 * sizeof(index_t)) / radix_width;
        int max_bucket_elements = 1000 / sizeof(db_entry); 

        db_entry d_r_table_swap;
        d_r_table_swap.size = d_r_table.size;
        cudaMalloc(&d_r_table_swap.hashes, d_r_table.size * sizeof(hash_t));
        cudaMalloc(&d_r_table_swap.indices, d_r_table.size * sizeof(index_t));

        db_entry d_s_table_swap;
        d_s_table_swap.size = d_s_table.size;
        cudaMalloc(&d_s_table_swap.hashes, d_s_table.size * sizeof(hash_t));
        cudaMalloc(&d_s_table_swap.indices, d_s_table.size * sizeof(index_t));

        
        BucketConfig r_bucket_config;
        r_bucket_config.buckets = bins;
        r_bucket_config.data = d_r_table;
        BucketConfig s_bucket_config;
        s_bucket_config.buckets = bins;
        s_bucket_config.data = d_s_table;

        partition_r_recursive(d_r_table_swap, radix_width, &r_bucket_config, max_bucket_elements, max_radix_steps, devices[0], true);
        partition_s_recursive(d_s_table_swap, radix_width, &s_bucket_config, &r_bucket_config, max_radix_steps, devices[0], true);
        r_bucket_config.print_compare(&s_bucket_config);


        probe_recursive(&r_bucket_config, &s_bucket_config, radix_width, max_bucket_elements, devices[0], true);
        
        cudaFree(d_r_table_swap.hashes);
        cudaFree(d_r_table_swap.indices);
        cudaFree(d_s_table_swap.hashes);
        cudaFree(d_s_table_swap.indices);
    } 

private: 

    std::vector<DeviceConfig> devices;

    void configure_devices() {
        DeviceConfig config;
        config.device_id = 0;
        cudaStreamCreate(&config.stream);
        devices.push_back(config); 
    }

    void partition_r_recursive(db_entry r_swap_entry, int radix_width, BucketConfig * config, index_t max_elements, index_t max_radix_steps, const DeviceConfig device_config, bool gpu=true) {
        if(config->data.size > 0 && config->data.size > max_elements && config->bucket_depth <= max_radix_steps) {
            int radix_shift = (radix_width * config->bucket_depth);
            int buckets = config->buckets;
            bool index_data = config->bucket_depth == 0;

            config->histogram = new index_t[buckets];
            config->offsets = new index_t[buckets];
            config->sub_buckets = new BucketConfig[config->buckets];

            if(gpu) {
                partition_gpu(config->data, r_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, device_config.stream);
            } else {                
                partition(config->data, r_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, gpu);
            }

            for(int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                index_t sub_elements = config->histogram[bucket_index];
                index_t sub_offset = config->offsets[bucket_index];
                

                BucketConfig * sub_bucket = &config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket and start partitioning on reduced data set
                db_entry entry_bucket;
                entry_bucket.size = sub_elements;
                entry_bucket.hashes = &r_swap_entry.hashes[sub_offset];
                entry_bucket.indices = &r_swap_entry.indices[sub_offset];
                sub_bucket->data = entry_bucket;

                db_entry swap_entry;
                swap_entry.size = sub_elements;
                swap_entry.hashes = &config->data.hashes[sub_offset];
                swap_entry.indices = &config->data.indices[sub_offset];
                partition_r_recursive(swap_entry, radix_width, sub_bucket, max_elements, max_radix_steps, device_config, gpu);
            } 
        }
    }

    void partition_s_recursive(db_entry s_swap_entry, int radix_width, BucketConfig * config, const BucketConfig * r_config, index_t max_radix_steps, DeviceConfig device_config, bool gpu=true) {
        
        if(r_config->data.size > 0 && r_config->sub_buckets && r_config->data.size > 0) {
            int radix_shift = (radix_width * config->bucket_depth);
            int buckets = config->buckets;
            bool index_data = config->bucket_depth == 0;
            
            config->histogram = new index_t[buckets];
            config->offsets = new index_t[buckets];
            config->sub_buckets = new BucketConfig[config->buckets];
            if(gpu) {
                partition_gpu(config->data, s_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, device_config.stream);
            } else {
                partition(config->data, s_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, gpu);
            }

            for(int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                index_t sub_elements = config->histogram[bucket_index];
                index_t sub_offset = config->offsets[bucket_index];
                
                // configure bucket
                BucketConfig * sub_bucket = &config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket
                db_entry entry_bucket;
                entry_bucket.size = sub_elements;
                entry_bucket.hashes = &s_swap_entry.hashes[sub_offset];
                entry_bucket.indices = &s_swap_entry.indices[sub_offset];
                sub_bucket->data = entry_bucket;

                // set data for swap
                db_entry swap_entry;
                swap_entry.size = sub_elements;
                swap_entry.hashes = &config->data.hashes[sub_offset];
                swap_entry.indices = &config->data.indices[sub_offset];

                partition_s_recursive(swap_entry, radix_width, sub_bucket, &r_config->sub_buckets[bucket_index], max_radix_steps, device_config);
            } 
        }
    }

    void probe_recursive(const BucketConfig * r_config, const BucketConfig * s_config, int radix_width, index_t max_build_elements, DeviceConfig device_config, bool gpu) {
        if(r_config->sub_buckets) {
            for(int bucket_index = 0; bucket_index < r_config->buckets; bucket_index++) {
                probe_recursive(&r_config->sub_buckets[bucket_index], &s_config->sub_buckets[bucket_index], radix_width, max_build_elements, device_config, gpu);
            }
        } else {
            if(r_config->data.size && s_config->data.size) {
                index_t r_data_size = r_config->data.size;
                for(int chunk_offset = 0; chunk_offset < r_config->data.size; chunk_offset += max_build_elements) {
                    // calc chunk size
                    index_t build_data_size = std::min(r_data_size, max_build_elements);
                    r_data_size -= build_data_size;

                    // offset for table key
                    int key_offset = (radix_width * r_config->bucket_depth);
                    /*std::cout << "r" << std::endl;
                    r_config->data.print();
                    std::cout << "s" << std::endl;
                    s_config->data.print();*/

                    bool multi_step = false;

                    // run probe and build
                    filter_mask * probe_results = nullptr;
                    if(gpu) {
                        cudaMallocAsync(&probe_results, s_config->data.size * build_data_size * sizeof(filter_mask), device_config.stream);
                    } else {
                        probe_results = new filter_mask[s_config->data.size * build_data_size];
                    }

                    if(multi_step) {
                        HashTable table(100, r_config->data.size);
                        table.links = new index_t[table.size];
                        table.table = new index_t[table.slots];
                        create_hash_table(r_config->data.size, r_config->data.hashes, table, key_offset, true);
                        table.print();
                        probe_hash_table(table, r_config->data.size, r_config->data.hashes, s_config->data.size, s_config->data.hashes, probe_results, key_offset);
                    } else {
                        if(gpu) {
                            build_and_probe_gpu(build_data_size, &r_config->data.hashes[chunk_offset], s_config->data.size, s_config->data.hashes, probe_results, key_offset, r_config->data.size, device_config.stream);
                            cudaStreamSynchronize(device_config.stream);
                            db_entry h_r_data;
                            h_r_data.size = r_config->data.size;
                            h_r_data.hashes = new hash_t[h_r_data.size];
                            h_r_data.indices = new index_t[h_r_data.size];
                            cudaMemcpy(h_r_data.hashes, r_config->data.hashes, h_r_data.size * sizeof(hash_t), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_r_data.indices, r_config->data.indices, h_r_data.size * sizeof(index_t), cudaMemcpyDeviceToHost);

                            db_entry h_s_data;
                            h_s_data.size = s_config->data.size;
                            h_s_data.hashes = new hash_t[h_s_data.size];
                            h_s_data.indices = new index_t[h_s_data.size];
                            cudaMemcpy(h_s_data.hashes, s_config->data.hashes, h_s_data.size * sizeof(hash_t), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_s_data.indices, s_config->data.indices, h_s_data.size * sizeof(index_t), cudaMemcpyDeviceToHost);
                            
                            filter_mask * h_probe_results = new filter_mask[h_r_data.size * h_s_data.size];
                            cudaMemcpy(h_probe_results, probe_results, h_s_data.size * build_data_size * sizeof(filter_mask), cudaMemcpyDeviceToHost);
                            print_probe_results(h_r_data, h_s_data, h_probe_results);
                        } else {
                            build_and_probe(build_data_size, &r_config->data.hashes[chunk_offset], s_config->data.size, s_config->data.hashes, probe_results, key_offset, r_config->data.size, gpu);
                            print_probe_results(r_config->data, s_config->data, probe_results);
                        }
                    }
                    
                    if(gpu) {
                        cudaFree(probe_results);
                    } else {
                        delete[] probe_results;
                    }
                }
            }
        }

    }

    void print_probe_results(const db_entry r_entry, const db_entry s_entry, const filter_mask * probe_results) {
        for(int s_index = 0; s_index < std::min(s_entry.size, (index_t)5); s_index++) {
            int probe_hits = 0;

            std::cout << "S=" << s_entry.hashes[s_index] << std::endl;
            for(int r_index = 0; r_index < r_entry.size; r_index++) {
                if(probe_results[s_index * r_entry.size + r_index]) {
                    std::cout << "\t" << r_entry.hashes[r_index] << "@" << r_entry.indices[r_index] << std::endl;
                }
                probe_hits += probe_results[s_index * r_entry.size + r_index] ? 1 : 0;
            }
        }
    }
};