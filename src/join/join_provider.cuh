#pragma once
#include <vector>
#include "join_provider.cu"

struct BucketConfig
{
    int buckets = 0;
    int bucket_depth = 0;
    BucketConfig *sub_buckets = nullptr;
    index_t *histogram = nullptr;
    index_t *offsets = nullptr;
    db_hash_table hash_table;
    db_table table;

    BucketConfig()
    {
    }

    ~BucketConfig()
    {
        delete[] histogram;
        delete[] offsets;
        if (sub_buckets)
        {
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                sub_buckets[bucket_index].~BucketConfig();
            }
        }
    }

    void print()
    {
        std::cout << std::string(bucket_depth, '-') << hash_table.size << std::endl;
        if (sub_buckets)
        {
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                sub_buckets[bucket_index].print();
            }
        }
    }

    void print_compare(const BucketConfig *compare_config)
    {
        std::cout << std::string(bucket_depth, '-') << hash_table.size << ":" << compare_config->hash_table.size << std::endl;
        if (sub_buckets)
        {
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                sub_buckets[bucket_index].print_compare(&compare_config->sub_buckets[bucket_index]);
            }
        }
    }
};

struct DeviceConfig
{
    int device_id = 0;
    cudaStream_t stream;

    void free() {
        cudaStreamDestroy(stream);
    }
};

typedef std::pair<BucketConfig *, BucketConfig *> bucket_pair_t;

class JoinProvider
{

public:
    void join(db_table r_table, db_table s_table, db_table &joined_rs_table)
    {
        auto join_start = std::chrono::high_resolution_clock::now();

        assert(r_table.gpu == s_table.gpu);
        assert(r_table.column_count == s_table.column_count);
        std::cout << "Join " << std::endl;

        configure_devices();

        int buckets = 2;
        int radix_width = std::ceil(std::log2(buckets));
        int bins = (1 << radix_width);

        int max_radix_steps = (8 * sizeof(index_t)) / radix_width;
        int max_bucket_elements = 200; // 64000 / (sizeof(index_t) + sizeof(index_t) + sizeof(hash_t));

        db_hash_table r_hash_table;
        db_hash_table r_hash_table_swap(r_table.size, r_table.gpu);
        db_table r_table_swap(r_table.column_count, r_table.size, r_table.gpu);

        db_hash_table s_hash_table;
        db_hash_table s_hash_table_swap(s_table.size, s_table.gpu);
        db_table s_table_swap(s_table.column_count, s_table.size, s_table.gpu);

        auto hash_start = std::chrono::high_resolution_clock::now();
        hash_table(r_table, r_hash_table);
        hash_table(s_table, s_hash_table);
        auto hash_end = std::chrono::high_resolution_clock::now();

        BucketConfig r_bucket_config;
        r_bucket_config.buckets = bins;
        r_bucket_config.table = r_table;
        r_bucket_config.hash_table = r_hash_table;

        BucketConfig s_bucket_config;
        s_bucket_config.buckets = bins;
        s_bucket_config.table = s_table;
        s_bucket_config.hash_table = s_hash_table;

        std::vector<bucket_pair_t> bucket_pairs;

        auto partition_start = std::chrono::high_resolution_clock::now();
        partition_r_recursive(r_table_swap, r_hash_table_swap, radix_width, &r_bucket_config, max_bucket_elements, max_radix_steps, devices[0], true);
        partition_s_recursive(s_table_swap, s_hash_table_swap, radix_width, &s_bucket_config, &r_bucket_config, max_radix_steps, devices[0], bucket_pairs, true);
        auto partition_end = std::chrono::high_resolution_clock::now();

#if DEBUG_PRINT
        r_bucket_config.print_compare(&s_bucket_config);
#endif

        auto probe_start = std::chrono::high_resolution_clock::now();
        std::vector<db_table> joined_rs_tables;
        probe(bucket_pairs, radix_width, max_bucket_elements, devices[0], joined_rs_tables);
        auto probe_end = std::chrono::high_resolution_clock::now();
        
        for(auto device : devices) {
            cudaStreamSynchronize(device.stream);
        }
        
        r_hash_table.free();
        r_hash_table_swap.free();
        r_table_swap.free();

        s_hash_table.free();
        s_hash_table_swap.free();
        s_table_swap.free();

        auto merge_start = std::chrono::high_resolution_clock::now();
        merge_joined_tables(joined_rs_tables, joined_rs_table);
        auto merge_end = std::chrono::high_resolution_clock::now();

        for(auto device : devices) {
            device.free();
        }

        auto join_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> join_druation = (join_end - join_start);
        std::chrono::duration<double> hash_druation = (hash_end - hash_start);
        std::chrono::duration<double> partition_druation = (partition_end - partition_start);
        std::chrono::duration<double> probe_druation = (probe_end - probe_start);
        std::chrono::duration<double> merge_druation = (merge_end - merge_start);

        std::cout << "Hash      " << ((r_table.size + s_table.size) / hash_druation.count()) << " Tuples/s" << std::endl;
        std::cout << "Partition " << ((r_table.size + s_table.size) / partition_druation.count()) << " Tuples/s" << std::endl;
        std::cout << "Probe     " << ((r_table.size + s_table.size) / probe_druation.count()) << " Tuples/s" << std::endl;
        std::cout << "Merge     " << ((r_table.size + s_table.size) / merge_druation.count()) << " Tuples/s" << std::endl;
        std::cout << joined_rs_table.size << " Joins " << ((r_table.size + s_table.size) / join_druation.count()) << " Tuples/s" <<std::endl;

        std::cout << "Bucket Hist:" << std::endl;
        std::vector<int> hist_buckets(max_radix_steps);
        for(auto bucket_it = bucket_pairs.begin(); bucket_it != bucket_pairs.end(); bucket_it++) {
            int bin = bucket_it->first->bucket_depth;
            hist_buckets[bin]++;
        }
        for(int hist_buckets_index = 0; hist_buckets_index < hist_buckets.size(); hist_buckets_index++) {
            if(hist_buckets[hist_buckets_index]) {
                std::cout << hist_buckets_index << ":" << hist_buckets[hist_buckets_index] << std::endl;
            }
        }
    }

private:
    std::vector<DeviceConfig> devices;
    db_hash_table h_r_entry;
    db_hash_table h_s_entry;

    void configure_devices()
    {
        DeviceConfig config;
        config.device_id = 0;
        cudaStreamCreate(&config.stream);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

#if DEBUG_PRINT
        std::cout << "Mem Free=" << free_mem / std::pow(10, 9) << "GiB Mem Total=" << total_mem / std::pow(10, 9) << "GiB " << std::endl;
#endif

        devices.push_back(config);
    }

    void partition_r_recursive(db_table r_table_swap, db_hash_table r_hash_table_swap, int radix_width, BucketConfig *config, index_t max_elements, index_t max_radix_steps, const DeviceConfig device_config, bool gpu = true)
    {
        if (config->hash_table.size > 0 && config->hash_table.size > max_elements && config->bucket_depth <= max_radix_steps)
        {
            int radix_shift = (radix_width * config->bucket_depth);
            int buckets = config->buckets;
            bool index_data = config->bucket_depth == 0;

            config->histogram = new index_t[buckets];
            config->offsets = new index_t[buckets];
            config->sub_buckets = new BucketConfig[config->buckets];

            if (gpu)
            {
                partition_gpu(config->table, config->hash_table, r_table_swap, r_hash_table_swap, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, device_config.stream);
            }
            else
            {
                //partition(config->data, r_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, gpu);
            }

            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                index_t sub_elements = config->histogram[bucket_index];
                index_t sub_offset = config->offsets[bucket_index];

                BucketConfig *sub_bucket = &config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket and start partitioning on reduced data set
                sub_bucket->hash_table = db_hash_table(sub_offset, sub_elements, r_hash_table_swap);
                sub_bucket->table = db_table(sub_offset, sub_elements, r_table_swap);

                db_table sub_r_table_swap(sub_offset, sub_elements, config->table);
                db_hash_table sub_r_hash_table_swap(sub_offset, sub_elements, config->hash_table);

                partition_r_recursive(sub_r_table_swap, sub_r_hash_table_swap, radix_width, sub_bucket, max_elements, max_radix_steps, device_config, gpu);
                if (sub_bucket->sub_buckets)
                {
                    sub_bucket->table.free();
                    sub_bucket->hash_table.free();
                }
            }
        }
    }

    void partition_s_recursive(db_table s_table_swap, db_hash_table s_hash_table_swap, int radix_width, BucketConfig *config, BucketConfig *r_config, index_t max_radix_steps, DeviceConfig device_config, std::vector<bucket_pair_t> &bucket_pairs, bool gpu = true)
    {

        if (config->hash_table.size > 0 && r_config->sub_buckets && r_config->hash_table.size > 0)
        {
            int radix_shift = (radix_width * config->bucket_depth);
            int buckets = config->buckets;
            bool index_data = config->bucket_depth == 0;

            config->histogram = new index_t[buckets];
            config->offsets = new index_t[buckets];
            config->sub_buckets = new BucketConfig[config->buckets];
            if (gpu)
            {
                partition_gpu(config->table, config->hash_table, s_table_swap, s_hash_table_swap, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, device_config.stream);
            }
            else
            {
                //partition(config->data, s_swap_entry, radix_width, radix_shift, buckets, config->histogram, config->offsets, index_data, gpu);
            }

            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                index_t sub_elements = config->histogram[bucket_index];
                index_t sub_offset = config->offsets[bucket_index];

                // configure bucket
                BucketConfig *sub_bucket = &config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket
                sub_bucket->table = db_table(sub_offset, sub_elements, s_table_swap);
                sub_bucket->hash_table = db_hash_table(sub_offset, sub_elements, s_hash_table_swap);

                // set data for swap
                db_table sub_s_table_swap(sub_offset, sub_elements, config->table);
                db_hash_table sub_s_hash_table_swap(sub_offset, sub_elements, config->hash_table);

                partition_s_recursive(sub_s_table_swap, sub_s_hash_table_swap, radix_width, sub_bucket, &r_config->sub_buckets[bucket_index], max_radix_steps, device_config, bucket_pairs, gpu);
                if (sub_bucket->sub_buckets)
                {
                    sub_bucket->table.free();
                    sub_bucket->hash_table.free();
                }
            }
            // add leaf bucket to bucke tpair list
        }
        else if (config->hash_table.size > 0 && !r_config->sub_buckets && r_config->hash_table.size > 0)
        {
            bucket_pairs.push_back(std::make_pair(r_config, config));
        }
    }

    void probe(std::vector<bucket_pair_t> &bucket_pairs, int radix_width, index_t max_build_elements, DeviceConfig device_config, std::vector<db_table> &joined_rs_tables)
    {
        for (size_t bucket_pair_index = 0; bucket_pair_index < bucket_pairs.size(); bucket_pair_index++)
        {
            BucketConfig *r_config = bucket_pairs[bucket_pair_index].first;
            BucketConfig *s_config = bucket_pairs[bucket_pair_index].second;
#if DEBUG_PRINT
            std::cout << "Probe " << (bucket_pair_index + 1) << "/" << bucket_pairs.size() << std::endl;          
            print_mem();
            std::cout << "Join " << r_config->hash_table.size << ":" << s_config->hash_table.size;
#endif
            std::cout << "Join " << r_config->hash_table.size << ":" << s_config->hash_table.size << std::endl;
            if (r_config->hash_table.size && s_config->hash_table.size)
            {
                // hash_t * bucket_hash = new hash_t[r_config->data.size];
                // index_t * bucket_index = new index_t[r_config->data.size];
                // cudaMemcpy(bucket_hash, r_config->data.hashes, r_config->data.size * sizeof(index_t), cudaMemcpyDeviceToHost);
                // cudaMemcpy(bucket_index, r_config->data.indices, r_config->data.size * sizeof(index_t), cudaMemcpyDeviceToHost);
                // index_t missmatch_counter = 0;
                // for(index_t index = 0; index < r_config->data.size; index++) {
                //     missmatch_counter += (index_t)(bucket_hash[index] != h_r_entry.hashes[bucket_index[index]]);
                // }
                // std::cout << "Missmatch=" << missmatch_counter << std::endl;

                db_table joined_rs_table;
                // offset for table key
                int key_offset = (radix_width * r_config->bucket_depth);
                ProbeConfig config;
                build_and_probe_gpu(r_config->table, r_config->hash_table, s_config->table, s_config->hash_table, joined_rs_table, key_offset, device_config.stream, config);
#if DEBUG_PRINT
                std::cout << " M=" << joined_rs_table.size << std::endl;
#endif
                joined_rs_tables.push_back(joined_rs_table);

                //cudaStreamSynchronize(device_config.stream);
                //gpuErrchk(cudaGetLastError());
            }
        }
    }

    void hash_table(db_table table, db_hash_table &hash_table)
    {
        hash_table.size = table.size;
        hash_table.gpu = table.gpu;
        hash_table.hashes = new hash_t[hash_table.size];

        column_t *column_values = nullptr;
        if (table.gpu)
        {
            column_values = new column_t[table.size * table.column_count];
            gpuErrchk(cudaMemcpy(column_values, table.column_values, table.column_count * table.size * sizeof(column_t), cudaMemcpyDeviceToHost));
        }
        else
        {
            column_values = table.column_values;
        }

        // create hash
        for (index_t hash_index = 0; hash_index < hash_table.size; hash_index++)
        {
            hash_t hash_value = 0;
            for (size_t column_index = 1; column_index < table.column_count; column_index++)
            {
                hash_value += column_values[hash_index * table.column_count + column_index];
            }
            hash_table.hashes[hash_index] = hash_value;
        }

        // create indices
        // copy hashes to gpu if required
        if (table.gpu)
        {
            cudaMalloc(&hash_table.indices, hash_table.size * sizeof(index_t));
            hash_t *d_hashes = nullptr;
            cudaMalloc(&d_hashes, hash_table.size * sizeof(hash_t));
            cudaMemcpy(d_hashes, hash_table.hashes, hash_table.size * sizeof(hash_t), cudaMemcpyHostToDevice);
            delete[] hash_table.hashes;
            hash_table.hashes = d_hashes;
            delete[] column_values;
        }
        else
        {
            hash_table.indices = new index_t[hash_table.size];
        }
    }

    void merge_joined_tables(std::vector<db_table> partial_rs_tables, db_table &joined_rs_table)
    {
        index_t merged_size = 0;
        std::for_each(std::begin(partial_rs_tables), std::end(partial_rs_tables), [&merged_size](const db_table &table)
                      { merged_size += table.size; });

        db_table table_properties = partial_rs_tables[0];
        joined_rs_table.column_count = table_properties.column_count;
        joined_rs_table.gpu = table_properties.gpu;
        joined_rs_table.data_owner = true;
        joined_rs_table.size = merged_size;

        index_t merge_offset = 0;
        if (joined_rs_table.gpu)
        {
            cudaMalloc(&joined_rs_table.column_values, joined_rs_table.column_count * joined_rs_table.size * sizeof(column_t));

            for (auto table_it = partial_rs_tables.begin(); table_it != partial_rs_tables.end(); table_it++)
            {
                for (int column_index = 0; column_index < joined_rs_table.column_count; column_index++)
                {
                    cudaMemcpy(&joined_rs_table.column_values[merge_offset], table_it->column_values, table_it->size * table_it->column_count * sizeof(column_t), cudaMemcpyDeviceToDevice);
                }
                merge_offset += table_it->size;
            }
        }
    }
};

// perfect hashing ( for hash alg)