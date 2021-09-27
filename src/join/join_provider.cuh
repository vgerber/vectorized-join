#pragma once
#include <vector>
#include <thread>
#include <mutex>

#include "hash/hash.cu"
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

typedef std::pair<BucketConfig *, BucketConfig *> bucket_pair_t;

struct bucket_pair_comparator {
    bool operator() (const bucket_pair_t &pair1, const bucket_pair_t &pair2) {
        return (pair1.first->hash_table.size * pair1.second->hash_table.size) > (pair2.first->hash_table.size * pair2.second->hash_table.size);
    }
};

struct JoinSummary {
    float hash_tuples_p_second = 0.0;
    std::vector<HashSummary> hash_summaries;

    float partition_tuples_p_second = 0.0;
    std::vector<PartitionSummary> partition_summaries;

    float probe_tuples_p_second = 0.0;
    std::vector<ProbeSummary> probe_summaries;

    float merge_tuples_p_second = 0.0;
    float merge_gb_p_second = 0.0;

    float join_tuples_p_second = 0.0;
    float join_gb_p_second = 0.0;

    index_t r_elements = 0;
    index_t s_elements = 0;
    index_t rs_elements = 0;


    JoinSummary& operator+=(const JoinSummary &js) {
        hash_tuples_p_second += js.hash_tuples_p_second;
        partition_tuples_p_second += js.partition_tuples_p_second;
        probe_tuples_p_second += js.probe_tuples_p_second;
        merge_tuples_p_second += js.merge_tuples_p_second;
        join_tuples_p_second += js.join_tuples_p_second;
        join_gb_p_second += js.join_gb_p_second;
        rs_elements += js.rs_elements;
        return *this;
    }

    JoinSummary& operator/=(const float &factor) {
        hash_tuples_p_second /= factor;
        partition_tuples_p_second /= factor;
        probe_tuples_p_second /= factor;
        merge_tuples_p_second /= factor;
        join_tuples_p_second /= factor;
        join_gb_p_second /= factor;
        rs_elements /= factor;
        return *this;
    }

    void print() {
        std::cout << "Hash      " << hash_tuples_p_second << " Tuples/s" << std::endl;
        std::cout << "Partition " << partition_tuples_p_second << " Tuples/s" << std::endl;
        std::cout << "Probe     " << probe_tuples_p_second << " Tuples/s" << std::endl;
        std::cout << "Merge     " << merge_tuples_p_second << " Tuples/s" << std::endl;
        std::cout << rs_elements << " Joins " << join_tuples_p_second << " Tuples/s" << join_gb_p_second << " GB/s" <<std::endl;
    }

};

struct JoinConfig {
    int tasks_p_device = 1;
    int devices = 1;
    bool profile_enabled = true;

    HashConfig hash_config;
    PartitionConfig partition_config;
    ProbeConfig probe_config;
};

struct DeviceConfig
{
    int device_id = 0;
    
    std::vector<cudaStream_t> streams;
    std::vector<ProbeConfig> stream_probe_configurations;
    std::vector<PartitionConfig> stream_partition_configurations;

    bool profile_enabled = false;
    std::vector<cudaEvent_t> profiling_events;
    
    

    void free() {
        for(int  probe_config_index = 0; probe_config_index < stream_probe_configurations.size(); probe_config_index++) {
            cudaStream_t stream = streams[probe_config_index];
            stream_probe_configurations[probe_config_index].free();
        }

        stream_probe_configurations.clear();

        for(auto stream : streams) {
            cudaStreamDestroy(stream);
        }
        streams.clear();

        for(auto profiling_event : profiling_events) {
            cudaEventDestroy(profiling_event);
        }
        profiling_events.clear();

        for(auto &probe_config : stream_probe_configurations) {
            probe_config.free();
        }
        stream_partition_configurations.clear();

        for(auto &partition_config : stream_partition_configurations) {
            partition_config.free();
        }
        stream_partition_configurations.clear();
    }

    int get_next_queue_index() {
        const std::lock_guard<std::mutex> lock(device_mutex);
        next_stream_index = (next_stream_index + 1) % streams.size(); 
        return next_stream_index;
    }

    std::pair<cudaEvent_t, cudaEvent_t> get_profiling_events(int stream_index) {
        return std::make_pair(profiling_events[stream_index*2], profiling_events[stream_index*2+1]);
    }

    void synchronize_device() {
        for(auto stream : streams) {
            cudaStreamSynchronize(stream);
        }
    }

private:
    int next_stream_index = 0;
    std::mutex device_mutex;
};

class JoinProvider
{

public:
    JoinProvider(JoinConfig join_config) {
        this->join_config = join_config;
        configure_devices();
    }

    
    void join(db_table r_table, db_table s_table, db_table &joined_rs_table) {
        assert(r_table.gpu == s_table.gpu);
        if(!r_table.gpu && !s_table.gpu) {
            join_cpu(r_table, s_table, joined_rs_table);
        } else if(r_table.gpu && s_table.gpu) {
            join_gpu(r_table, s_table, joined_rs_table);
        }
    }

    void join_cpu(db_table r_table, db_table s_table, db_table &joined_rs_table) {
        joined_rs_table.gpu = false;
        joined_rs_table.size = 0;
        joined_rs_table.data_owner = true;
        joined_rs_table.column_count = 2; // (r_table.column_count + s_table.column_count);
        int rs_half_column_count = joined_rs_table.column_count / 2;

        // matching
        // allocate matching intermed buffer
        index_t matched_tuples = 0;
        bool *matching_results = new bool[r_table.size * s_table.size];

        // match all r with all s values
        for(index_t s_index = 0; s_index < s_table.size; s_index++) {
            for(index_t r_index = 0; r_index < r_table.size; r_index++) {
                int match_counter = 0;
                for(int column_index = 1; column_index < r_table.column_count; column_index++) {
                    column_t r_value = r_table.column_values[r_index * r_table.column_count + column_index];
                    column_t s_value = s_table.column_values[s_index * s_table.column_count + column_index];
                    match_counter += (r_value == s_value);
                }
                bool matching_result = match_counter == (r_table.column_count-1);
                matching_results[s_index * r_table.size + r_index] = matching_result;
                matched_tuples += matching_result;
            }
        }

        // join
        // allocated rs table based on matching results
        joined_rs_table.size = matched_tuples;
        joined_rs_table.column_values = new column_t[matched_tuples * joined_rs_table.column_count];
        
        // build rs table from matching results
        index_t rs_index = 0;
        
        for(int match_index = 0; match_index < r_table.size * s_table.size; match_index++) {
            if(matching_results[match_index]) {            
                index_t r_index = match_index % r_table.size;
                index_t s_index = match_index / r_table.size;

                for(int column_index = 0; column_index < rs_half_column_count; column_index++) {
                    index_t rs_offset = rs_index * joined_rs_table.column_count;
                    joined_rs_table.column_values[rs_offset + column_index] = r_table.column_values[r_index * r_table.column_count + column_index];
                    joined_rs_table.column_values[rs_offset + rs_half_column_count + column_index] = s_table.column_values[s_index * s_table.column_count + column_index]; 
                }
                rs_index++;
            }
        }
    }

    void join_gpu(db_table r_table, db_table s_table, db_table &joined_rs_table)
    {
        join_summary = JoinSummary();
        auto join_start = std::chrono::high_resolution_clock::now();

        assert(r_table.gpu == s_table.gpu);
        assert(r_table.column_count == s_table.column_count);
#if DEBUG_PRINT
        std::cout << "Join " << std::endl;
#endif

        int buckets = 8;
        int radix_width = std::ceil(std::log2(buckets));
        int bins = (1 << radix_width);

        int max_radix_steps = (8 * sizeof(index_t)) / radix_width;
        int max_bucket_r_elements = join_config.probe_config.max_r_elements;

        db_hash_table r_hash_table(r_table.size, r_table.gpu);
        db_hash_table r_hash_table_swap(r_table.size, r_table.gpu);
        db_table r_table_swap(r_table.column_count, r_table.size, r_table.gpu);

        db_hash_table s_hash_table(s_table.size, r_table.gpu);
        db_hash_table s_hash_table_swap(s_table.size, s_table.gpu);
        db_table s_table_swap(s_table.column_count, s_table.size, s_table.gpu);

        auto hash_start = std::chrono::high_resolution_clock::now();
        hash_table(r_table, r_hash_table, devices[0], join_config.hash_config);
        hash_table(s_table, s_hash_table, devices[0], join_config.hash_config);
        
        BucketConfig r_bucket_config;
        r_bucket_config.buckets = bins;
        r_bucket_config.table = r_table;
        r_bucket_config.hash_table = r_hash_table;

        BucketConfig s_bucket_config;
        s_bucket_config.buckets = bins;
        s_bucket_config.table = s_table;
        s_bucket_config.hash_table = s_hash_table;

        std::vector<bucket_pair_t> bucket_pairs;

        for(auto device : devices) {
            device->synchronize_device();
        }
        auto hash_end = std::chrono::high_resolution_clock::now();

        auto partition_start = std::chrono::high_resolution_clock::now();
        partition_r_recursive(r_table_swap, r_hash_table_swap, radix_width, &r_bucket_config, max_bucket_r_elements, max_radix_steps, devices[0], true);
        partition_s_recursive(s_table_swap, s_hash_table_swap, radix_width, &s_bucket_config, &r_bucket_config, max_radix_steps, devices[0], &bucket_pairs, true);
        auto partition_end = std::chrono::high_resolution_clock::now();

#if DEBUG_PRINT
        r_bucket_config.print_compare(&s_bucket_config);
#endif

        auto probe_start = std::chrono::high_resolution_clock::now();
        std::vector<db_table> joined_rs_tables;
        probe(bucket_pairs, radix_width, devices[0], joined_rs_tables);
        auto probe_end = std::chrono::high_resolution_clock::now();
        
        for(auto device : devices) {
            device->synchronize_device();
        }
        

        r_hash_table.free();
        r_hash_table_swap.free();
        r_table_swap.free();

        s_hash_table.free();
        s_hash_table_swap.free();
        s_table_swap.free();

        bucket_pairs.clear();


        auto merge_start = std::chrono::high_resolution_clock::now();
        merge_joined_tables(joined_rs_tables, joined_rs_table, devices[0]);
        auto merge_end = std::chrono::high_resolution_clock::now();

        joined_rs_tables.clear();

        auto join_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> join_druation = (join_end - join_start);
        std::chrono::duration<double> hash_druation = (hash_end - hash_start);
        std::chrono::duration<double> partition_druation = (partition_end - partition_start);
        std::chrono::duration<double> probe_druation = (probe_end - probe_start);
        std::chrono::duration<double> merge_druation = (merge_end - merge_start);

        join_summary.hash_tuples_p_second = ((r_table.size + s_table.size) / hash_druation.count());
        join_summary.partition_tuples_p_second = ((r_table.size + s_table.size) / partition_druation.count());
        join_summary.probe_tuples_p_second = ((r_table.size + s_table.size) / probe_druation.count());
        join_summary.merge_tuples_p_second = ((r_table.size + s_table.size) / merge_druation.count());
        join_summary.join_tuples_p_second = ((r_table.size * s_table.size) / join_druation.count());
        join_summary.join_gb_p_second = (r_table.size + s_table.size) * r_table.column_count * sizeof(column_t) / pow(10, 9) / join_druation.count();
        join_summary.rs_elements = joined_rs_table.size;
        join_summary.r_elements = r_table.size;
        join_summary.s_elements = s_table.size;
    }


    JoinSummary get_join_summary() {
        return join_summary;
    }

    ~JoinProvider() {
        for(auto device : devices) {
            device->free();
            delete device;
        }
    }

private:
    JoinConfig join_config;
    std::vector<DeviceConfig*> devices;
    db_hash_table h_r_entry;
    db_hash_table h_s_entry;
    JoinSummary join_summary;
    std::mutex partition_lock;

    void configure_devices()
    {
        DeviceConfig *device_config = new DeviceConfig();
        device_config->device_id = devices.size();
        device_config->profile_enabled = join_config.profile_enabled;
        cudaSetDevice(device_config->device_id);
        for(int stream_index = 0; stream_index < join_config.tasks_p_device; stream_index++) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            
            device_config->streams.push_back(stream);


            ProbeConfig probe_config = join_config.probe_config;
            probe_config.stream = stream;

            PartitionConfig partition_config = join_config.partition_config;
            partition_config.stream = stream;

            if(join_config.profile_enabled) {
                cudaEvent_t e_start, e_end;
                cudaEventCreate(&e_start);
                cudaEventCreate(&e_end);
                device_config->profiling_events.push_back(e_start);
                device_config->profiling_events.push_back(e_end);
            
                probe_config.enable_profiling(e_start, e_end);
                partition_config.enable_profiling(e_start, e_end);
            }

            device_config->stream_partition_configurations.push_back(partition_config);
            device_config->stream_probe_configurations.push_back(probe_config);
            
        }
        

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

#if DEBUG_PRINT
        std::cout << "Mem Free=" << free_mem / std::pow(10, 9) << "GiB Mem Total=" << total_mem / std::pow(10, 9) << "GiB " << std::endl;
#endif

        devices.push_back(device_config);
    }

    void partition_r_recursive(db_table r_table_swap, db_hash_table r_hash_table_swap, int radix_width, BucketConfig *bucket_config, index_t max_elements, index_t max_radix_steps, DeviceConfig *device_config, bool gpu = true)
    {
        if (bucket_config->hash_table.size > 0 && bucket_config->hash_table.size > max_elements && bucket_config->bucket_depth <= max_radix_steps)
        {
            
            int radix_shift = (radix_width * bucket_config->bucket_depth);
            int buckets = bucket_config->buckets;
            bool index_data = bucket_config->bucket_depth == 0;

            bucket_config->histogram = new index_t[buckets];
            bucket_config->offsets = new index_t[buckets];
            bucket_config->sub_buckets = new BucketConfig[bucket_config->buckets];


            if (gpu)
            {
                //std::cout << "P " << bucket_config << " -> " << bucket_config->table.size << ":" << r_table_swap.size << std::endl;
                
                int stream_index = device_config->get_next_queue_index();
                PartitionConfig partition_config = device_config->stream_partition_configurations[stream_index];
                partition_config.set_radix_width(radix_width);
                partition_gpu(bucket_config->table, bucket_config->hash_table, r_table_swap, r_hash_table_swap, radix_shift, bucket_config->histogram, bucket_config->offsets, index_data, partition_config);
            
                join_summary.partition_summaries.push_back(partition_config.profiling_summary);
            }
            else
            {
                //partition(bucket_config->data, r_swap_entry, radix_width, radix_shift, buckets, bucket_config->histogram, bucket_config->offsets, index_data, gpu);
            }

            std::vector<std::thread> partition_threads;
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                index_t sub_elements = bucket_config->histogram[bucket_index];
                index_t sub_offset = bucket_config->offsets[bucket_index];

                BucketConfig *sub_bucket = &bucket_config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = bucket_config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket and start partitioning on reduced data set
                sub_bucket->hash_table = db_hash_table(sub_offset, sub_elements, r_hash_table_swap);
                sub_bucket->table = db_table(sub_offset, sub_elements, r_table_swap);

                db_table sub_r_table_swap(sub_offset, sub_elements, bucket_config->table);
                db_hash_table sub_r_hash_table_swap(sub_offset, sub_elements, bucket_config->hash_table);

                //std::cout << "PP " << sub_bucket << " -> " << sub_r_table_swap.size << ":" << sub_bucket->table.size << std::endl;

                auto partition_child = [this, sub_r_table_swap, sub_r_hash_table_swap, radix_width, sub_bucket, max_elements, max_radix_steps, device_config, gpu]() { 
                    partition_r_recursive(sub_r_table_swap, sub_r_hash_table_swap, radix_width, sub_bucket, max_elements, max_radix_steps, device_config, gpu); 
                };
                partition_child();
                //partition_threads.push_back(std::thread(partition_child));
            }

            for(auto &partition_thread : partition_threads) {
                partition_thread.join();
            }
        }
    }

    void partition_s_recursive(db_table s_table_swap, db_hash_table s_hash_table_swap, int radix_width, BucketConfig *bucket_config, BucketConfig *r_config, index_t max_radix_steps, DeviceConfig *device_config, std::vector<bucket_pair_t> *bucket_pairs, bool gpu = true)
    {

        if (bucket_config->hash_table.size > 0 && r_config->sub_buckets && r_config->hash_table.size > 0)
        {
            int radix_shift = (radix_width * bucket_config->bucket_depth);
            int buckets = bucket_config->buckets;
            bool index_data = bucket_config->bucket_depth == 0;

            bucket_config->histogram = new index_t[buckets];
            bucket_config->offsets = new index_t[buckets];
            bucket_config->sub_buckets = new BucketConfig[bucket_config->buckets];
            if (gpu)
            {
                int stream_index = device_config->get_next_queue_index();
                PartitionConfig partition_config = device_config->stream_partition_configurations[stream_index];
                partition_config.set_radix_width(radix_width);
                partition_gpu(bucket_config->table, bucket_config->hash_table, s_table_swap, s_hash_table_swap, radix_shift, bucket_config->histogram, bucket_config->offsets, index_data, partition_config);

                join_summary.partition_summaries.push_back(partition_config.profiling_summary);
            }
            else
            {
                //partition(bucket_config->data, s_swap_entry, radix_width, radix_shift, buckets, bucket_config->histogram, bucket_config->offsets, index_data, gpu);
            }

            std::vector<std::thread> partition_threads;
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++)
            {
                index_t sub_elements = bucket_config->histogram[bucket_index];
                index_t sub_offset = bucket_config->offsets[bucket_index];

                // configure bucket
                BucketConfig *sub_bucket = &bucket_config->sub_buckets[bucket_index];
                sub_bucket->bucket_depth = bucket_config->bucket_depth + 1;
                sub_bucket->histogram = nullptr;
                sub_bucket->offsets = nullptr;
                sub_bucket->sub_buckets = nullptr;
                sub_bucket->buckets = buckets;

                // set data for bucket
                sub_bucket->table = db_table(sub_offset, sub_elements, s_table_swap);
                sub_bucket->hash_table = db_hash_table(sub_offset, sub_elements, s_hash_table_swap);

                // set data for swap
                db_table sub_s_table_swap(sub_offset, sub_elements, bucket_config->table);
                db_hash_table sub_s_hash_table_swap(sub_offset, sub_elements, bucket_config->hash_table);

                BucketConfig *r_sub_bucket = &r_config->sub_buckets[bucket_index];
                //partition_s_recursive(sub_s_table_swap, sub_s_hash_table_swap, radix_width, sub_bucket, r_sub_bucket, max_radix_steps, device_config, bucket_pairs, gpu);
                
                auto partition_child = [this, sub_s_table_swap, sub_s_hash_table_swap, radix_width, sub_bucket, r_sub_bucket, max_radix_steps, device_config, bucket_pairs, gpu](){ 
                    partition_s_recursive(sub_s_table_swap, sub_s_hash_table_swap, radix_width, sub_bucket, r_sub_bucket, max_radix_steps, device_config, bucket_pairs, gpu); 
                };
                partition_child();
                //partition_threads.push_back(std::thread(partition_child));
            }
            for(auto &partition_thread : partition_threads) {
                partition_thread.join();
            }

            
        }
        // add leaf bucket to bucket pair list
        else if (bucket_config->hash_table.size > 0 && !r_config->sub_buckets && r_config->hash_table.size > 0)
        {
            std::lock_guard<std::mutex> lock(partition_lock);
            bucket_pairs->push_back(std::make_pair(r_config, bucket_config));
        }
    }

    void probe(std::vector<bucket_pair_t> &bucket_pairs, int radix_width, DeviceConfig *device_config, std::vector<db_table> &joined_rs_tables)
    {

        std::sort(std::begin(bucket_pairs), std::end(bucket_pairs), bucket_pair_comparator());
        for (size_t bucket_pair_index = 0; bucket_pair_index < bucket_pairs.size(); bucket_pair_index++)
        {
            BucketConfig *r_config = bucket_pairs[bucket_pair_index].first;
            BucketConfig *s_config = bucket_pairs[bucket_pair_index].second;

            if (r_config->hash_table.size && s_config->hash_table.size)
            {
                // offset for table key
                int key_offset = (radix_width * r_config->bucket_depth);

                int queue_index = device_config->get_next_queue_index();
                ProbeConfig *probe_config = &device_config->stream_probe_configurations[queue_index];

                index_t total_r_elements = r_config->table.size;
                index_t r_offset = 0;
                while (total_r_elements > 0)
                {
                    db_table joined_rs_table;
                    index_t elements_step = min((index_t)(*probe_config).max_r_elements, total_r_elements-r_offset);
                    build_and_probe_gpu(db_table(r_offset, elements_step, r_config->table), db_hash_table(r_offset, elements_step, r_config->hash_table), s_config->table, s_config->hash_table, joined_rs_table, key_offset, *probe_config);
                    total_r_elements -= elements_step;
                    r_offset += elements_step;
                    joined_rs_tables.push_back(joined_rs_table);
                
                    if((*probe_config).profiling_enabled) {
                        join_summary.probe_summaries.push_back((*probe_config).profiling_summary);
                    }
                }
            }
        }
    }

    void hash_table(db_table table, db_hash_table &hash_table, DeviceConfig *device_config, HashConfig hash_config)
    {
        // create indices
        // copy hashes to gpu if required
        if (table.gpu)
        {
            int queue_index = device_config->get_next_queue_index();
            hash_config.stream = device_config->streams[queue_index];

            if(device_config->profile_enabled) {
                auto profile_events = device_config->get_profiling_events(queue_index);
                hash_config.enable_profile(profile_events.first, profile_events.second);
            }

            int element_size = table.column_count * sizeof(column_t);
            hash_func(table.size, sizeof(column_t) / sizeof(chunk_t), element_size / sizeof(chunk_t), (chunk_t*)table.column_values, hash_table.hashes, hash_config);
        
            if(device_config->profile_enabled) {
                join_summary.hash_summaries.push_back(hash_config.profile_summary);
            }
        }
        else
        {
            for (index_t hash_index = 0; hash_index < hash_table.size; hash_index++)
            {
                hash_t hash_value = 0;
                for (size_t column_index = 1; column_index < table.column_count; column_index++)
                {
                    hash_value += table.column_values[hash_index * table.column_count + column_index];
                }
                hash_table.hashes[hash_index] = hash_value;
            }
        }
    }

    void merge_joined_tables(std::vector<db_table> &partial_rs_tables, db_table &joined_rs_table, DeviceConfig *device_config)
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
                    int stream_index = device_config->get_next_queue_index();
                    cudaMemcpyAsync(&joined_rs_table.column_values[merge_offset], table_it->column_values, table_it->size * table_it->column_count * sizeof(column_t), cudaMemcpyDeviceToDevice, device_config->streams[stream_index]);
                }
                merge_offset += table_it->size * table_it->column_count;
                table_it->free(device_config->streams[device_config->get_next_queue_index()]);
            }
        }
    }
};

// perfect hashing ( for hash alg)