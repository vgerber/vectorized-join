#pragma once
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "hash/hash.cu"
#include "join_provider.cu"

struct BucketConfig {
    int buckets = 0;
    int bucket_depth = 0;
    std::vector<std::shared_ptr<BucketConfig>> sub_buckets;
    index_t *histogram = nullptr;
    index_t *offsets = nullptr;
    db_hash_table hash_table;
    db_table table;

    ~BucketConfig() {
        delete[] histogram;
        delete[] offsets;
    }

    void print() {
        std::cout << std::string(bucket_depth, '-') << hash_table.size << std::endl;
        if (sub_buckets.size() > 0) {
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                sub_buckets[bucket_index]->print();
            }
        }
    }

    void print_compare(const std::shared_ptr<BucketConfig> compare_config) {
        std::cout << std::string(bucket_depth, '-') << hash_table.size << ":" << compare_config->hash_table.size << std::endl;
        if (sub_buckets.size() > 0) {
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                sub_buckets[bucket_index]->print_compare(compare_config->sub_buckets[bucket_index]);
            }
        }
    }
};

typedef std::shared_ptr<BucketConfig> bucket_ptr;
typedef std::pair<bucket_ptr, bucket_ptr> bucket_pair_t;

struct bucket_pair_comparator {
    bool operator()(const bucket_pair_t &pair1, const bucket_pair_t &pair2) {
        return (pair1.first->hash_table.size * pair1.second->hash_table.size) > (pair2.first->hash_table.size * pair2.second->hash_table.size);
    }
};

struct JoinSummary {
    JoinStatus join_status;

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

    JoinSummary &operator+=(const JoinSummary &js) {
        hash_tuples_p_second += js.hash_tuples_p_second;
        partition_tuples_p_second += js.partition_tuples_p_second;
        probe_tuples_p_second += js.probe_tuples_p_second;
        merge_tuples_p_second += js.merge_tuples_p_second;
        join_tuples_p_second += js.join_tuples_p_second;
        join_gb_p_second += js.join_gb_p_second;
        rs_elements += js.rs_elements;
        return *this;
    }

    JoinSummary &operator/=(const float &factor) {
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
        std::cout << rs_elements << " Joins " << join_tuples_p_second << " Tuples/s" << join_gb_p_second << " GB/s" << std::endl;
    }
};

struct JoinConfig {
    int tasks_p_device = 1;
    int devices = 1;
    bool profile_enabled = true;
    bool vectorize = false;
    int vector_bytes_size = 0;
    int buckets = 2;

    HashConfig hash_config;
    PartitionConfig partition_config;
    ProbeConfig probe_config;
};

struct DeviceConfig {
    int device_id = 0;

    std::vector<std::shared_ptr<std::mutex>> stream_locks;
    std::vector<cudaStream_t> streams;
    std::vector<ProbeConfig> stream_probe_configurations;
    std::vector<PartitionConfig> stream_partition_configurations;

    bool profile_enabled = false;
    std::vector<cudaEvent_t> profiling_events;

    void free_probe_resources() {
        for (auto &probe_config : stream_probe_configurations) {
            probe_config.free();
        }
        stream_probe_configurations.clear();
    }

    void free() {
        for (int probe_config_index = 0; probe_config_index < stream_probe_configurations.size(); probe_config_index++) {
            cudaStream_t stream = streams[probe_config_index];
        }

        free_probe_resources();

        for (auto stream : streams) {
            cudaStreamDestroy(stream);
        }
        streams.clear();

        for (auto profiling_event : profiling_events) {
            cudaEventDestroy(profiling_event);
        }
        profiling_events.clear();

        for (auto &probe_config : stream_probe_configurations) {
            probe_config.free();
        }
        stream_partition_configurations.clear();

        for (auto &partition_config : stream_partition_configurations) {
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
        return std::make_pair(profiling_events[stream_index * 2], profiling_events[stream_index * 2 + 1]);
    }

    void synchronize_device() {
        for (auto stream : streams) {
            gpuErrchk(cudaStreamSynchronize(stream));
        }
    }

  private:
    int next_stream_index = 0;
    std::mutex device_mutex;
};

class JoinProvider {

  public:
    JoinProvider(JoinConfig join_config) {
        this->join_config = join_config;
        configure_devices();
    }

    JoinStatus join(db_table r_table, db_table s_table, db_table &joined_rs_table) {
        assert(r_table.gpu == s_table.gpu);
        if (!r_table.gpu && !s_table.gpu) {
            return join_cpu(r_table, s_table, joined_rs_table);
        } else if (r_table.gpu && s_table.gpu) {
            return join_gpu(r_table, s_table, joined_rs_table);
        }
        return JoinStatus(false, "Unable to figure out the join algrithm");
    }

    JoinStatus join_cpu(db_table r_table, db_table s_table, db_table &joined_rs_table) {

        // matching
        // allocate matching intermed buffer
        index_t matched_tuples = 0;
        bool *matching_results = new bool[r_table.size * s_table.size];

        // match all r with all s values
        for (index_t s_index = 0; s_index < s_table.size; s_index++) {
            for (index_t r_index = 0; r_index < r_table.size; r_index++) {
                int match_counter = 0;
                for (int column_index = 0; column_index < r_table.column_count; column_index++) {
                    column_t r_value = r_table.column_values[r_index * r_table.column_count + column_index];
                    column_t s_value = s_table.column_values[s_index * s_table.column_count + column_index];
                    match_counter += (r_value == s_value);
                }
                bool matching_result = match_counter == r_table.column_count;
                matching_results[s_index * r_table.size + r_index] = matching_result;
                matched_tuples += matching_result;
            }
        }

        // join
        // allocated rs table based on matching results
        joined_rs_table = db_table(2, matched_tuples, false);
        int rs_half_column_count = joined_rs_table.column_count / 2;

        // build rs table from matching results
        index_t rs_index = 0;

        for (int match_index = 0; match_index < r_table.size * s_table.size; match_index++) {
            if (matching_results[match_index]) {
                index_t r_index = match_index % r_table.size;
                index_t s_index = match_index / r_table.size;

                joined_rs_table.primary_keys[rs_index] = rs_index + 1;

                index_t rs_offset = rs_index * joined_rs_table.column_count;
                joined_rs_table.column_values[rs_offset] = r_table.primary_keys[r_index];
                joined_rs_table.column_values[rs_offset + 1] = s_table.primary_keys[s_index];
                rs_index++;
            }
        }
        return JoinStatus(true);
    }

    JoinStatus join_gpu(db_table r_table, db_table s_table, db_table &joined_rs_table) {
        join_summary = JoinSummary();
        auto join_start = std::chrono::high_resolution_clock::now();

        assert(r_table.gpu == s_table.gpu);
        assert(r_table.column_count == s_table.column_count);
#if DEBUG_PRINT
        std::cout << "Join " << std::endl;
#endif

        int buckets = join_config.buckets;
        int radix_width = std::ceil(std::log2(buckets));
        int bins = (1 << radix_width);

        int max_radix_steps = (8 * sizeof(index_t)) / radix_width;

        auto remaining_memory = get_memory_left().first;
        auto swap_memory = (r_table.get_bytes() + s_table.get_bytes() + (r_table.size + s_table.size) * 2 * sizeof(hash_t));
        if (remaining_memory < swap_memory) {
            free_all();
            float missing_memory_gb = ((int64_t)remaining_memory - swap_memory) / powf(10, 9);
            auto swap_status = JoinStatus(false, "Not enough memory for swap tables and hashes " + std::to_string(missing_memory_gb) + "GB missing");
            join_summary.join_status = swap_status;
            return swap_status;
        }

        r_hash_table = db_hash_table(r_table.size, r_table.gpu);
        r_hash_table_swap = db_hash_table(r_table.size, r_table.gpu);
        r_table_swap = db_table(r_table.column_count, r_table.size, r_table.gpu);

        s_hash_table = db_hash_table(s_table.size, r_table.gpu);
        s_hash_table_swap = db_hash_table(s_table.size, s_table.gpu);
        s_table_swap = db_table(s_table.column_count, s_table.size, s_table.gpu);

        auto hash_start = std::chrono::high_resolution_clock::now();

        auto hash_status = hash_table(r_table, r_hash_table, devices[0], join_config.hash_config);
        if (hash_status.hash_failed()) {
            free_all();
            join_summary.join_status = hash_status;
            return hash_status;
        }

        hash_status = hash_table(s_table, s_hash_table, devices[0], join_config.hash_config);
        if (hash_status.hash_failed()) {
            free_all();
            join_summary.join_status = hash_status;
            return hash_status;
        }

        gpuErrchk(cudaGetLastError());
        bucket_ptr r_bucket_config = std::make_shared<BucketConfig>();

        r_bucket_config->buckets = bins;
        r_bucket_config->table = r_table;
        r_bucket_config->hash_table = r_hash_table;

        bucket_ptr s_bucket_config = std::make_shared<BucketConfig>();
        s_bucket_config->buckets = bins;
        s_bucket_config->table = s_table;
        s_bucket_config->hash_table = s_hash_table;

        std::vector<bucket_pair_t> bucket_pairs;

        for (auto device : devices) {
            device->synchronize_device();
        }
        auto hash_end = std::chrono::high_resolution_clock::now();

        auto partition_start = std::chrono::high_resolution_clock::now();
        partition_recursive(r_table_swap, r_hash_table_swap, s_table_swap, s_hash_table_swap, radix_width, r_bucket_config, s_bucket_config, &bucket_pairs, join_config.vector_bytes_size, max_radix_steps, devices[0], true);
        auto partition_end = std::chrono::high_resolution_clock::now();
        gpuErrchk(cudaGetLastError());
#if DEBUG_PRINT
        r_bucket_config->print_compare(s_bucket_config);
#endif

        auto probe_start = std::chrono::high_resolution_clock::now();
        auto probe_status = probe(bucket_pairs, radix_width, devices[0], joined_rs_tables);
        if (probe_status.hash_failed()) {
            free_all();
            join_summary.join_status = probe_status;
            return probe_status;
        }

        for (auto device : devices) {
            device->synchronize_device();
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        gpuErrchk(cudaGetLastError());

        for (auto device : devices) {
            device->free_probe_resources();
        }

        r_hash_table.free();
        r_hash_table_swap.free();
        r_table_swap.free();

        s_hash_table.free();
        s_hash_table_swap.free();
        s_table_swap.free();

        bucket_pairs.clear();

        auto merge_start = std::chrono::high_resolution_clock::now();
        auto merge_status = merge_joined_tables(joined_rs_tables, joined_rs_table, devices[0]);
        if (merge_status.hash_failed()) {
            free_all();
            join_summary.join_status = merge_status;
            return merge_status;
        }

        for (auto device : devices) {
            device->synchronize_device();
        }
        auto merge_end = std::chrono::high_resolution_clock::now();
        gpuErrchk(cudaGetLastError());

        for (auto &partial_rs_table : joined_rs_tables) {
            partial_rs_table.free();
        }
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
        join_summary.join_tuples_p_second = ((r_table.size + s_table.size) / join_druation.count());
        join_summary.join_gb_p_second = (r_table.size + s_table.size) * r_table.column_count * sizeof(column_t) / pow(10, 9) / join_druation.count();
        join_summary.rs_elements = joined_rs_table.size;
        join_summary.r_elements = r_table.size;
        join_summary.s_elements = s_table.size;

        join_summary.join_status = JoinStatus(true);
        return join_summary.join_status;
    }

    JoinSummary get_join_summary() {
        return join_summary;
    }

    ~JoinProvider() {
        free_all();
    }

  private:
    JoinConfig join_config;
    std::vector<DeviceConfig *> devices;
    db_hash_table h_r_entry;
    db_hash_table h_s_entry;
    JoinSummary join_summary;
    std::mutex partition_lock;

    db_hash_table r_hash_table;
    db_hash_table r_hash_table_swap;
    db_table r_table_swap;

    db_hash_table s_hash_table;
    db_hash_table s_hash_table_swap;
    db_table s_table_swap;

    std::vector<db_table> joined_rs_tables;

    void free_all() {
        r_hash_table.free();
        r_hash_table_swap.free();
        r_table_swap.free();

        s_hash_table.free();
        s_hash_table_swap.free();
        s_table_swap.free();

        for (auto &table : joined_rs_tables) {
            table.free();
        }
        joined_rs_tables.clear();

        for (auto device : devices) {
            device->free();
            delete device;
        }
        devices.clear();
    }

    size_t get_probe_size(bucket_pair_t bucket_pair, ProbeConfig probe_config) {
        int64_t total_size = 0;
        if (probe_config.probe_mode == ProbeConfig::MODE_GLOBAL_R) {
            total_size += probe_config.get_table_size(bucket_pair.first->hash_table.size);
        }
        total_size += (bucket_pair.first->table.size * bucket_pair.second->table.size + 1) * sizeof(index_s_t);

        // handle overflow
        if (total_size < 0) {
            return SIZE_MAX;
        }

        return total_size;
    }

    void configure_devices() {
        DeviceConfig *device_config = new DeviceConfig();
        device_config->device_id = devices.size();
        device_config->profile_enabled = join_config.profile_enabled;
        cudaSetDevice(device_config->device_id);
        for (int stream_index = 0; stream_index < join_config.tasks_p_device; stream_index++) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            device_config->streams.push_back(stream);
            device_config->stream_locks.push_back(std::make_shared<std::mutex>());

            ProbeConfig probe_config = join_config.probe_config;
            probe_config.stream = stream;

            PartitionConfig partition_config = join_config.partition_config;
            partition_config.stream = stream;

            if (join_config.profile_enabled) {
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

    void partition_recursive(db_table r_table_swap, db_hash_table r_hash_table_swap, db_table s_table_swap, db_hash_table s_hash_table_swap, int radix_width, bucket_ptr r_bucket_config, bucket_ptr s_bucket_config, std::vector<bucket_pair_t> *bucket_pairs, int max_partition_bytes,
                             index_t max_radix_steps, DeviceConfig *device_config, bool gpu = true) {
        int partition_bytes = r_table_swap.get_bytes() + r_hash_table_swap.get_bytes() + s_table_swap.get_bytes() + s_hash_table_swap.get_bytes();
        bool is_any_table_empty = r_bucket_config->hash_table.size == 0 || s_bucket_config->hash_table.size == 0;
        if (!is_any_table_empty && r_bucket_config->bucket_depth < max_radix_steps && partition_bytes > max_partition_bytes) {
            int radix_shift = (radix_width * r_bucket_config->bucket_depth);
            int buckets = r_bucket_config->buckets;

            r_bucket_config->histogram = new index_t[buckets];
            r_bucket_config->offsets = new index_t[buckets];

            s_bucket_config->histogram = new index_t[buckets];
            s_bucket_config->offsets = new index_t[buckets];
            if (gpu) {
                int stream_index = device_config->get_next_queue_index();
                std::lock_guard<std::mutex> stream_lock(*device_config->stream_locks[stream_index]);

                PartitionConfig r_partition_config = device_config->stream_partition_configurations[stream_index];
                r_partition_config.set_radix_width(radix_width);
                partition_gpu(r_bucket_config->table, r_bucket_config->hash_table, r_table_swap, r_hash_table_swap, radix_shift, r_bucket_config->histogram, r_bucket_config->offsets, r_partition_config);
                join_summary.partition_summaries.push_back(r_partition_config.profiling_summary);

                PartitionConfig s_partition_config = device_config->stream_partition_configurations[stream_index];
                s_partition_config.set_radix_width(radix_width);
                partition_gpu(s_bucket_config->table, s_bucket_config->hash_table, s_table_swap, s_hash_table_swap, radix_shift, s_bucket_config->histogram, s_bucket_config->offsets, s_partition_config);

                gpuErrchk(cudaStreamSynchronize(s_partition_config.stream));
                join_summary.partition_summaries.push_back(s_partition_config.profiling_summary);
            } else {
                // partition(bucket_config->data, s_swap_entry, radix_width, radix_shift, buckets, bucket_config->histogram, bucket_config->offsets, index_data, gpu);
            }
            r_bucket_config->buckets = buckets;
            s_bucket_config->buckets = buckets;

            r_bucket_config->sub_buckets = std::vector<bucket_ptr>(buckets);
            s_bucket_config->sub_buckets = std::vector<bucket_ptr>(buckets);
            std::vector<std::thread> partition_threads;
            for (int bucket_index = 0; bucket_index < buckets; bucket_index++) {
                /*
                 * R - Swap
                 */
                index_t r_sub_elements = r_bucket_config->histogram[bucket_index];
                index_t r_sub_offset = r_bucket_config->offsets[bucket_index];
                index_t s_sub_elements = s_bucket_config->histogram[bucket_index];
                index_t s_sub_offset = s_bucket_config->offsets[bucket_index];

                // configure bucket
                bucket_ptr r_sub_bucket = std::make_shared<BucketConfig>();
                r_bucket_config->sub_buckets[bucket_index] = r_sub_bucket;
                r_sub_bucket->bucket_depth = r_bucket_config->bucket_depth + 1;
                r_sub_bucket->histogram = nullptr;
                r_sub_bucket->offsets = nullptr;
                r_sub_bucket->buckets = buckets;

                bucket_ptr s_sub_bucket = std::make_shared<BucketConfig>();
                s_bucket_config->sub_buckets[bucket_index] = s_sub_bucket;
                s_sub_bucket->bucket_depth = s_bucket_config->bucket_depth + 1;
                s_sub_bucket->histogram = nullptr;
                s_sub_bucket->offsets = nullptr;
                s_sub_bucket->buckets = buckets;

                // set data for bucket
                r_sub_bucket->table = db_table(r_sub_offset, r_sub_elements, r_table_swap);
                r_sub_bucket->hash_table = db_hash_table(r_sub_offset, r_sub_elements, r_hash_table_swap);

                s_sub_bucket->table = db_table(s_sub_offset, s_sub_elements, s_table_swap);
                s_sub_bucket->hash_table = db_hash_table(s_sub_offset, s_sub_elements, s_hash_table_swap);

                // set data for swap
                db_table sub_r_table_swap(r_sub_offset, r_sub_elements, r_bucket_config->table);
                db_hash_table sub_r_hash_table_swap(r_sub_offset, r_sub_elements, r_bucket_config->hash_table);

                db_table sub_s_table_swap(s_sub_offset, s_sub_elements, s_bucket_config->table);
                db_hash_table sub_s_hash_table_swap(s_sub_offset, s_sub_elements, s_bucket_config->hash_table);

                auto partition_child = [this, sub_r_table_swap, sub_r_hash_table_swap, sub_s_table_swap, sub_s_hash_table_swap, radix_width, r_sub_bucket, s_sub_bucket, bucket_pairs, max_partition_bytes, max_radix_steps, device_config, gpu]() {
                    partition_recursive(sub_r_table_swap, sub_r_hash_table_swap, sub_s_table_swap, sub_s_hash_table_swap, radix_width, r_sub_bucket, s_sub_bucket, bucket_pairs, max_partition_bytes, max_radix_steps, device_config, gpu);
                };
                partition_child();
                // partition_threads.push_back(std::thread(partition_child));
            }
            for (auto &partition_thread : partition_threads) {
                partition_thread.join();
            }
        }
        // add leaf bucket to bucket pair list
        bool is_leaf = r_bucket_config->sub_buckets.size() == 0 && s_bucket_config->sub_buckets.size() == 0;
        if (!is_any_table_empty && is_leaf) {
            std::lock_guard<std::mutex> lock(partition_lock);

            // r_bucket_config->table.print();
            // r_bucket_config->hash_table.print();
            // std::cout << "Vectorize " << r_bucket_config->table.size << " X " << s_bucket_config->table.size << std::endl;
            add_and_vectorize_bucket(r_bucket_config, s_bucket_config, bucket_pairs);
        }
    }

    void add_and_vectorize_bucket(bucket_ptr r_bucket, bucket_ptr s_bucket, std::vector<bucket_pair_t> *vectorized_buckets) {
        // split r data into buckets fitting the vector size limitation for r

        auto probe_config = join_config.probe_config;
        int r_size_limitation = join_config.vector_bytes_size / 2;
        index_t max_r_elements = r_bucket->table.size;
        if (join_config.probe_config.probe_mode == ProbeConfig::MODE_PARTITION_S) {
            max_r_elements = join_config.probe_config.get_max_table_elements();
        }

        index_t max_r_vector_elements = probe_config.get_max_vector_elements(r_size_limitation, r_bucket->table.column_count);
        max_r_vector_elements = min(max_r_vector_elements, max_r_elements);

        int r_sub_buckets_size = 1;
        int r_sub_elements = min(r_bucket->table.size, max_r_vector_elements);
        r_sub_buckets_size = max(1, (int)ceil((float)r_bucket->table.size / r_sub_elements));
        int r_bytes = db_table::get_bytes(r_sub_elements, r_bucket->table.column_count);

        std::vector<bucket_ptr> r_sub_buckets = std::vector<bucket_ptr>(r_sub_buckets_size);
        for (int sub_bucket_index = 0; sub_bucket_index < r_sub_buckets_size; sub_bucket_index++) {
            r_sub_buckets[sub_bucket_index] = std::make_shared<BucketConfig>();
            auto r_sub_bucket = r_sub_buckets[sub_bucket_index];
            r_sub_bucket->bucket_depth = r_bucket->bucket_depth;
            r_sub_bucket->buckets = 0;
            int r_offset = sub_bucket_index * r_sub_elements;

            if (sub_bucket_index < r_sub_buckets_size - 1) {
                r_sub_bucket->table = db_table(r_offset, r_sub_elements, r_bucket->table);
                r_sub_bucket->hash_table = db_hash_table(r_offset, r_sub_elements, r_bucket->hash_table);
            } else {
                r_sub_bucket->table = db_table(r_offset, r_bucket->table.size - r_offset, r_bucket->table);
                r_sub_bucket->hash_table = db_hash_table(r_offset, r_bucket->table.size - r_offset, r_bucket->hash_table);
            }
        }

        // calculate remaining memory for s table in vector
        int s_vector_size_limitation = join_config.vector_bytes_size - r_bytes;
        int s_max_vector_elements = ProbeConfig::get_max_s_vector_elements(s_vector_size_limitation, s_bucket->table);
        if (s_max_vector_elements >= s_bucket->table.size) {
            for (int r_sub_bucket_index = 0; r_sub_bucket_index < r_sub_buckets_size; r_sub_bucket_index++) {
                vectorized_buckets->push_back(std::make_pair(r_sub_buckets[r_sub_bucket_index], s_bucket));
            }
        } else {
            int s_sub_elements = min(s_bucket->table.size, (index_t)s_max_vector_elements);
            int s_sub_buckets_size = max(1, (int)ceil((float)s_bucket->table.size / s_sub_elements));
            int s_sub_decrement = s_sub_elements;

            std::vector<bucket_ptr> s_sub_buckets = std::vector<bucket_ptr>(s_sub_buckets_size);
            for (int s_sub_bucket_index = 0; s_sub_bucket_index < s_sub_buckets_size; s_sub_bucket_index++) {
                int s_offset = s_sub_decrement * s_sub_bucket_index;
                s_sub_buckets[s_sub_bucket_index] = std::make_shared<BucketConfig>();
                auto s_sub_bucket = s_sub_buckets[s_sub_bucket_index];
                s_sub_bucket->bucket_depth = s_bucket->bucket_depth;
                s_sub_bucket->buckets = 0;

                if (s_sub_bucket_index < s_sub_buckets_size - 1) {
                    s_sub_bucket->table = db_table(s_offset, s_sub_elements, s_bucket->table);
                    s_sub_bucket->hash_table = db_hash_table(s_offset, s_sub_elements, s_bucket->hash_table);
                } else {
                    s_sub_bucket->table = db_table(s_offset, s_bucket->table.size - s_offset, s_bucket->table);
                    s_sub_bucket->hash_table = db_hash_table(s_offset, s_bucket->table.size - s_offset, s_bucket->hash_table);
                }
            }

            for (int r_sub_bucket_index = 0; r_sub_bucket_index < r_sub_buckets_size; r_sub_bucket_index++) {
                for (int s_sub_bucket_index = 0; s_sub_bucket_index < s_sub_buckets_size; s_sub_bucket_index++) {
                    // std::cout << "Vector " << r_sub_buckets[r_sub_bucket_index]->table.size << " X " << s_sub_buckets[s_sub_bucket_index]->table.size << std::endl;
                    vectorized_buckets->push_back(std::make_pair(r_sub_buckets[r_sub_bucket_index], s_sub_buckets[s_sub_bucket_index]));
                }
            }
        }
    }

    JoinStatus probe(std::vector<bucket_pair_t> &bucket_pairs, int radix_width, DeviceConfig *device_config, std::vector<db_table> &joined_rs_tables) {
        std::sort(std::begin(bucket_pairs), std::end(bucket_pairs), bucket_pair_comparator());
        for (size_t bucket_pair_index = 0; bucket_pair_index < bucket_pairs.size(); bucket_pair_index++) {
            bucket_ptr r_config = bucket_pairs[bucket_pair_index].first;
            bucket_ptr s_config = bucket_pairs[bucket_pair_index].second;

            if (r_config->hash_table.size && s_config->hash_table.size) {
                // offset for table key
                int key_offset = (radix_width * r_config->bucket_depth);

                int queue_index = device_config->get_next_queue_index();
                ProbeConfig *probe_config = &device_config->stream_probe_configurations[queue_index];
                probe_config->probe_mode = join_config.probe_config.probe_mode;

                int64_t required_probe_memory = (int64_t)get_probe_size(bucket_pairs[bucket_pair_index], *probe_config) - probe_config->get_allocated_memory();
                if (required_probe_memory > 0 && get_memory_left().first < required_probe_memory) {
                    return JoinStatus(false, "Cannot allocate new memory for probing");
                }

                db_table joined_rs_table;
                auto probe_status = build_and_probe_gpu(r_config->table, r_config->hash_table, s_config->table, s_config->hash_table, joined_rs_table, key_offset, *probe_config);
                if (probe_status.hash_failed()) {
                    return probe_status;
                }

                joined_rs_tables.push_back(joined_rs_table);
                join_summary.probe_summaries.push_back((*probe_config).profiling_summary);
            }
        }
        return JoinStatus(true);
    }

    JoinStatus hash_table(db_table table, db_hash_table &hash_table, DeviceConfig *device_config, HashConfig hash_config) {
        // create indices
        // copy hashes to gpu if required
        if (table.gpu) {
            int queue_index = device_config->get_next_queue_index();
            hash_config.stream = device_config->streams[queue_index];

            if (device_config->profile_enabled) {
                auto profile_events = device_config->get_profiling_events(queue_index);
                hash_config.enable_profile(profile_events.first, profile_events.second);
            }

            if ((table.column_count * sizeof(column_t)) % sizeof(chunk_t)) {
                return JoinStatus(false, "Columns do not fit fully into chunk buffer and padding is not provided");
            }

            int element_size = table.column_count * sizeof(column_t);
            hash_func(table.size, 0, element_size / sizeof(chunk_t), (chunk_t *)table.column_values, hash_table.hashes, hash_config);

            if (device_config->profile_enabled) {
                join_summary.hash_summaries.push_back(hash_config.profile_summary);
            }
        } else {
            for (index_t hash_index = 0; hash_index < hash_table.size; hash_index++) {
                hash_t hash_value = 0;
                for (size_t column_index = 0; column_index < table.column_count; column_index++) {
                    hash_value += table.column_values[hash_index * table.column_count + column_index];
                }
                hash_table.hashes[hash_index] = hash_value;
            }
        }
        return JoinStatus(true);
    }

    JoinStatus merge_joined_tables(std::vector<db_table> &partial_rs_tables, db_table &joined_rs_table, DeviceConfig *device_config) {
        if (partial_rs_tables.size() == 0) {
            return JoinStatus(true);
        }

        index_t merged_size = 0;
        std::for_each(std::begin(partial_rs_tables), std::end(partial_rs_tables), [&merged_size](const db_table &table) { merged_size += table.size; });

        db_table table_properties = partial_rs_tables[0];
        joined_rs_table.column_count = table_properties.column_count;
        joined_rs_table.gpu = table_properties.gpu;
        joined_rs_table.data_owner = true;
        joined_rs_table.size = merged_size;

        index_t merge_offset = 0;
        if (joined_rs_table.gpu) {
            // print_mem();

            auto memory_free = get_memory_left().first;
            if (memory_free < joined_rs_table.get_bytes()) {
                return JoinStatus(false, "Not enough memory for full rs table (" + std::to_string(joined_rs_table.size) + " Rows)");
            }

            gpuErrchk(cudaMalloc(&joined_rs_table.column_values, joined_rs_table.column_count * joined_rs_table.size * sizeof(column_t)));
            gpuErrchk(cudaMalloc(&joined_rs_table.primary_keys, joined_rs_table.size * sizeof(column_t)));

            for (auto table_it = partial_rs_tables.begin(); table_it != partial_rs_tables.end(); table_it++) {
                int stream_index = device_config->get_next_queue_index();
                gpuErrchk(cudaMemcpyAsync(&joined_rs_table.column_values[merge_offset * table_it->column_count], table_it->column_values, table_it->size * table_it->column_count * sizeof(column_t), cudaMemcpyDeviceToDevice, device_config->streams[stream_index]));
                gpuErrchk(cudaMemcpyAsync(&joined_rs_table.primary_keys[merge_offset], table_it->primary_keys, table_it->size * sizeof(column_t), cudaMemcpyDeviceToDevice, device_config->streams[stream_index]));
                merge_offset += table_it->size;
                // table_it->free(device_config->streams[stream_index]);
            }
        }
        return JoinStatus(true);
    }
};