#include "join/join_provider.cuh"
#include "benchmark/data_generator.hpp"
#include "benchmark/configuration.hpp"

#include "json.hpp"

using json = nlohmann::json;

struct JoinBenchmarkResults
{
    JoinSummary join_summary;
    BenchmarkConfig benchmark_config;
    ProbeBenchmarkConfig probe_config;
    JoinBenchmarkConfig join_config;

    static std::string to_string_header()
    {
        std::ostringstream string_stream;
        string_stream
            << BenchmarkConfig::to_string_header() << ","
            << ProbeBenchmarkConfig::to_string_header() << ","
            << JoinBenchmarkConfig::to_string_header() << ","
            << "hash_tuple_p_second,"
            << "partition_tuple_p_second,"
            << "probe_tuple_p_second,"
            << "merge_tuple_p_second,"
            << "join_tuple_p_second,"
            << "join_gb_p_second";
        return string_stream.str();
    }

    std::string to_string()
    {
        std::ostringstream string_stream;
        string_stream
            << benchmark_config.to_string() << ","
            << probe_config.to_string() << ","
            << join_config.to_string() << ","
            << join_summary.hash_tuples_p_second << ","
            << join_summary.partition_tuples_p_second << ","
            << join_summary.probe_tuples_p_second << ","
            << join_summary.merge_tuples_p_second << ","
            << join_summary.join_tuples_p_second << ","
            << join_summary.join_gb_p_second;
        return string_stream.str();
    }
};

void store_join_summary(json &parent_json, JoinSummary summary, int id)
{
    json summary_json;
    summary_json["rs_elements"] = summary.rs_elements;
    summary_json["r_elements"] = summary.r_elements;
    summary_json["s_elements"] = summary.s_elements;
    summary_json["hash_tuples_p_second"] = summary.hash_tuples_p_second;
    summary_json["partition_tuples_p_second"] = summary.partition_tuples_p_second;
    summary_json["probe_tuples_p_second"] = summary.probe_tuples_p_second;
    summary_json["merge_tuples_p_second"] = summary.merge_tuples_p_second;
    summary_json["join_tuples_p_second"] = summary.join_tuples_p_second;
    json hash_summaries_json = json::array();
    for (auto hash_summary : summary.hash_summaries)
    {
        json hash_summary_json;
        hash_summary_json["k_tuples_p_second"] = hash_summary.k_tuples_p_seconds;
        hash_summary_json["k_gb_p_second"] = hash_summary.k_gb_p_seconds;
        hash_summary_json["elements"] = hash_summary.elements;
        hash_summary_json["element_bytes"] = hash_summary.element_bytes;
        hash_summary_json["element_offset"] = hash_summary.element_offset;
        hash_summary_json["algorithm"] = hash_summary.algorithm;
        hash_summaries_json.push_back(hash_summary_json);
    }
    summary_json["hash_summaries"] = hash_summaries_json;

    json partition_summaries_json = json::array();
    for (auto partition_summary : summary.partition_summaries)
    {
        json partition_summary_json;
        partition_summary_json["elements"] = partition_summary.elements;
        partition_summary_json["k_histogram_tuples_p_second"] = partition_summary.k_histrogram_elements_p_second;
        partition_summary_json["k_histogram_gb_p_second"] = partition_summary.k_histrogram_gb_p_second;
        partition_summary_json["k_swap_tuples_p_second"] = partition_summary.k_swap_elements_p_second;
        partition_summary_json["k_swap_gb_p_second"] = partition_summary.k_swap_gb_p_second;
        partition_summaries_json.push_back(partition_summary_json);
    }
    summary_json["partition_summaries"] = partition_summaries_json;

    json probe_summaries_json = json::array();
    for (auto probe_summary : summary.probe_summaries)
    {
        json probe_summary_json;
        probe_summary_json["k_build_probe_tuples_p_second"] = probe_summary.k_build_probe_tuples_p_second;
        probe_summary_json["k_build_probe_gb_p_second"] = probe_summary.k_build_probe_gb_p_second;
        probe_summary_json["k_extract_tuples_p_second"] = probe_summary.k_extract_tuples_p_second;
        probe_summary_json["k_extract_gb_p_second"] = probe_summary.k_extract_gb_p_second;
        probe_summary_json["r_elements"] = probe_summary.r_elements;
        probe_summary_json["s_elements"] = probe_summary.s_elements;
        probe_summary_json["rs_elements"] = probe_summary.rs_elements;
        probe_summaries_json.push_back(probe_summary_json);
    }
    summary_json["probe_summaries"] = probe_summaries_json;
    parent_json[id] = summary_json;
}

void verify(db_table expected, db_table actual)
{
    int error_count = 0;

    expected = expected.copy(false);
    actual = actual.copy(false);

    error_count += expected.column_count != actual.column_count;
    error_count += expected.data_owner != actual.data_owner;
    error_count += expected.size != actual.size;

    if (error_count > 0)
    {
        std::cout << "Table meta data not equal" << std::endl;
        exit(-1);
    }

    int half_column_count = expected.column_count / 2;
    std::vector<column_t> tested_values;
    for (index_t expected_test_index = 0; expected_test_index < expected.size; expected_test_index++)
    {
        column_t expected_test_value = expected.column_values[expected_test_index * expected.column_count];

        if (std::find(std::begin(tested_values), std::end(tested_values), expected_test_value) == std::end(tested_values))
        {
            tested_values.push_back(expected_test_value);

            std::vector<column_t> expected_values;
            std::vector<column_t> actual_values;
            for (index_t expected_index = 0; expected_index < expected.size; expected_index++)
            {
                column_t expected_value = expected.column_values[expected_index * expected.column_count];
                if (expected_value == expected_test_value)
                {
                    expected_values.push_back(expected.column_values[expected_index * expected.column_count + half_column_count]);
                }
            }

            for (index_t actual_index = 0; actual_index < actual.size; actual_index++)
            {
                column_t actual_value = actual.column_values[actual_index * actual.column_count];
                if (actual_value == expected_test_value)
                {
                    actual_values.push_back(actual.column_values[actual_index * actual.column_count + half_column_count]);
                }
            }

            std::sort(std::begin(expected_values), std::end(expected_values));
            std::sort(std::begin(actual_values), std::end(actual_values));

            if (expected_values != actual_values)
            {
                std::cout << "Join on " << expected_test_value << " failed " << actual_values.size() - expected_values.size() << std::endl;
                exit(-1);
            }
        }
    }

    expected.free();
    actual.free();
}

int main(int argc, char **argv)
{
    std::srand(0);

    BenchmarkSetup benchmark_setup;
    if (!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup))
    {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    if (!benchmark_setup.has_join_setup)
    {
        std::cout << "Failed to load join config" << std::endl;
        return -1;
    }

    if (!benchmark_setup.has_probe_setup)
    {
        std::cout << "Failed to load probe config" << std::endl;
        return -1;
    }

    if (!benchmark_setup.has_hash_setup)
    {
        std::cout << "Failed to load hash config" << std::endl;
        return -1;
    }

    std::fstream run_csv_stream;
    run_csv_stream.open((benchmark_setup.output_file_path + "/run.csv"), std::ios::app);
    run_csv_stream << JoinBenchmarkResults::to_string_header() << std::endl;

    auto benchmark_configs = get_benchmark_configs(benchmark_setup);
    auto join_configs = get_join_configs(benchmark_setup);
    auto probe_configs = get_probe_benchmark_configs(benchmark_setup);
    auto hash_configs = get_hash_benchmark_configs(benchmark_setup);

    bool gpu = true;

    json profile_json;
    int config_index = 0;
    for (BenchmarkConfig benchmark_config : benchmark_configs)
    {
        for (JoinBenchmarkConfig join_benchmark_config : join_configs)
        {
            for (HashBenchmarkConfig hash_benchmark_config : hash_configs)
            {
                for (ProbeBenchmarkConfig probe_benchmark_config : probe_configs)
                {
                    HashConfig hash_config;
                    hash_config.algorithm = hash_benchmark_config.algorithm;
                    hash_config.threads_per_block = hash_benchmark_config.thread_size;

                    ProbeConfig probe_config;
                    probe_config.build_n_per_thread = probe_benchmark_config.build_n_per_thread;
                    probe_config.build_table_load = probe_benchmark_config.build_table_load;
                    probe_config.build_threads = probe_benchmark_config.build_threads;
                    probe_config.extract_n_per_thread = probe_benchmark_config.extract_n_per_thread;
                    probe_config.extract_threads = probe_benchmark_config.extract_threads;
                    probe_config.d_probe_buffer = nullptr;

                    JoinConfig join_config;
                    join_config.profile_enabled = benchmark_config.profile;
                    join_config.devices = benchmark_config.gpus;
                    join_config.tasks_p_device = benchmark_config.max_streams_p_gpu;
                    join_config.probe_config = probe_config;
                    join_config.hash_config = hash_config;
                    JoinProvider join_provider(join_config);

                    JoinSummary total_join_summary;
                    for (int run_index = 0; run_index < benchmark_config.runs; run_index++)
                    {
                        std::cout << "C " << config_index << " " << run_index << std::endl;
                        int column_count = join_benchmark_config.rs_columns;
                        index_t r_table_size = benchmark_config.elements;
                        index_t s_table_size = r_table_size / join_benchmark_config.rs_scale;
                        index_t max_value = benchmark_config.element_max;

                        db_table r_table;
                        db_table s_table;
                        db_table rs_table;

                        generate_table(r_table_size, column_count, max_value, r_table, gpu);
                        generate_table(s_table_size, column_count, max_value, s_table, gpu);

                        //r_table.print();
                        //s_table.print();
                        //rs_table.print();

                        if (benchmark_config.verify)
                        {
                            db_table r_copy_table = r_table.copy(false);
                            db_table s_copy_table = s_table.copy(false);

                            join_provider.join(r_table, s_table, rs_table);
                            store_join_summary(profile_json, join_provider.get_join_summary(), config_index);

                            db_table rs_expected_table;
                            join_provider.join(r_copy_table, s_copy_table, rs_expected_table);

                            //rs_table.print();
                            //std::cout << "====" << std::endl;
                            //s_expected_table.print();

                            verify(rs_expected_table, rs_table);

                            r_copy_table.free();
                            s_copy_table.free();
                            rs_expected_table.free();
                        }
                        else
                        {
                            join_provider.join(r_table, s_table, rs_table);
                            store_join_summary(profile_json, join_provider.get_join_summary(), config_index);
                        }

                        r_table.free();
                        s_table.free();
                        rs_table.free();

                        total_join_summary += join_provider.get_join_summary();
                    }
                    config_index++;
                    total_join_summary /= benchmark_config.runs;

                    JoinBenchmarkResults result;
                    result.benchmark_config = benchmark_config;
                    result.join_config = join_benchmark_config;
                    result.probe_config = probe_benchmark_config;
                    result.join_summary = total_join_summary;
                    run_csv_stream << result.to_string() << std::endl;
                }
            }
        }
    }
    std::fstream run_profile_stream;
    run_profile_stream.open((benchmark_setup.output_file_path + "/run_profile.json"), std::ios::app);
    run_profile_stream << profile_json.dump(1) << std::endl;

    return 0;
}