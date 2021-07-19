#include "join/join_provider.cuh"
#include "benchmark/data_generator.hpp"
#include "benchmark/configuration.hpp"


int main(int argc, char** argv) {

    std::srand(0);

    BenchmarkSetup benchmark_setup;
    if(!load_benchmark_setup(std::string(argv[1]), std::string(argv[2]), &benchmark_setup)) {
        std::cout << "Failed to load config" << std::endl;
        return -1;
    }

    bool gpu = true;
    
    int column_count = 1;
    index_t table_size = benchmark_setup.elements;
    index_t max_value = benchmark_setup.element_max;

    db_table r_table;
    db_table s_table;
    db_table rs_table;

    
    generate_table(table_size, column_count, max_value, r_table, gpu);
    generate_table(r_table.size * 0.75, column_count, max_value, s_table, gpu);

    JoinProvider join_provider;
    
    join_provider.join(r_table, s_table, rs_table);

    //r_table.print();
    //s_table.print();
    //rs_table.print();
    

    r_table.free();
    s_table.free();
    rs_table.free();

    return 0;
}