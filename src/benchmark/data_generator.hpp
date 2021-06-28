#pragma once
#include <iostream>
//#include <<experimental/random>
#include <random>

template<class V>
void generate_random_numbers(size_t size, V max_int, V *buffer) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<V> distr(0, max_int);

    for(size_t x_index = 0; x_index < size; x_index++) {
        buffer[x_index] = distr(gen);
    }
}