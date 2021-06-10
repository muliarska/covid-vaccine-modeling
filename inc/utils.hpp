//
// Created by Yaroslav Morozevych on 4/21/21.
//
#ifndef COVID_MODEL_UTILS_HPP
#define COVID_MODEL_UTILS_HPP


#include "constants.hpp"


template<typename T, typename V>
std::vector<T> initialize_array(T symbol, V size) {
    std::vector<T> vec;
    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec.emplace_back(symbol);
    }
    return vec;
}


/* Generate random number in range [0.0, 1.0] */
double randfloat() {
    return (double)rand() / RAND_MAX;
}


template <typename T>
int randint(T min_arg, T max_arg) {
    return min_arg + rand() % (max_arg - 1);
}


/* Return the ith younger bit of a number (starting with 1 up to 32) */
int get_required_bit(int window, size_t position) {
    return (window >> (WINDOW_SIZE - position)) & 1;
}


/* Return the ith (taking into consideration offset (col - row)) symmetric bit corresponding to ith row */
int get_col_required_bit(int window, size_t col, size_t row) {
    size_t bit_idx = (col - row) % WINDOW_SIZE + 1; // we add 1 at the end as we start bit indexing from 1
    return get_required_bit(window, bit_idx);
}


/* Determine the number of all windows for creating adj_matrix */
int calc_window_num(size_t people_num) {
    int window_num = 0;
    for (size_t i = 0; i < people_num; i += WINDOW_SIZE) {
        window_num += (int)ceil((double)(people_num - i) / WINDOW_SIZE) * WINDOW_SIZE;
    }
    return window_num;
}


/* Create an array of starting windows for each person to speed up calculations */
int* get_window_indices(size_t people_num) {
    int* window_indices = new int[people_num];

    int window_idx = 0;
    for (size_t i = 0; i < people_num; i++) {
        window_indices[i] = window_idx;
        window_idx += ceil((double)(people_num - i) / WINDOW_SIZE);
    }
    return window_indices;
}
#endif //COVID_MODEL_UTILS_HPP