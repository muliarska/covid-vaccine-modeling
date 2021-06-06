//
// Created by Yaroslav Morozevych on 4/21/21.
//
#ifndef COVID_MODEL_UTILS_HPP
#define COVID_MODEL_UTILS_HPP




template<typename T, typename V>
std::vector<T> initialize_array(T symbol, V size) {
    std::vector<T> vec;
    vec.reserve(size);

    for(size_t i = 0; i < size; i++) {
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
int get_required_bit(int num, size_t position) {
    return (num >> (WINDOW_SIZE - position)) & 1;
}


/* Return the ith (taking into consideration offset (col - row)) symmetric bit corresponding to ith row */
int get_col_required_bit(std::vector<std::vector<unsigned int>> &adj_matrix, size_t col, size_t row) {
    size_t bit_idx = (col - row) % WINDOW_SIZE + 1; // we add 1 at the end as we start bit indexing from 1
    size_t window_idx = ceil((float)(col - row) / WINDOW_SIZE) - 1;
    return get_required_bit(adj_matrix[row][window_idx], bit_idx);
}

#endif //COVID_MODEL_UTILS_HPP
