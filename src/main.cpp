#include "../inc/base_header.hpp"
#include "../inc/config.hpp"




void write_results(std::string &file_path, std::vector<std::map<char, double>> &states_perc, config_t &config) {
    std::ofstream out(file_path);

    for(size_t day = 0; day < config.days; day++) {
        std::string line{};
        for(auto pair: states_perc[day]) {
            line += (std::to_string(pair.second) + " ");
        }
        line += "\n";
        out << line;
    }
}

int get_window_idx3(size_t blockIdx, size_t people_num) {
    int window_idx = 0;
    for(size_t i = 0; i < blockIdx;  i++) {
        window_idx += ceil((double)(people_num - i) / WINDOW_SIZE);
    }
    return window_idx;
}


int get_required_bit3(int window, size_t position) {
    return (window >> (WINDOW_SIZE - position)) & 1;
}


int get_col_required_bit3(int* adj_matrix, size_t col, size_t row, size_t people_num) {
    size_t bit_idx = (col - row) % WINDOW_SIZE + 1; // we add 1 at the end as we start bit indexing from 1
    size_t window_idx = get_window_idx3(row, people_num) + (col - row) / WINDOW_SIZE;
    return get_required_bit3(adj_matrix[window_idx], bit_idx);
}


int* get_window_indices2(size_t people_num) {
    int* window_indices = new int[people_num];

    int window_idx = 0;
    for(size_t i = 0; i < people_num;  i++) {
        window_indices[i] = window_idx;
        window_idx += ceil((double)(people_num - i) / WINDOW_SIZE);
    }
    return window_indices;
}


int main() {
    std::string cfg_path = "../config.dat";
    std::string out_path = "../output.txt";

    // set up srand() for random
    srand(time(NULL));

    config_t config{};
    states_t states{};
    read_config(cfg_path, config, states);
//    int* w = get_window_indices2(64);
//    for(int i = 0; i < 64; i++) {
//        std::cout << i << " : " << w[i] << std::endl;
//    }

    CovidModel covid_model = CovidModel{config, states};
    covid_model.covid_model();

    std::vector<std::map<char, double>> perc_people_each_state = covid_model.get_state_percentages();
    write_results(out_path, perc_people_each_state, config);
    return 0;
}