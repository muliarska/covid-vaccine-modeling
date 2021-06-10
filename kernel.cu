#include "inc/base_header.hpp"
#include "inc/config.hpp"
#include "inc/cuda_kernels.cuh"


int rand_int(int min_arg, int max_arg) {
    return min_arg + rand() % (max_arg - 1);
}


void CovidModel::build_matrix() {
    int seed = rand_int(MIN_SEED, MAX_SEED);

    // copy seed
    int* d_seed;
    gpuErrorCheck(cudaMalloc(&d_seed, sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_seed, &seed, sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel
    encode_bits<<<config.people_num, THREAD_NUM>>>(d_adj_matrix, d_window_indices, d_prob_connect, d_people_num, d_thread_range, d_seed);

    // copy results back
    gpuErrorCheck(cudaMemcpy(adj_matrix, d_adj_matrix, window_num * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaFree(d_seed));
}


void CovidModel::run_simulation(char* temp_states, int day) {
    thrust::host_vector<int> s_state_people_vec{};
    std::vector<int> rest_state_people_vec{};
    int s_counter = 0, rest_counter = 0;

    // separate people on those who are in S state from others
    for (int person = 0; person < config.people_num; person++) {
        if (people_states[person] == S_STATE) {
            s_state_people_vec.push_back(person);
            s_counter++;
        }
        else {
            rest_state_people_vec.push_back(person);
            rest_counter++;
        }
    }
    int seed = rand_int(MIN_SEED, MAX_SEED);

    int* s_state_people = new int[s_counter];
    for (int i = 0; i < s_counter; i++) {
        s_state_people[i] = s_state_people_vec[i];
    }

    // copy S state people indices
    int* d_s_state;
    gpuErrorCheck(cudaMalloc(&d_s_state, s_counter * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_s_state, s_state_people, s_counter * sizeof(int), cudaMemcpyHostToDevice));

    // copy temp states
    char* d_temp_states;
    gpuErrorCheck(cudaMalloc(&d_temp_states, config.people_num * sizeof(char)));
    gpuErrorCheck(cudaMemcpy(d_temp_states, temp_states, config.people_num * sizeof(char), cudaMemcpyHostToDevice));

    // update data
    gpu_vars.day = day; gpu_vars.last_person = s_state_people[s_counter - 1];
    gpu_vars.start_vaccine = config.start_vaccine; gpu_vars.seed = seed;

    struct gpu_vars_t* d_gpu_vars;
    gpuErrorCheck(cudaMalloc((void**)&d_gpu_vars, sizeof(struct gpu_vars_t)));
    gpuErrorCheck(cudaMemcpy(d_gpu_vars, &gpu_vars, sizeof(struct gpu_vars_t), cudaMemcpyHostToDevice));

    // launch kernel
    s_state_simulation<<<s_counter / (THREAD_NUM / 2), (THREAD_NUM / 2)>>>(d_gpu_vars, d_s_trans_states, d_s_state, d_temp_states, d_people_states, d_window_indices, d_adj_matrix);

    // copy results and free memory
    gpuErrorCheck(cudaMemcpy(temp_states, d_temp_states, config.people_num * sizeof(char), cudaMemcpyDeviceToHost));
    cudaFree(d_gpu_vars); cudaFree(d_temp_states); cudaFree(d_s_state);

    // lanch kernel for people in other states apart from S
    for (size_t person : rest_state_people_vec) {
        char current_state = people_states[person], next_state = transition_states[current_state].first;
        double trans_prob = transition_states[current_state].second;
        // switch state
        temp_states[person] = get_next_state(current_state, trans_prob, next_state);
    }
}


void write_results(std::string& file_path, std::vector<std::map<char, double>>& states_perc, config_t& config) {
    std::ofstream out(file_path);

    for (size_t day = 0; day < config.days; day++) {
        std::string line{};
        for (auto pair : states_perc[day]) {
            line += (std::to_string(pair.second) + " ");
        }
        line += "\n";
        out << line;
    }
}


int main() {
    std::string cfg_path = "config.txt";
    std::string out_path = "output.txt";

    // set up srand() for random
    srand(time(NULL));

    config_t config{};
    states_t states{};
    read_config(cfg_path, config, states);

    CovidModel covid_model = CovidModel{ config, states };
    covid_model.covid_model();

    std::vector<std::map<char, double>> perc_people_each_state = covid_model.get_state_percentages();
    write_results(out_path, perc_people_each_state, config);

    return 0;
}