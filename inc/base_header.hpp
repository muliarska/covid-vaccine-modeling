//
// Created by Yaroslav Morozevych on 4/21/21.
//
#ifndef COVID_MODEL_BASE_HEADER_HPP
#define COVID_MODEL_BASE_HEADER_HPP

// INCLUDE STREAM LIBS
#include <iostream>
#include <fstream>
#include <sstream>

// INCLUDE DATA STRUCTURES
#include <unordered_map>
#include <map>
#include <vector>

// INCLUDE HELPER LIBS
#include <algorithm>
#include <iterator>
#include <cassert>
#include <cmath>

// INCLUDE HEADERS FOR RANDOM
#include <cstdlib>
#include <ctime>

// INCLUDE CUSTOM HEADERS
#include "constants.hpp"

// INCLUDE CUDA LIBS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"


// DEFINE ERROR CHECKING METHOD FOR CUDA
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


// INCLUDE STRUCTS
struct config_t {
    int people_num;
    int days, start_vaccine;
    int when_vaccinated;
    int max_contacts;
    double prob_connect;
    bool who_vaccinated;
    bool is_lockdown;
};

struct states_t {
    double fi;      // from M to S
    double gamma;   // from EIR to M
    double alpha;   // from E to I
    double beta;    // from S to E == contact rate
    double sigma;   // from I to R
    double omega;   // from S to V, num of vaccinated men per day * vaccine quality
    double delta;   // from R to D
    double theta;   // from V to S, vaccine duration
    double lambda;
};

struct gpu_vars_t {
    int people_num;         // equivalent to config.people_num
    int last_person;        // representation of the last person in person array   
    int seed;               // random seed
    int day;

    double beta, omega;     // vital coefficients for S state people

    bool who_vaccinated;
    int start_vaccine;
    int max_contacts;
};

// INCLUDE MAIN CLASS
class CovidModel {
    config_t config{};
    states_t states{};
    
    int* adj_matrix;        // random network adjacency matrix
    int* window_indices;    // indices of entry windows for each person
    char* people_states;    // array of people states

    double* prob_connect;   // probability of two persons to be in contact
    size_t window_num;      // number of all windows
    
    // arrays that represent transition tables
    char s_trans_states[4] = { S_STATE, E_STATE, V_STATE, V_STATE };
    std::unordered_map<char, std::pair<char, double>> transition_states;
    std::vector<std::map<char, double>> perc_of_people_each_state;

    // covid model adjustment vars
    double limit_amount_of_r = 0.001, limit_r_for_vaccine = 0.001;
    int check_lockdown = 0, check_vaccine = 0;
    int max_increasing = 5, decreasing_of_r = 0;
    int max_point_vaccine = 16;

    // thread local GPU memory vars
    struct states_t* d_states;
    struct config_t* d_config;
    struct gpu_vars_t gpu_vars;

    int* d_adj_matrix;
    int* d_people_num;
    int* d_window_indices;
    int* d_thread_range;
    char* d_s_trans_states;
    char* d_people_states;
    double* d_prob_connect;
   
public:
    CovidModel(config_t& cfg, states_t& st);
    ~CovidModel();
    void print_matrix();
    void covid_model();
    std::vector<std::map<char, double>> get_state_percentages();
private:
    void init_states();
    void init_prob_connect();
    void init_transition_states();
    void init_gpu_memory();
    void build_matrix();
    void run_simulation(char* temp_states, int day);
    char get_next_state(char curr_state, double trans_prob, char next_state);
    std::map<char, double> get_states();
};

#endif //COVID_MODEL_BASE_HEADER_HPP