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
#include <cassert>
#include <cmath>

// INCLUDE HEADERS FOR RANDOM
#include <cstdlib>
#include <ctime>

// INCLUDE CUSTOM HEADERS
#include "constants.hpp"

// INCLUDE STRUCTS
struct config_t {
    long people_num;
    long days, start_vaccine;
    bool who_vaccinated;
    int when_vaccinated;
    double prob_connect;
    long max_contacts;
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

// INCLUDE MAIN CLASS
class CovidModel {
    config_t config{};
    states_t states{};

    std::vector<char> people_states;
    std::vector<std::vector<unsigned int>> adj_matrix;
    std::unordered_map<char, std::pair<char, double>> transition_states;
    std::vector<std::map<char, double>> perc_of_people_each_state;

    int check_lockdown = 0; // TODO
    double limit_amount_of_r = 0.001;
    int max_increasing = 5;

    int check_vaccine = 0;
    int decreasing_of_r = 0;
    int max_point_vaccine = 16;
    double limit_r_for_vaccine = 0.001;
public:
    CovidModel(config_t &cfg, states_t &st);
    void print_matrix();
    void covid_model();
    std::vector<std::map<char, double>> get_state_percentages();
private:
    void set_up_states();
    std::vector<double> init_prob_connect(double mean, double variance);
    void build_matrix();
    void init_transition_states();
    void run_simulation(std::vector<char> &temp_states, size_t day);
    std::map<char, double> get_states();
    std::pair<size_t, size_t> get_infected_num(size_t person);
};

#endif //COVID_MODEL_BASE_HEADER_HPP
