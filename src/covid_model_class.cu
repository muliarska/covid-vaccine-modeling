#include "../inc/base_header.hpp"
#include "../inc/utils.hpp"


CovidModel::CovidModel(config_t& cfg, states_t& st) {
    config = cfg;
    states = st;
    gpu_vars = gpu_vars_t{};

    config.prob_connect = 100.0 / (double)config.people_num; // TODO: add to config
    config.max_contacts = (long)(config.people_num * 0.01);

    window_indices = get_window_indices(config.people_num);
    window_num = calc_window_num(config.people_num);
    adj_matrix = new int[window_num];

    init_states();
    init_transition_states();
    init_prob_connect();
    init_gpu_memory();
}


/* Assign S state to not infected people and E state to the infected ones. */
void CovidModel::init_states() {
    long infected_num = (long)(INFECT_PROB * (double)config.people_num);

    std::vector<char >people_states_ = initialize_array(S_STATE, config.people_num - infected_num);
    std::vector<char> infected = initialize_array(E_STATE, infected_num);

    people_states_.insert(people_states_.end(), infected.begin(), infected.end());

    // rewrite people states to be a dynamically allocated array
    people_states = new char[config.people_num];
    for (int i = 0; i < people_states_.size(); i++) {
        people_states[i] = people_states_[i];
    }

    perc_of_people_each_state.reserve(config.days);
}


void CovidModel::init_transition_states() {
    std::unordered_map<char, std::pair<char, double>>
        transition{ {E_STATE, std::make_pair(I_STATE, states.alpha)},
                   {I_STATE, std::make_pair(R_STATE, states.sigma)},
                   {R_STATE, std::make_pair(D_STATE, states.delta)},
                   {V_STATE, std::make_pair(S_STATE, states.theta)},
                   {M_STATE, std::make_pair(S_STATE, states.fi)},
                   {D_STATE, std::make_pair(S_STATE, states.lambda)} };
    transition_states = transition;
}


void CovidModel::init_prob_connect() {
    prob_connect = new double[config.people_num];
    for (int i = 0; i < config.max_contacts; i++) {
        prob_connect[i] = config.prob_connect * 100;
    }
    for (int i = config.max_contacts; i < config.max_contacts *(int) 2; i++) {
        prob_connect[i] = config.prob_connect / 100;
    }
    for (int i = config.max_contacts * 2; i < config.people_num; i++) {
        prob_connect[i] = config.prob_connect;
    }
}


void CovidModel::init_gpu_memory() {
    // copy people states
    gpuErrorCheck(cudaMalloc(&d_people_states, config.people_num * sizeof(char)));
    gpuErrorCheck(cudaMemcpy(d_people_states, people_states, config.people_num * sizeof(char), cudaMemcpyHostToDevice));

    // copy window indicies
    gpuErrorCheck(cudaMalloc(&d_window_indices, config.people_num * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_window_indices, window_indices, config.people_num * sizeof(int), cudaMemcpyHostToDevice));

    // copy S transition states
    gpuErrorCheck(cudaMalloc(&d_s_trans_states, S_TRANS_STATE_NUM * sizeof(char)));
    gpuErrorCheck(cudaMemcpy(d_s_trans_states, s_trans_states, S_TRANS_STATE_NUM * sizeof(char), cudaMemcpyHostToDevice));

    // copy adj matrix
    gpuErrorCheck(cudaMalloc(&d_adj_matrix, window_num * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_adj_matrix, adj_matrix, window_num * sizeof(int), cudaMemcpyHostToDevice));

    // copy probability connect array to GPU memory
    gpuErrorCheck(cudaMalloc(&d_prob_connect, config.people_num * sizeof(double)));
    gpuErrorCheck(cudaMemcpy(d_prob_connect, prob_connect, config.people_num * sizeof(double), cudaMemcpyHostToDevice));

    // determine and copy thread range to GPU memory
    int thread_range = config.people_num / THREAD_NUM;
    gpuErrorCheck(cudaMalloc(&d_thread_range, sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_thread_range, &thread_range, sizeof(int), cudaMemcpyHostToDevice));

    // copy people num to GPU memory
    gpuErrorCheck(cudaMalloc(&d_people_num, sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_people_num, &config.people_num, sizeof(int), cudaMemcpyHostToDevice));

    // init constant GPU vars
    gpu_vars.beta = states.beta; gpu_vars.omega = states.omega;
    gpu_vars.max_contacts = config.max_contacts; gpu_vars.people_num = config.people_num;
    gpu_vars.who_vaccinated = config.who_vaccinated;
}


void CovidModel::print_matrix() {
    // for every person
    for (size_t i = 0; i < config.people_num; i += 1) {
        // for every window
        size_t window_idx = window_indices[i];
        for (size_t j = i; j < config.people_num; j += WINDOW_SIZE) {
            // check window end
            int window_end = WINDOW_SIZE;
            if (j + WINDOW_SIZE >= config.people_num) {
                window_end = (int)(config.people_num - j);
            }
            // decode bits
            for (size_t bit = 1; bit <= window_end; bit++) {
                std::cout << get_required_bit(adj_matrix[window_idx], bit);
            }
            window_idx++;
        }
        std::cout << "\n";
    }
}


std::map<char, double> CovidModel::get_states() {
    std::map<char, size_t> all_states = { {S_STATE, 0}, {E_STATE, 0}, {I_STATE, 0}, {R_STATE, 0},
                                                 {V_STATE, 0}, {D_STATE, 0}, {M_STATE, 0} };
    for (size_t i = 0; i < config.people_num; i++) {
        ++all_states[people_states[i]];
    }

    std::map<char, double> states_in_perc{};
    for (auto& pair : all_states) {
        states_in_perc[pair.first] = pair.second / (double)config.people_num;
    }
    return states_in_perc;
}


void CovidModel::covid_model() {
    config.start_vaccine = config.days;

    if (!config.is_lockdown) {
        limit_amount_of_r = 1;
    }

    int lockdown = 0;
    for (size_t day = 0; day < config.days; day++) {
        if (day % BUILD_TIMESTEP == 0) {
            std::cout << "Day: " << day << std::endl;
            build_matrix();
        }
        // Start Lockdown
        if (check_lockdown == max_increasing) {
            config.prob_connect = 0.0;
            build_matrix();
            check_lockdown = 0;
            lockdown = day;
        }
        // End of Lockdown
        if (day == lockdown + LOCKDOWN_DURATION) {
            config.prob_connect = 100 / (double)config.people_num;
            build_matrix();
            lockdown = 0;
        }

        if ((config.when_vaccinated == -1) && (config.start_vaccine == config.days) && (check_vaccine == max_point_vaccine / 2)) {
            config.start_vaccine = day + 1;
            std::cout << "Start vaccine on day " << day + 1 << std::endl;
        }
        else if ((config.when_vaccinated == 0) && (config.start_vaccine == config.days) && (check_vaccine == max_point_vaccine)) {
            config.start_vaccine = day + 1;
            std::cout << "Start vaccine on day " << day + 1 << std::endl;
        }
        else if ((config.when_vaccinated == 1) && (config.start_vaccine == config.days) && (decreasing_of_r == (int)(max_point_vaccine / 1.4))) {
            config.start_vaccine = day + 1;
            std::cout << "Start vaccine on day " << day + 1 << std::endl;
        }

        char* temp_states = new char[config.people_num];
        memcpy(temp_states, people_states, config.people_num * sizeof(char));
        run_simulation(temp_states, day);

        perc_of_people_each_state.push_back(get_states());

        if (!lockdown && day > 0) {
            if ((perc_of_people_each_state[day][R_STATE] - perc_of_people_each_state[day - 1][R_STATE]) > limit_amount_of_r) {
                ++check_lockdown;
            }
        }
        else if (!lockdown) {
            check_lockdown = 0;
        }

        if ((day > 0) && ((perc_of_people_each_state[day][R_STATE] - perc_of_people_each_state[day - 1][R_STATE]) > limit_r_for_vaccine)) {
            ++check_vaccine;
        }
        if ((day > 0) && ((perc_of_people_each_state[day - 1][R_STATE] - perc_of_people_each_state[day][R_STATE]) > limit_r_for_vaccine)) {
            ++decreasing_of_r;
        }

        if (!perc_of_people_each_state[day][E_STATE]) {
            int rand_person = randint(0, config.people_num - 1);
            temp_states[rand_person] = E_STATE;
        }
        if (!perc_of_people_each_state[day][I_STATE]) {
            int rand_person = randint(0, config.people_num - 1);
            temp_states[rand_person] = I_STATE;
        }
        people_states = temp_states;
        gpuErrorCheck(cudaMemcpy(d_people_states, people_states, config.people_num * sizeof(char), cudaMemcpyHostToDevice));
    }
}


char CovidModel::get_next_state(char curr_state, double trans_prob, char next_state) {
    std::map<int, char> trans_states = {
        {0, curr_state},
        {1, next_state},
        {2, M_STATE}
    };

    double rand_f = randfloat();
    int tr_state = 0;

    bool a = rand_f < trans_prob;
    bool b = HAS_FLAG(curr_state) && trans_prob <= rand_f && rand_f <= trans_prob + states.gamma;

    tr_state += b;
    tr_state = (tr_state << 1) + a;
    return trans_states[ceil(log2(tr_state + 1))];
}


std::vector<std::map<char, double>> CovidModel::get_state_percentages() {
    return perc_of_people_each_state;
}


CovidModel::~CovidModel() {
    cudaFree(d_adj_matrix); cudaFree(d_window_indices); cudaFree(d_people_states);
    cudaFree(d_people_num); cudaFree(d_thread_range); cudaFree(d_s_trans_states);
    cudaFree(d_prob_connect);

    free(adj_matrix);
    free(window_indices);
    free(prob_connect);
    free(people_states);
}