#include "../inc/base_header.hpp"
#include "../inc/utils.hpp"


CovidModel::CovidModel(config_t &cfg, states_t &st) {
    config = cfg;
    states = st;
    config.prob_connect = 100.0 / (double)config.people_num; // TODO: add to config
    config.max_contacts = (long) (config.people_num * 0.01);

    window_indices = get_window_indices(config.people_num);
    init_states();
    init_transition_states();
    init_prob_connect();
}


/* Assign S state to not infected people and E state to the infected ones. */
void CovidModel::init_states() {
    long infected_num = (long)(INFECT_PROB * (double)config.people_num);

    people_states = initialize_array(S_STATE, config.people_num - infected_num);
    std::vector<char> infected = initialize_array(E_STATE, infected_num);

    people_states.insert(people_states.end(),infected.begin(),infected.end());
    perc_of_people_each_state.reserve(config.days);
}


void CovidModel::init_transition_states() {
    std::unordered_map<char, std::pair<char, double>>
    transition{{E_STATE, std::make_pair(I_STATE, states.alpha)},
               {I_STATE, std::make_pair(R_STATE, states.sigma)},
               {R_STATE, std::make_pair(D_STATE, states.delta)},
               {V_STATE, std::make_pair(S_STATE, states.theta)},
               {M_STATE, std::make_pair(S_STATE, states.fi)},
               {D_STATE, std::make_pair(S_STATE, states.lambda)}};
    transition_states = transition;
}


void CovidModel::init_prob_connect() {
    std::vector<double> prob_connect_array;
    for (size_t i = 0; i < config.max_contacts; i++) {
        prob_connect_array.push_back(config.prob_connect * 100);
    }
    for (size_t i = config.max_contacts; i < config.max_contacts * 2; i++) {
        prob_connect_array.push_back(config.prob_connect / 100);
    }
    for (size_t i = config.max_contacts * 2; i < config.people_num; i++) {
        prob_connect_array.push_back(config.prob_connect);
    }

    prob_connect = prob_connect_array;
}


void CovidModel::run_threads(int threadIdx, int blockIdx, int thread_range, int* matrix) {
    // determine thread boundaries
    int start = blockIdx + threadIdx * thread_range;
    int end = start + thread_range;

    int window_idx = window_indices[blockIdx] + threadIdx;
    for(size_t m = start; m < end; m += WINDOW_SIZE) {
        // let the spare bits of the last window be encoded
        // as we don't care about what values these bits have
        int encoded_bits = 0;
        for (size_t entry = 1; entry < WINDOW_SIZE; entry++) {
            encoded_bits = (encoded_bits << 1) + (randfloat() < prob_connect[blockIdx]);
        }
        matrix[window_idx] = encoded_bits;
        window_idx += 1;
    }
}


void CovidModel::build_matrix() {
    // free dynamically allocated previous matrix as it is invalid
    free(adj_matrix);
    // set up new matrix
    size_t window_num = calc_window_num(config.people_num);
    auto* matrix = new int[window_num];
    // simulate multiple thread executing
    for(size_t blockIdx = 0; blockIdx < config.people_num; blockIdx++) {
        size_t threads = config.people_num / THREAD_RANGE;

        for(size_t threadIdx = 0; threadIdx < threads; threadIdx += 1) {
            run_threads(threadIdx, blockIdx, THREAD_RANGE, matrix);
        }
    }
    adj_matrix = matrix;
}


void CovidModel::print_matrix() {
    // for every person
    for(size_t i = 0; i < config.people_num; i += 1) {
        // for every window
        size_t window_idx = window_indices[i];
        for(size_t j = i; j < config.people_num; j+= WINDOW_SIZE) {
            // check window end
            int window_end = WINDOW_SIZE;
            if (j + WINDOW_SIZE >= config.people_num) {
                window_end = (int)(config.people_num - j);
            }
            // decode bits
            for(size_t bit = 1; bit <= window_end; bit++) {
                std::cout << get_required_bit(adj_matrix[window_idx], bit);
            }
            window_idx++;
        }
        std::cout << "\n";
    }
}


std::map<char, double> CovidModel::get_states() {
    std::map<char, size_t> all_states = {{S_STATE, 0}, {E_STATE, 0}, {I_STATE, 0}, {R_STATE, 0},
                                                 {V_STATE, 0}, {D_STATE, 0}, {M_STATE, 0}};
    for(size_t i = 0; i < config.people_num; i++) {
        ++all_states[people_states[i]];
    }

    std::map<char, double> states_in_perc{};
    for(auto &pair: all_states) {
        states_in_perc[pair.first] = pair.second / (double)config.people_num;
    }
    return states_in_perc;
}


void CovidModel::covid_model() {
    config.start_vaccine = config.days;

    if (not config.is_lockdown) {
        limit_amount_of_r = 1;
    }

    int lockdown = 0;
    for(size_t day = 0; day < config.days; day++) {
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
            config.prob_connect = 100 / (double) config.people_num;
            build_matrix();
            lockdown = 0;
        }

        if ((config.when_vaccinated == -1) && (config.start_vaccine == config.days) && (check_vaccine == max_point_vaccine / 2)) {
            config.start_vaccine = day + 1;
            std::cout<< "Start vaccine on day " << day+1 << std::endl;
        }
        else if ((config.when_vaccinated == 0) && (config.start_vaccine == config.days) && (check_vaccine == max_point_vaccine)) {
            config.start_vaccine = day + 1;
            std::cout<< "Start vaccine on day " << day+1 << std::endl;
        }
        else if ((config.when_vaccinated == 1) && (config.start_vaccine == config.days) && (decreasing_of_r == (int) (max_point_vaccine / 1.4))) {
            config.start_vaccine = day + 1;
            std::cout<< "Start vaccine on day " << day+1 << std::endl;
        }

        std::vector<char> temp_states = people_states;
        run_simulation(temp_states, day);

        perc_of_people_each_state.push_back(get_states());

        if (not lockdown && day > 0) {
            if ((perc_of_people_each_state[day][R_STATE] - perc_of_people_each_state[day - 1][R_STATE]) > limit_amount_of_r) {
                ++check_lockdown;
            }
        } else if (not lockdown) {
            check_lockdown = 0;
        }

        if ((day > 0) && ((perc_of_people_each_state[day][R_STATE] - perc_of_people_each_state[day-1][R_STATE]) > limit_r_for_vaccine)) {
            ++check_vaccine;
        }
        if ((day > 0) && ((perc_of_people_each_state[day-1][R_STATE] - perc_of_people_each_state[day][R_STATE]) > limit_r_for_vaccine)) {
            ++decreasing_of_r;
        }

        if (not perc_of_people_each_state[day][E_STATE]) {
            int rand_person = randint((size_t)0, config.people_num - 1);
            temp_states[rand_person] = E_STATE;
        }
        if (not perc_of_people_each_state[day][I_STATE]) {
            int rand_person = randint((size_t)0, config.people_num - 1);
            temp_states[rand_person] = I_STATE;
        }
        people_states = temp_states;
    }
}


char CovidModel::get_s_next_state(double beta, size_t day, size_t person) {
    double rand_f = randfloat();

    int tr_state = 0;
    bool a = rand_f < beta;
    bool b = rand_f >= beta && rand_f <= beta + states.omega && day >= config.start_vaccine;
    bool c = (config.who_vaccinated && person < config.max_contacts && day >= config.start_vaccine) || \
             (not config.who_vaccinated && config.max_contacts < person && person < (config.max_contacts * 2)
             && day >= config.start_vaccine);

    tr_state += c;
    tr_state = (tr_state << 1) + b;
    tr_state = (tr_state << 1) + a;

    return s_trans_states[ceil(log2(tr_state + 1))]; // add 1 to balance out binary numbers
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


void CovidModel::run_simulation(std::vector<char> &temp_states, size_t day) {
    std::vector<size_t> s_state_people{};
    std::vector<size_t> rest_state_people{};

    // separate people on those who are in S state from others
    for(size_t person = 0; person < config.people_num; person++) {
        if (people_states[person] == S_STATE) {
            s_state_people.push_back(person);
        } else {
            rest_state_people.push_back(person);
        }
    }

    // launch kernel for people in S state
    for(size_t person : s_state_people) {
        // check if healthy man is in contact with the infected ones
        std::pair<size_t, size_t> infection_num = get_infected_num(person);
        // examine the beta coefficicent
        size_t inf = infection_num.first, n_inf = infection_num.second;
        double beta = (inf + n_inf != 0) * (states.beta * ((double)inf / (double)(n_inf + inf)));
        // switch state
        temp_states[person] = get_s_next_state(beta, day, person);
    }

    // lanch kernel for people in other states apart from S
    for(size_t person: rest_state_people) {
        char current_state = people_states[person], next_state = transition_states[current_state].first;
        double trans_prob = transition_states[current_state].second;
        // switch state
        temp_states[person] = get_next_state(current_state, trans_prob, next_state);
    }
}


std::pair<size_t, size_t> CovidModel::get_infected_num(size_t person) {
    size_t infected = 0, not_infected = 0;
    bool is_infected;
    // USE FORMULA
    for(size_t upper_neighbour = 0; upper_neighbour < person; upper_neighbour++) {
        // determine window index and required bit of a neighbour
        size_t window_idx = window_indices[upper_neighbour] + (person - upper_neighbour) / WINDOW_SIZE;
        int connection = get_col_required_bit(adj_matrix[window_idx], person, upper_neighbour);

        is_infected = (connection && (people_states[upper_neighbour] == E_STATE || people_states[upper_neighbour] == I_STATE));

        infected += is_infected;
        not_infected += (not is_infected) && connection;
    }

    int window_idx = window_indices[person];
    for(size_t lower_neighbour = person; lower_neighbour < config.people_num; lower_neighbour += WINDOW_SIZE) {
        // check if those encoded bits are on the edge of a matrix,
        // thus define the endpoint of a window
        bool is_window_edge = lower_neighbour + WINDOW_SIZE >= config.people_num;
        int window_end = WINDOW_SIZE * (not is_window_edge) + (int)(config.people_num - lower_neighbour) * is_window_edge;
        // decode bits
        for (size_t bit = 1; bit <= window_end; bit++) {
            int connection = get_required_bit(adj_matrix[window_idx], bit);
            is_infected = (connection && (people_states[lower_neighbour] == E_STATE || people_states[lower_neighbour] == I_STATE));

            infected += is_infected;
            not_infected += (not is_infected) && connection;
        }
        window_idx++;
    }
    return std::make_pair(infected, not_infected);
}


std::vector<std::map<char, double>> CovidModel::get_state_percentages() {
    return perc_of_people_each_state;
}


CovidModel::~CovidModel() {
    free(adj_matrix);
    free(window_indices);
}
