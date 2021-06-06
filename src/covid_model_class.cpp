#include "../inc/base_header.hpp"
#include "../inc/utils.hpp"


CovidModel::CovidModel(config_t &cfg, states_t &st) {
    config = cfg;
    states = st;
    config.prob_connect = 100.0 / (double)config.people_num; // TODO: add to config
    config.max_contacts = (long) (config.people_num * 0.01);
    set_up_states();
    init_transition_states();
}


/* Assign S state to not infected people and E state to the infected ones. */
void CovidModel::set_up_states() {
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


std::vector<double> CovidModel::init_prob_connect() {
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
    return prob_connect_array;
}


void CovidModel::build_matrix() {
    std::vector<double> prob_connect_array = init_prob_connect();
    std::vector<std::vector<unsigned int>> matrix{};
    matrix.reserve(config.people_num);

    for(size_t n = 0; n < config.people_num; n++) {
        std::vector<unsigned int> row{};
        row.reserve(ceil((double)(config.people_num - n) / WINDOW_SIZE));

        for(size_t m = n; m < config.people_num; m += WINDOW_SIZE) {
            int window_end = WINDOW_SIZE;
            // Consider the case when we have less than 32 entries to encode in bits
            // In this case, we want to shift all our bits to the front (make them "older")
            if (m + WINDOW_SIZE >= config.people_num) {
                window_end = (int)(config.people_num - m);
            }
            // Encode entries
            unsigned int encoded_bits = 0;
            for (size_t entry = 1; entry < window_end; entry++) {
                if (randfloat() < prob_connect_array[n]) {
                    encoded_bits = (encoded_bits << 1) + 1;
                } else {
                    encoded_bits <<= 1;
                }
            }
            // Shift all our bits to the front (make them "older") in special case
            if (m + WINDOW_SIZE >= config.people_num) {
                encoded_bits <<= WINDOW_SIZE - window_end;
            }
            row.push_back(encoded_bits);
        }
        matrix.push_back(row);
    }
    adj_matrix = matrix;
}


void CovidModel::print_matrix() {
    // for every row
    for(size_t n = 0; n < config.people_num; n++) {
        // for every window
        int window_idx = 0;  // we need to keep track of our entries as we have shifting windows
        for(size_t m = n; m < config.people_num; m += WINDOW_SIZE) {
            // check for special case
            int window_end = WINDOW_SIZE;
            if (m + WINDOW_SIZE >= config.people_num) {
                window_end = (int)(config.people_num - m);
            }
            // decode bits
            for (size_t bit = 1; bit <= window_end; bit++) {
                std::cout << get_required_bit(adj_matrix[n][window_idx], bit) << " ";
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
            int rand_person = randint(0l, config.people_num - 1);
            temp_states[rand_person] = E_STATE;
        }
        if (not perc_of_people_each_state[day][I_STATE]) {
            int rand_person = randint(0l, config.people_num - 1);
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
    // USE FORMULA
    for(size_t upper_neighbour = 0; upper_neighbour < person; upper_neighbour++) {
        int connection = get_col_required_bit(adj_matrix, person, upper_neighbour);
        if (connection && (people_states[upper_neighbour] == E_STATE || people_states[upper_neighbour] == I_STATE)) {
            infected += 1;
        } else if (connection) {
            not_infected += 1;
        }
    }

    int window_idx = 0;
    for(size_t lower_neighbour = person; lower_neighbour < config.people_num; lower_neighbour += WINDOW_SIZE) {
        // check for special case
        int window_end = WINDOW_SIZE;
        if (lower_neighbour + WINDOW_SIZE >= config.people_num) {
            window_end = (int)(config.people_num - lower_neighbour);
        }
        // decode bits
        for (size_t bit = 1; bit <= window_end; bit++) {
            int connection = get_required_bit(adj_matrix[person][window_idx], bit);
            if (connection && (people_states[lower_neighbour] == E_STATE || people_states[lower_neighbour] == I_STATE)) {
                infected += 1;
            } else if (connection) {
                not_infected += 1;
            }
        }
        window_idx++;
    }
    return std::make_pair(infected, not_infected);
}


std::vector<std::map<char, double>> CovidModel::get_state_percentages() {
    return perc_of_people_each_state;
}
