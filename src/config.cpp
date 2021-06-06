//
// Created by Yaroslav Morozevych on 4/21/21.
//
#include "../inc/base_header.hpp"
#include "../inc/config.hpp"


void read_config(std::string &filename, config_t &setup, states_t &states) {
    std::ifstream cfg(filename);

    if (!cfg.is_open()) {
        exit(IO_READ_ERR);
    }

    std::string line{};
    std::unordered_map<std::string, std::string> config_data;
    while (std::getline(cfg, line)) {
        /* If the line contains an equals operator -- we treat it as valid data
         * std::string::npos means "until the end of the string". */
        if (line.find('=') != std::string::npos) {
            // Split data in line into a key and value
            std::istringstream iss{ line };
            std::string key{}, value{};

            /* Operation std::getline(iss, id, ':') extracts a string from the std::istringstream
             * and assigns it to variable "id".*/
            if (std::getline(std::getline(iss, key, '=') >> std::ws, value)) {
                key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
                config_data[key] = value;
            }
        }
    }
    cfg.close();
    extract_config_args(config_data, setup, states);
}


void extract_config_args(std::unordered_map<std::string, std::string> &config_data, config_t &config, states_t &states) {
    try {
        // Read PROGRAM SETUP
        config.people_num = std::stol(config_data.at("people_num"));
        assert(config.people_num >= 0 && "<people_num> should be in range [0, inf).");

        config.days = std::stol(config_data.at("days"));
        assert(config.days >= 0 && "<days> should be in range [0, inf).");

        config.start_vaccine = std::stol(config_data.at("start_vaccine"));
        assert(config.start_vaccine >= 0 && "<start_vaccine> should be in range [0, inf).");
        
        config.who_vaccinated = std::stoi(config_data.at("who_vaccinated"));
        assert(config.who_vaccinated >= 0 &&  config.who_vaccinated <= 1 && "<who_vaccinated> should be in range [0, 1].");

        config.when_vaccinated = std::stoi(config_data.at("when_vaccinated"));
        assert(config.when_vaccinated >= -1 &&  config.when_vaccinated <= 1 && "<when_vaccinated> should be in range [-1, 1].");

        config.is_lockdown = std::stoi(config_data.at("is_lockdown"));
        assert(config.is_lockdown >= 0 &&  config.is_lockdown <= 1 && "<is_lockdown> should be in range [0, 1].");

        config.prob_connect = std::stod(config_data.at("prob_connect"));
        assert(config.prob_connect >= 0 && "<prob_connect> should be in range [0, 1].");

        // Read STATES
        states.fi = std::stod(config_data.at("fi"));
        assert(states.fi >= 0 && "<fi> should be in range [0, inf).");

        states.gamma = std::stod(config_data.at("gamma"));
        assert(states.gamma >= 0 && "<gamma> should be in range [0, inf).");

        states.alpha = std::stod(config_data.at("alpha"));
        assert(states.alpha >= 0 && "<alpha> should be in range [0, inf).");

        states.beta = std::stod(config_data.at("beta"));
        assert(states.beta >= 0 && "<beta> should be in range [0, inf).");

        states.sigma = std::stod(config_data.at("sigma"));
        assert(states.sigma >= 0 && "<sigma> should be in range [0, inf).");

        states.omega = std::stod(config_data.at("omega"));
        assert(states.omega >= 0 && "<omega> should be in range [0, inf).");

        states.delta = std::stod(config_data.at("delta"));
        assert(states.delta >= 0 && "<delta> should be in range [0, inf).");

        states.theta = std::stod(config_data.at("theta"));
        assert(states.theta >= 0 && "<theta> should be in range [0, inf).");

        states.lambda = std::stod(config_data.at("lambda"));
        assert(states.lambda >= 0 && "<lambda> should be in range [0, inf).");

    } catch (std::out_of_range &e) {
        std::cout << "Missing an argument in the config!\n" << std::endl;
        exit(CFG_VALUE_ERROR);
    } catch (std::invalid_argument &e) {
        std::cout << "Invalid argument! All config arguments should be numbers!" << std::endl;
        exit(CFG_VALUE_ERROR);
    }
}
