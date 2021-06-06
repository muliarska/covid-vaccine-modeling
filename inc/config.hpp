//
// Created by Yaroslav Morozevych on 4/21/21.
//

#ifndef COVID_MODEL_CONFIG_HPP
#define COVID_MODEL_CONFIG_HPP

void read_config(std::string &filename, config_t &setup, states_t &states);
void extract_config_args(std::unordered_map<std::string, std::string> &config_data, config_t &config, states_t &states);

#endif //COVID_MODEL_CONFIG_HPP
