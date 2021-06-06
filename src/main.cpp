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


int main() {
    std::string cfg_path = "../config.dat";
    std::string out_path = "../output.txt";

    // set up srand() for random
    srand(time(NULL));

    config_t config{};
    states_t states{};
    read_config(cfg_path, config, states);

    CovidModel covid_model = CovidModel{config, states};
    covid_model.covid_model();

    std::vector<std::map<char, double>> perc_people_each_state = covid_model.get_state_percentages();
    write_results(out_path, perc_people_each_state, config);
    return 0;
}