import subprocess
import numpy as np
from matplotlib import pyplot as plt

STATES = ["S", "E", "I", "R", "V", "D", "M"]


def read_output(path):
    with open(path, "r") as output_file:
        output = output_file.readlines()

    perc_of_people_each_state = []
    for line in output:
        states = list(map(lambda x: float(x), line.strip().split(" ")))
        right_order_states = [states[5], states[1], states[2], states[4], states[6], states[0], states[3]]
        perc_of_people_each_state.append(right_order_states)

    return perc_of_people_each_state


def run_model(num_of_simulations):
    array_of_states = []
    for i in range(num_of_simulations):
        proc = subprocess.Popen(["../build/covid_model"])
        proc.wait()
        array_of_states.append(read_output("../output.txt"))

    days = len(array_of_states[0])

    result = []
    for i in range(days):
        result.append([np.mean([array_of_states[j][i][k] for j in range(num_of_simulations)]) for k in range(7)])

    time_steps = np.linspace(0, days, days)
    t = np.asarray(time_steps)
    u = np.asarray(result)

    return t, u


def plot(t, u):
    plt.plot(t, u[:, 0], label="S")
    plt.plot(t, u[:, 1], label="E")
    plt.plot(t, u[:, 2], label="I")
    plt.plot(t, u[:, 3], label="R")
    plt.plot(t, u[:, 4], label="V")
    plt.plot(t, u[:, 5], label="D")
    plt.plot(t, u[:, 6], label="M")

    plt.legend()
    plt.show()


def change_config_line(cfg_path, name_of_line, new_amount):
    output_file = open(cfg_path, "r")
    output = output_file.readlines()

    for line in output:
        if line[:len(name_of_line)] == name_of_line:
            output[output.index(line)] = line.replace(line[len(name_of_line) + 3:-1], str(new_amount))

    output_file = open(cfg_path, "w")
    output_file.writelines(output)
    output_file.close()


def compare_state_for_lockdown(num_of_simulations, state_number):
    fig, plts = plt.subplots(len(state_number))

    change_config_line("../config.dat", "max_min_rand_vaccination", 0)
    change_config_line("../config.dat", "is_lockdown", 0)

    t_1, u_1 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_1, u_1[:, state_number[i]], label=STATES[state_number[i]] + " vaccination without lockdown")

    change_config_line("../config.dat", "is_lockdown", 1)

    t_2, u_2 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_2, u_2[:, state_number[i]], label=STATES[state_number[i]] + " vaccination with lockdown")

    change_config_line("../config.dat", "start_vaccine", 10000)

    t_3, u_3 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_3, u_3[:, state_number[i]], label=STATES[state_number[i]] + " lockdown without vaccination")

    for i in range(len(state_number)):
        plts[i].legend()

    plt.savefig("../plots/impact_of_lockdown.png")


def compare_vaccine_max_min_contacts(num_of_simulations, state_number):
    fig, plts = plt.subplots(len(state_number))

    change_config_line("../config.dat", "max_min_rand_vaccination", 1)
    t_1, u_1 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_1, u_1[:, state_number[i]], label=STATES[state_number[i]] + " vaccinated max connection ")

    change_config_line("../config.dat", "max_min_rand_vaccination", 0)
    t_2, u_2 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_2, u_2[:, state_number[i]], label=STATES[state_number[i]] + " vaccinated random connection ")

    change_config_line("../config.dat", "max_min_rand_vaccination", -1)
    t_3, u_3 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_3, u_3[:, state_number[i]], label=STATES[state_number[i]] + " vaccinated min connection ")

    for i in range(len(state_number)):
        plts[i].legend()

    plt.savefig("../plots/max_min_random.png")


def compare_vaccine_on_peaks(num_of_simulations, state_number):
    fig, plts = plt.subplots(len(state_number))

    change_config_line("../config.dat", "when_vaccinated", -1)
    t_1, u_1 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_1, u_1[:, state_number[i]], label=STATES[state_number[i]] + " vaccinated on the beginning of the peak ")

    change_config_line("../config.dat", "when_vaccinated", 0)
    t_2, u_2 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_2, u_2[:, state_number[i]], label=STATES[state_number[i]] + " vaccinated on the peak ")

    change_config_line("../config.dat", "when_vaccinated", 1)
    t_3, u_3 = run_model(num_of_simulations)
    for i in range(len(state_number)):
        plts[i].plot(t_3, u_3[:, state_number[i]], label=STATES[state_number[i]] + "vaccinated on middle of the "
                                                                                   "descent")

    for i in range(len(state_number)):
        plts[i].legend()

    plt.savefig("../plots/peaks.png")


def plot_probs(path):
    with open(path, "r") as output_file:
        output = output_file.readlines()

    probs = []
    for line in output:
        # print(".........", line, float(line))
        probs.append(float(line))

    # print(probs)
    plt.plot([i for i in range(len(probs))], probs, label="probs distribution")
    plt.legend()
    plt.savefig("../plots/test_distr")
    plt.show()


def plot_connections(path):
    with open(path, "r") as output_file:
        output = output_file.readlines()

    num_people = []
    number_con = []
    min_c = 50
    max_c = 60

    while min_c < len(output)/30:
        number_con.append((max_c+min_c)/2)
        count_con = 0
        for line in output:
            line = int(line)
            if min_c <= line < max_c:
                count_con += 1
        num_people.append(count_con)

        min_c += 10
        max_c += 10

    # plt.plot(number_con, num_people, label="connections distribution")
    # plt.legend()
    # plt.savefig("../plots/test")
    # plt.show()

    plt.bar(number_con, num_people)
    plt.xticks(number_con)
    plt.yticks(num_people)
    plt.xlabel("Number of connections")
    plt.ylabel("Number of people")

    # plt.legend()
    plt.show()


if __name__ == '__main__':
    # compare_state_for_lockdown(5, [3, 5])
    # compare_vaccine_max_min_contacts(5, [3, 5])
    compare_vaccine_on_peaks(5, [3, 5])
