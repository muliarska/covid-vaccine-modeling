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
    change_config_line("../config.dat", "is_lockdown", 1)

    t_1, u_1 = run_model(num_of_simulations)
    plt.plot(t_1, u_1[:, state_number], label=STATES[state_number] + " with lockdown")

    change_config_line("../config.dat", "is_lockdown", 0)

    t_2, u_2 = run_model(num_of_simulations)
    plt.plot(t_2, u_2[:, state_number], label=STATES[state_number] + " without lockdown")

    plt.legend()
    plt.show()

    plt.savefig("../plots/lockdown.png")


def compare_state_for_vaccine(num_of_simulations, days, state_number):
    start_vaccine = 100

    while start_vaccine < days:
        change_config_line("../config.dat", "start_vaccine", start_vaccine)

        t, u = run_model(num_of_simulations)
        plt.plot(t, u[:, state_number], label=STATES[state_number] + " vaccine start: " + str(start_vaccine))

        start_vaccine += 100

    plt.legend()
    # plt.show(

    plt.savefig("../plots/vaccine.png")


def compare_vaccine_max_min_contacts(num_of_simulations, state_number):
    change_config_line("../config.dat", "who_vaccinated", 1)
    t_1, u_1 = run_model(num_of_simulations)
    plt.plot(t_1, u_1[:, state_number], label=STATES[state_number] + " vaccinated max connection ")
    plt.plot(t_1, u_1[:, 5], label=STATES[5] + " vaccinated max connection ")

    change_config_line("../config.dat", "who_vaccinated", 0)
    t_2, u_2 = run_model(num_of_simulations)
    plt.plot(t_2, u_2[:, state_number], label=STATES[state_number] + " vaccinated min connection ")
    plt.plot(t_2, u_2[:, 5], label=STATES[5] + " vaccinated min connection ")

    change_config_line("../config.dat", "who_vaccinated", -1)
    t_3, u_3 = run_model(num_of_simulations)
    plt.plot(t_3, u_3[:, state_number], label=STATES[state_number] + " vaccinated random connection ")
    plt.plot(t_3, u_3[:, 5], label=STATES[5] + " vaccinated random connection ")

    plt.legend()
    # plt.show()
    plt.savefig("../plots/max_min_random.png")


def compare_vaccine_on_peaks(num_of_simulations, state_number):
    change_config_line("../config.dat", "when_vaccinated", -1)
    t_1, u_1 = run_model(num_of_simulations)
    plt.plot(t_1, u_1[:, state_number], label=STATES[state_number] + " vaccinated on middle of the peak ")
    plt.plot(t_1, u_1[:, 5], label=STATES[5] + " vaccinated on middle of the peak ")

    change_config_line("../config.dat", "when_vaccinated", 0)
    t_1, u_1 = run_model(num_of_simulations)
    plt.plot(t_1, u_1[:, state_number], label=STATES[state_number] + " vaccinated on the peak ")
    plt.plot(t_1, u_1[:, 5], label=STATES[5] + " vaccinated on the peak ")

    change_config_line("../config.dat", "when_vaccinated", 1)
    t_1, u_1 = run_model(num_of_simulations)
    plt.plot(t_1, u_1[:, state_number], label=STATES[state_number] + " vaccinated on middle of the descent ")
    plt.plot(t_1, u_1[:, 5], label=STATES[5] + " vaccinated on middle of the descent ")

    plt.legend()
    # plt.show(

    plt.savefig("../plots/peaks.png")


if __name__ == '__main__':

    # compare_state_for_lockdown(1, 3)
    # compare_state_for_vaccine(1, 300, 3)
    # compare_vaccine_max_min_contacts(3, 3)
    compare_vaccine_on_peaks(1, 3)

