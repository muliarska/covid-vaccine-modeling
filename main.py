import covid_model
import numpy as np
from matplotlib import pyplot as plt
STATES = ["S", "E", "I", "R", "V", "D", "M"]


def read_config(filename):

    fi = 1 / 120  # from M to S
    gamma = 1 / 14  # from EIR to M
    alpha = 0.2  # from E to I
    beta = 0.4  # from S to E == contact rate
    sigma = alpha  # from I to R
    omega = 0.002 * 0.9  # from S to V, кількість вакцинованих за день * якість вакцини
    delta = 1 / 100  # from R to D ?
    theta = 1 / 100  # from V to S, тривалість дії вакцини
    lambda_ = 0 # from D to S

    return fi, gamma, alpha, beta, sigma, omega, delta, theta, lambda_

# 1) локдаун
# 2) різний початок вакцинації
# 3) доступність вакцини фізично і психологічно

def average_plot(number_of_simulations, number_people, days, start_vacine, is_lockdown, who_vaccinated):
    array_of_states = []
    result = []

    fi, gamma, alpha, beta, sigma, omega, delta, theta, lambda_ = read_config('config.txt')

    for _ in range(number_of_simulations):
        temp_model = covid_model.CovidModel(number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta, lambda_)
        temp_model.build_matrix()
        if not is_lockdown:
            temp_model.limit_amount_of_r = 1
        temp_model.covid_model(start_vacine, who_vaccinated)
        array_of_states.append(temp_model.number_of_people_each_state)

    for i in range(days):
        result.append([np.mean([array_of_states[j][i][k] for j in range(number_of_simulations)]) for k in range(7)])

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


def compare_state_for_lockdown(number_of_simulations, number_people, days, start_vacine, state_number):
    t, u = average_plot(number_of_simulations, number_people, days, start_vacine, True, 1)
    plt.plot(t, u[:, state_number], label=STATES[state_number] + " with lockdown")

    t, u = average_plot(number_of_simulations, number_people, days, start_vacine, False, 1)
    plt.plot(t, u[:, state_number], label=STATES[state_number] + " without lockdown")

    plt.legend()
    plt.show()


def compare_state_for_vaccine(number_of_simulations, number_people, days, state_number):
    start_vaccine = 100
    while start_vaccine < days:
        t, u = average_plot(number_of_simulations, number_people, days, start_vaccine, True, 1)
        plt.plot(t, u[:, state_number], label=STATES[state_number] + " vaccine start: " + str(start_vaccine))
        start_vaccine += 100

    plt.legend()
    plt.show()


def compare_vaccine_max_min_contacts(number_of_simulations, number_people, days, state_number, start_vaccine):
    t, u = average_plot(number_of_simulations, number_people, days, start_vaccine, True, 1)
    plt.plot(t, u[:, state_number], label=STATES[state_number] + " vaccinated max connection start: " + str(start_vaccine))

    t, u = average_plot(number_of_simulations, number_people, days, start_vaccine, True, 0)
    plt.plot(t, u[:, state_number], label=STATES[state_number] + " vaccinated min connection start: " + str(start_vaccine))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    number_people = 1000
    days = 400
    start_vaccine = 100
    # is_lockdown = True
    
    number_of_simulations = 1
    
    # t, u = average_plot(number_of_simulations, number_people, days, start_vacine, is_lockdown)
    # plot(t, u)

    state_numb = 3
    # compare_state_for_vaccine(number_of_simulations, number_people, days, state_numb)
    compare_vaccine_max_min_contacts(number_of_simulations, number_people, days, state_numb, start_vaccine)