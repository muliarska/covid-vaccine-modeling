import covid_model
import numpy as np
from matplotlib import pyplot as plt

def read_config(filename):

    fi = 1 / 120  # from M to S
    gamma = 1 / 14  # from EIR to M
    alpha = 0.2  # from E to I
    beta = 0.4  # from S to E == contact rate
    sigma = alpha  # from I to R
    omega = 0.002 #* 0.9  # from S to V, кількість вакцинованих за день * якість вакцини
    delta = 1 / 50  # from R to D
    theta = 1 / 100  # from V to S, тривалість дії вакцини

    return fi, gamma, alpha, beta, sigma, omega, delta, theta


def average_plot(number_of_simulations, number_people, days, start_vacine):
    array_of_states = []
    result = []

    fi, gamma, alpha, beta, sigma, omega, delta, theta = read_config('config.txt')

    for _ in range(number_of_simulations):
        temp_model = covid_model.CovidModel(number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta)
        temp_model.run_simulation(start_vacine)
        array_of_states.append(temp_model.number_of_people_each_state)

    for i in range(days):
        result.append([np.mean([array_of_states[j][i][k] for j in range(number_of_simulations)]) for k in range(7)])

    time_steps = np.linspace(0, days, days)
    t = np.asarray(time_steps)
    u = np.asarray(result)


    plt.plot(t, u[:, 0], label="S")
    plt.plot(t, u[:, 1], label="E")
    plt.plot(t, u[:, 2], label="I")
    plt.plot(t, u[:, 3], label="R")
    plt.plot(t, u[:, 4], label="V")
    plt.plot(t, u[:, 5], label="D")
    plt.plot(t, u[:, 6], label="M")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    number_people = 500
    days = 600
    start_vacine = 200
    
    number_of_simulations = 2
    
    average_plot(number_of_simulations, number_people, days, start_vacine)
