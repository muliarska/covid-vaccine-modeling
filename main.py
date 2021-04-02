import covid_model
import numpy as np
from matplotlib import pyplot as plt

def read_config(filename):
    with open(filename) as file:
        fi_arr = file.readline().split()[2].split('/')
        fi = float(fi_arr[0])/float(fi_arr[1])
        gamma_arr = file.readline().split()[2].split('/')
        gamma = float(gamma_arr[0])/float(gamma_arr[1])
        alpha = float(file.readline().split()[2])
        beta = float(file.readline().split()[2])
        sigma = float(file.readline().split()[2])
        omega_arr = file.readline().split()[2].split('*')
        omega = float(omega_arr[0])*float(omega_arr[1])
        delta_arr = file.readline().split()[2].split('/')
        delta = float(delta_arr[0])/float(delta_arr[1])
        theta_arr = file.readline().split()[2].split('/')
        theta = float(theta_arr[0])/float(theta_arr[1])

    return fi, gamma, alpha, beta, sigma, omega, delta, theta

        



def average_plot(number_of_simulations, number_people, days):
    array_of_states = []
    result = []

    fi, gamma, alpha, beta, sigma, omega, delta, theta = read_config('config.txt')

    for _ in range(number_of_simulations):
        temp_model = covid_model.CovidModel(number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta)
        temp_model.run_simulation()
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
    number_people = 100
    days = 250
    
    number_of_simulations = 100
    
    average_plot(number_of_simulations, number_people, days)
    # read_config('config.txt')