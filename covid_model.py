import random
import numpy as np
from matplotlib import pyplot as plt

# people_states = []

# S E I R V D M
'''
S' = -beta*(E + I)*S + fi*M - omega*S + theta*V
E' = beta*(E + I)*S - (gamma + alpha)E
I' = alpha*E - (gamma + sigma)I
R' = sigma*I - gamma*R - delta*R
V' = omega*S - theta*V
D' = delta*R
M' = gamma*(E + I + R) - fi*M
M = імунний
'''


class CovidModel:
    def __init__(self, number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta):
        self.number_people = number_people
        self.days = days

        self.prob_connect = beta

        self.beta = beta
        self.omega = omega
        self.fi = fi
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.delta = delta
        self.theta = theta

        self.number_of_people_each_state = np.zeros((days, 7))

        self.set_up_states()

    def set_up_states(self):
        number_infected = int(0.1 * self.number_people)
        self.people_states = ['s' for _ in range(self.number_people-number_infected)]
        self.people_states.extend(['e' for _ in range(number_infected)])

    def build_matrix(self):
        # будується матриця з звязками
        self.matrix = [[0 for _ in range(self.number_people)] for _ in range(self.number_people)]

        for i in range(self.number_people):
            # всі зі всіма

            # self.matrix[i] = [1 for _ in range(self.number_people)]
            # self.matrix[i][i] = 0

            for x in range(i):
                self.matrix[i][x] = self.matrix[x][i]
            
            for j in range(i + 1, self.number_people):
                random_numb = random.random()
                if random_numb < self.prob_connect:
                    self.matrix[i][j] = 1

    def print_matrix(self):
        for m in self.matrix:
            print(m)
        print("\n")

    def get_states(self):

        sum_s = 0
        sum_e = 0
        sum_i = 0
        sum_r = 0
        sum_v = 0
        sum_d = 0
        sum_m = 0

        for i in range(self.number_people):
            if self.people_states[i] == 's':
                sum_s += 1
            elif self.people_states[i] == 'e':
                sum_e += 1
            elif self.people_states[i] == 'i':
                sum_i += 1
            elif self.people_states[i] == 'r':
                sum_r += 1
            elif self.people_states[i] == 'v':
                sum_v += 1
            elif self.people_states[i] == 'd':
                sum_d += 1
            elif self.people_states[i] == 'm':
                sum_m += 1

        return np.asarray([sum_s/self.number_people, sum_e/self.number_people,
                           sum_i/self.number_people, sum_r/self.number_people,
                           sum_v/self.number_people, sum_d/self.number_people,
                           sum_m/self.number_people])
        # return np.asarray([sum_s, sum_e,
        #                    sum_i, sum_r,
        #                    sum_v, sum_d,
        #                    sum_m])

    def covid_model(self, numb_of_days):

        for day in range(numb_of_days):

            if day % 10 == 0:
                # змінює звязки кожні 10 днів
                # print(self.people_states)
                # print("\n")
                self.build_matrix()

            temporary_states = self.people_states.copy()

            for person in range(self.number_people):

                if self.people_states[person] == 's':
                    random_numb = random.random()
                    infected = 0
                    not_infected = 0
                    for ind in range(self.number_people):
                        if (self.people_states[ind] == 'e') or (self.people_states[ind] == 'i'):
                            # здорова людина контактує з хворим
                            infected += 1
                        else:
                            not_infected += 1
                    beta = self.beta * (infected/(not_infected+infected))
                    if random_numb < beta:
                        temporary_states[person] = 'e'
                        break
                    if (random_numb >= beta) and (random_numb <= beta + self.omega):
                        temporary_states[person] = 'v'

                elif self.people_states[person] == 'e':
                    random_numb = random.random()
                    if random_numb < self.alpha:
                        temporary_states[person] = 'i'
                    elif (self.alpha <= random_numb) and (random_numb <= self.alpha + self.gamma):
                        self.people_states[person] = 'm'

                elif self.people_states[person] == 'i':
                    random_numb = random.random()
                    if random_numb < self.sigma:
                        temporary_states[person] = 'r'
                    elif (self.sigma <= random_numb) and (random_numb <= self.sigma + self.gamma):
                        temporary_states[person] = 'm'

                elif self.people_states[person] == 'r':
                    random_numb = random.random()
                    if random_numb < self.delta:
                        temporary_states[person] = 'd'
                    elif (self.delta <= random_numb) and (random_numb <= self.delta + self.gamma):
                        temporary_states[person] = 'm'

                elif self.people_states[person] == 'v':
                    random_numb = random.random()
                    if random_numb < self.theta:
                        temporary_states[person] = 's'

                elif self.people_states[person] == 'm':
                    random_numb = random.random()
                    if random_numb < self.fi:
                        temporary_states[person] = 's'


            self.number_of_people_each_state[day] = self.get_states()
            self.people_states = temporary_states

    def run_simulation(self):
        self.build_matrix()
        self.covid_model(self.days)

    def plot(self):
        self.run_simulation()

        time_steps = np.linspace(0, self.days, self.days)
        t = np.asarray(time_steps)
        u = np.asarray(self.number_of_people_each_state)

        plt.plot(t, u[:, 0], label="S")
        plt.plot(t, u[:, 1], label="E")
        plt.plot(t, u[:, 2], label="I")
        plt.plot(t, u[:, 3], label="R")
        plt.plot(t, u[:, 4], label="V")
        plt.plot(t, u[:, 5], label="D")
        plt.plot(t, u[:, 6], label="M")

    

        return plt


if __name__ == '__main__':

    fi = 1 / 120  # from V to S
    gamma = 1 / 14  # from EIR to V
    alpha = 0.2  # from E to I
    beta = 0.3  # from S to E == contact rate
    sigma = alpha  # from I to R
    omega = 0.008 * 0.9  # from S to V, кількість вакцинованих за день * якість вакцини  
    delta = 1 / 50  # from R to D
    theta = 1/365  # from V to S, тривалість дії вакцини

    number_people = 1000
    days = 1000
    

    model = CovidModel(number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta)
    model.plot()