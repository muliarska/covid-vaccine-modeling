import random
import numpy as np
from matplotlib import pyplot as plt

# people_states = []

# S E I R V D M
'''
S' = -beta*(E + I)*S + fi*M - omega*S + theta*V + lambda*D
E' = beta*(E + I)*S - (gamma + alpha)E
I' = alpha*E - (gamma + sigma)I
R' = sigma*I - gamma*R - delta*R
V' = omega*S - theta*V
D' = delta*R - lambda*D
M' = gamma*(E + I + R) - fi*M
M = імунний
'''


class CovidModel:
    def __init__(self, number_people, days, beta, omega, fi, gamma, alpha, sigma, delta, theta, lambda_):
        self.number_people = number_people
        self.days = days

        self.max_contacts = int(self.number_people * 0.01)
        
        self.prob_connect = 100/self.number_people

        self.beta = beta
        self.omega = omega
        self.fi = fi
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.delta = delta
        self.theta = theta
        self.lambda_ = lambda_

        self.number_of_people_each_state = np.zeros((days, 7))

        self.check_lockdown = 0
        self.limit_amount_of_r = 0.001
        self.max_increasing = 10
        # статистика

        self.set_up_states()

    def set_up_states(self):
        number_infected = int(0.1 * self.number_people)
        self.people_states = ['s' for _ in range(self.number_people-number_infected)]
        self.people_states.extend(['e' for _ in range(number_infected)])

    def build_matrix(self):
        # будується матриця з звязками
        self.matrix = [[0 for _ in range(self.number_people)] for _ in range(self.number_people)]

        def make_contacts(prob_connect):
            for x in range(i):
                self.matrix[i][x] = self.matrix[x][i]

            for j in range(i + 1, self.number_people):
                random_numb = random.random()
                if random_numb < prob_connect:
                    self.matrix[i][j] = 1

        for i in range(self.max_contacts):
            make_contacts(self.prob_connect * 100)

        for i in range(self.max_contacts, self.max_contacts * 2):
            make_contacts(self.prob_connect / 100)

        for i in range(self.max_contacts * 2, self.number_people):
            make_contacts(self.prob_connect)

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

    def covid_model(self, start_vacine, who_vaccinated):
        lockdown = 0

        for day in range(self.days):

            if day % 10 == 0:
                # змінює звязки кожні 10 днів
                # print(self.people_states)
                # print("\n")
                self.build_matrix()

            if self.check_lockdown == self.max_increasing:
                print("Start lockdown: {}".format(day))
                # self.prob_connect = 5 / self.number_people
                self.prob_connect = 0
                self.build_matrix()
                self.check_lockdown = 0
                lockdown = day

            if day == lockdown + 21:
                print("End lockdown: {}".format(day))
                self.prob_connect = 100 / self.number_people
                self.build_matrix()
                lockdown = 0


            temporary_states = self.people_states.copy()

            for person in range(self.number_people):

                if self.people_states[person] == 's':
                    if (who_vaccinated == 1) and (person < self.max_contacts) and (day >= start_vacine):
                        temporary_states[person] = 'v'
                    elif (who_vaccinated == 0) and (self.max_contacts < person < (self.max_contacts * 2))\
                            and (day >= start_vacine):
                        temporary_states[person] = 'v'

                    random_numb = random.random()
                    infected = 0
                    not_infected = 0
                    for ind in range(self.number_people):
                        if self.matrix[person][ind] == 1:
                            if (self.people_states[ind] == 'e') or (self.people_states[ind] == 'i'):
                                # здорова людина контактує з хворим
                                infected += 1
                            else:
                                not_infected += 1
                    if (not_infected+infected) == 0:
                        beta = 0
                    else:
                        beta = self.beta * (infected/(not_infected+infected))
                    if random_numb < beta:
                        temporary_states[person] = 'e'

                    if (random_numb >= beta) and (random_numb <= beta + self.omega)\
                            and (day >= start_vacine):
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

                elif self.people_states[person] == 'd':
                    random_numb = random.random()
                    if random_numb < self.lambda_:
                        temporary_states[person] = 's'

            self.number_of_people_each_state[day] = self.get_states()

            if lockdown == 0 and (self.number_of_people_each_state[day][3] - self.number_of_people_each_state[day-1][3]) > self.limit_amount_of_r:
                self.check_lockdown += 1
            elif lockdown == 0:
                self.check_lockdown = 0

            if self.number_of_people_each_state[day][1] == 0:
                rand_person = random.randint(0, self.number_people - 1)
                temporary_states[rand_person] = 'e'
            if self.number_of_people_each_state[day][2] == 0:
                rand_person = random.randint(0, self.number_people - 1)
                temporary_states[rand_person] = 'i'

            self.people_states = temporary_states
            if day % 50 == 0:
                print("\nDay", day)

