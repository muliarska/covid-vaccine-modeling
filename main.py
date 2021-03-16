import random
import math
from config import person_states, build_coeffs


number_people = 1000

percent_vaccined = 1/700

days_of_infection, people_states, probability_get_infected, probability_get_dead = build_coeffs(number_people)


def build_matrix():
    # будується матриця з звязками

    matrix = [[0 for _ in range(number_people)] for _ in range(number_people)]
    min_connections = 2
    max_connections = math.sqrt(number_people)

    p = 20/number_people
    # вводимо параметр пе що керує кількістю звязків (почитати про модель)

    for i in range(number_people):
        if people_states[i] == 'd':
            clear_connections(matrix, i)
            continue

        # coeff_connections = random.uniform(min_connections, max_connections) / number_people

        for x in range(i):
            matrix[i][x] = matrix[x][i]

        for j in range(i + 1, number_people):
            random_numb = random.random()
            if random_numb < p:
                matrix[i][j] = 1

    return matrix


def print_matrix(matrix):
    for m in matrix:
        print(m)
    print("\n")


def get_states():
    sum_infected = 0
    sum_recovered = 0
    sum_susceptible = 0
    sum_dead = 0

    for i in range(number_people):
        if people_states[i] == 'i' or people_states[i] == 'is':
            sum_infected += 1
        elif people_states[i] == 'r':
            sum_recovered += 1
        elif people_states[i] == 's':
            sum_susceptible += 1
        else:
            sum_dead += 1

    print(sum_infected)
    print(sum_dead)
    print(sum_susceptible)
    print(sum_recovered)




def covid_model(numb_of_days, if_vaccine):

    matrix = build_matrix()
    # print_matrix(matrix)

    for day in range(numb_of_days):
        # print_matrix(matrix)
        if day % 10 == 0:
            get_states()
            print("\n")

        if day % 10 == 0:
            # змінює звязки кожні 10 днів
            matrix = build_matrix()


        vaccined_people = 0
        needed_vaccined_people = percent_vaccined * number_people


        for person in range(number_people):

            if if_vaccine and vaccined_people < needed_vaccined_people:
                if people_states[person] != 'v':
                    people_states[person] = 'v'
                    vaccined_people += 1

            if people_states[person] == 's':
                for indx, numb in enumerate(matrix[person]):
                    if numb and (people_states[indx] == 'i'):
                        # здорова людина контактує з хворим
                        random_numb = random.random()
                        if random_numb < probability_get_infected:
                            if random_numb < 0.5:
                                # людина йде на ізоляцію
                                people_states[person] = 'is'
                            else:
                                people_states[person] = 'i'
                            break

            elif people_states[person] == 'i' or people_states[person] == 'is':
                if days_of_infection[person] == 21:
                    # хвора людина видужує та отримує імунітет
                    people_states[person] = 'r'
                    days_of_infection[person] = 0
                else:
                    random_numb = random.random()
                    if random_numb < probability_get_dead:
                        # хвора людина помирає
                        people_states[person] = 'd'
                        # clear_connections(matrix, person)
                    else:
                        days_of_infection[person] += 1

            elif people_states[person] == 'r':
                if days_of_infection[person] == 120:
                    # в людини закінчується імунітет
                    people_states[person] = 's'
                else:
                    days_of_infection[person] += 1

    # print(people_states)

