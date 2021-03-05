import random
import math
from config import person_states, build_coeffs


number_people = 100
days_of_infection, people_states, coeffs_infected, coeffs_dead = build_coeffs(number_people)


def build_matrix():
    # будується матриця з звязками

    matrix = [[0 for _ in range(number_people)] for _ in range(number_people)]
    min_connections = 2
    max_connections = math.sqrt(number_people)

    p = 0.1
    # вводимо параметр пе що керує кількістю звязків (почитати про модель)

    for i in range(number_people):
        if people_states[i] == 'd':
            clear_connections(matrix, i)
            continue

        coeff_connections = random.uniform(min_connections, max_connections) / number_people

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


def clear_connections(matrix, person):
    # очищає звязки для мертвих людей
    for lst in matrix:
        lst[person] = 0
    matrix[person] = [0 for _ in range(len(matrix))]


def main():

    matrix = build_matrix()

    for day in range(100):
        # print_matrix(matrix)
        if day % 10 == 0:
            print(people_states)
            print("\n")

        if day % 10 == 0:
            # змінює звязки кожні 10 днів
            matrix = build_matrix()

        for person in range(number_people):

            if people_states[person] == 's':
                for indx, numb in enumerate(matrix[person]):
                    if numb and (people_states[indx] == 'i'):
                        # здорова людина контактує з хворим
                        random_numb = random.random()
                        if random_numb < coeffs_infected[person]:
                            if random_numb < (0.5 + (coeffs_dead[person]*10)):
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
                    if random_numb < coeffs_dead[person]:
                        # хвора людина помирає
                        people_states[person] = 'd'
                        clear_connections(matrix, person)
                    else:
                        days_of_infection[person] += 1

            elif people_states[person] == 'r':
                if days_of_infection[person] == 120:
                    # в людини закінчується імунітет
                    people_states[person] = 's'
                else:
                    days_of_infection[person] += 1

    print(people_states)


if __name__ == '__main__':
    main()
