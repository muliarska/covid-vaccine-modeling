import random
from config import person_states, build_coeffs


def build_matrix(number_people):
    matrix = []
    for i in range(number_people):
        person_connections = []

        for x in range(i):
            person_connections.append(matrix[x][i])

        person_connections.append(0)

        person_connections.extend([random.randint(0, 1) for j in range(i+1, number_people)])
        matrix.append(person_connections)

    return matrix

def print_matrix(matrix):
    for m in matrix:
        print(m)
    print("\n")


def clear_connections(matrix, person):
    for lst in matrix:
        lst[person] = 0
    matrix[person] = [0 for _ in range(len(matrix))]


def main():
    number_people = 100
    days_of_infection, people_states, coeffs_infected, coeffs_dead = build_coeffs(number_people)
    print(people_states)
    matrix = build_matrix(number_people)

    for _ in range(50):
        #print_matrix(matrix)

        for person in range(number_people):

            if people_states[person] == 's':
                for indx, numb in enumerate(matrix[person]):
                    if numb and (people_states[indx] == 'i'):
                        random_numb = random.random()
                        if random_numb < coeffs_infected[person]:
                            people_states[person] = 'i'
                            break

            elif people_states[person] == 'i':
                if days_of_infection[person] == 21:
                    people_states[person] = 's'
                else:
                    random_numb = random.random()
                    if random_numb < coeffs_dead[person]:
                        people_states[person] = 'd'
                        clear_connections(matrix, person)
                    else:
                        days_of_infection[person] += 1

    print(people_states)





if __name__ == '__main__':
    main()