import random

person_states = ['s', 'i', 'r', 'd', 'v']

# coeffs_recovered = []
# coeffs_vaccinated


def build_coeffs(number_people):
    days_of_infection = [0 for _ in range(number_people)]
    # рандомно згенерувати кількість днів для вже хворих людей

    # карантинна зона + ввести імунітет
    coeffs_infected = [random.random() for j in range(number_people)]

    all = 31829
    recovered = 27609
    dead = 627

    infected_prob = all / 1000000
    people_states = []

    for x in range(number_people):
        x = random.random()
        if x <= infected_prob:
            people_states.append('i')
        else:
            people_states.append('s')

    dead_prob = dead / all
    dead_error = 0.01

    coeffs_dead = [random.uniform(dead_prob-dead_error, dead_prob+dead_error) for j in range(number_people)]

    return days_of_infection,\
            people_states,\
           coeffs_infected,\
           coeffs_dead
