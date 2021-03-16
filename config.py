import random

# стани людей
person_states = ['s', 'i', 'is', 'r', 'd', 'v']


def build_coeffs(number_people):
    # дні інфікування або імунітету
    days_of_infection = [0 for _ in range(number_people)]

    infected_per_day = 6300
    dead_per_day = 220
    all_infected = 212000

    probability_get_infected = infected_per_day / all_infected
    probability_get_dead = dead_per_day / all_infected

    # статистика по Україні на 1 млн осіб
    all = 31829

    infected_prob = all / 1000000
    people_states = []

    # визначаю хто який стан матиме на початок моделі
    for x in range(number_people):
        x = random.random()
        if x <= infected_prob:
            people_states.append('i')
        else:
            people_states.append('s')


    return days_of_infection, people_states, probability_get_infected, probability_get_dead
