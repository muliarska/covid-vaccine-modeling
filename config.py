import random

# стани людей
person_states = ['s', 'i', 'is', 'r', 'd', 'v']


def build_coeffs(number_people):
    # дні інфікування або імунітету
    days_of_infection = [0 for _ in range(number_people)]

    # рандомно згенерувати кількість днів для вже хворих людей ?

    # імовірності заразитися
    coeffs_infected = [random.uniform(0, 0.4) for _ in range(number_people)]

    # статистика по Україні на 1 млн осіб
    all = 31829
    recovered = 27609
    dead = 627

    infected_prob = all / 1000000
    people_states = []

    # визначаю хто який стан матиме на початок моделі
    for x in range(number_people):
        x = random.random()
        if x <= infected_prob:
            people_states.append('i')
        else:
            people_states.append('s')

    dead_prob = dead / all
    dead_error = 0.01

    # коефіцієнт смерті
    coeffs_dead = [random.uniform(dead_prob-dead_error, dead_prob+dead_error) for j in range(number_people)]

    return days_of_infection, people_states, coeffs_infected, coeffs_dead
