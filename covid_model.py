import random

people_states = []


class CovidModel:
    def __init__(self, number_people, people_states):
        self.number_people = number_people
        self.people_states = people_states if len(people_states) != 0 else self.set_up_states()
        self.percent_vaccined = 1 / 700
        self.prob_connect = 20 / self.number_people

        self.set_up()

    def __copy__(self):
        new_model = CovidModel(self.number_people, self.people_states)
        return new_model

    def set_up(self):
        self.days_of_infection = [0 for _ in range(self.number_people)]

        infected_per_day = 6300
        dead_per_day = 220
        all_infected = 212000

        self.probability_get_infected = infected_per_day / all_infected
        self.probability_get_dead = dead_per_day / all_infected

    def set_up_states(self):
        # статистика по Україні на 1 млн осіб
        all = 31829
        infected_prob = all / 1000000

        people_states = []

        # визначаю хто який стан матиме на початок моделі
        for x in range(self.number_people):
            x = random.random()
            if x <= infected_prob:
                people_states.append('i')
            else:
                people_states.append('s')

        return people_states

    def build_matrix(self):
        # будується матриця з звязками
        self.matrix = [[0 for _ in range(self.number_people)] for _ in range(self.number_people)]

        for i in range(self.number_people):
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

        sum_infected = 0
        sum_recovered = 0
        sum_susceptible = 0
        sum_dead = 0
        sum_vaccined = 0

        for i in range(self.number_people):
            if self.people_states[i] == 'i' or self.people_states[i] == 'is':
                sum_infected += 1
            elif self.people_states[i] == 'r':
                sum_recovered += 1
            elif self.people_states[i] == 's':
                sum_susceptible += 1
            elif self.people_states[i] == 'v':
                sum_vaccined += 1
            else:
                sum_dead += 1

        print("Infected:", sum_infected)
        print("Dead:", sum_dead)
        print("Susceptible:", sum_susceptible)
        print("Recovered:", sum_recovered)
        print("Vaccinated:", sum_vaccined)

    def covid_model(self, numb_of_days, if_vaccine):
        for day in range(numb_of_days):
            # print_matrix(matrix)
            if day % (numb_of_days - 1) == 0:
                # print("Day ", day)
                self.get_states()
                print(if_vaccine)
                print("\n")

            if day % 10 == 0:
                # змінює звязки кожні 10 днів
                self.build_matrix()

            vaccined_people = 0
            needed_vaccined_people = self.percent_vaccined * self.number_people


            for person in range(self.number_people):

                if if_vaccine and vaccined_people < needed_vaccined_people:
                    if self.people_states[person] != 'v':
                        self.people_states[person] = 'v'
                        vaccined_people += 1

                if self.people_states[person] == 's':
                    for indx, numb in enumerate(self.matrix[person]):
                        if numb and (self.people_states[indx] == 'i'):
                            # здорова людина контактує з хворим
                            random_numb = random.random()
                            if random_numb < self.probability_get_infected:
                                if random_numb < 0.5:
                                    # людина йде на ізоляцію
                                    self.people_states[person] = 'is'
                                else:
                                    self.people_states[person] = 'i'
                                break

                elif self.people_states[person] == 'i' or self.people_states[person] == 'is':
                    if self.days_of_infection[person] == 21:
                        # хвора людина видужує та отримує імунітет
                        self.people_states[person] = 'r'
                        self.days_of_infection[person] = 0
                    else:
                        random_numb = random.random()
                        if random_numb < self.probability_get_dead:
                            # хвора людина помирає
                            self.people_states[person] = 'd'
                            # clear_connections(matrix, person)
                        else:
                            self.days_of_infection[person] += 1

                elif self.people_states[person] == 'r':
                    if self.days_of_infection[person] == 120:
                        # в людини закінчується імунітет
                        self.people_states[person] = 's'
                    else:
                        self.days_of_infection[person] += 1

    def run_simulation(self, numb_of_days, start_vaccine):
        self.build_matrix()

        if start_vaccine != -1:
            self.covid_model(start_vaccine, False)
            self.covid_model(numb_of_days-start_vaccine, True)
        else:
            self.covid_model(numb_of_days, False)


if __name__ == '__main__':

    model = CovidModel(2000, [])
    model_copy = model.__copy__()

    print("WITHOUT VACCINE")
    model_copy.run_simulation(50, -1)

    start = 40
    while start >= 0:
        print("WITHOUT VACCINE:", start, "WITH VACCINE:", 50-start, "\n")
        model_copy = model.__copy__()
        model_copy.run_simulation(50, start)
        start -= 10