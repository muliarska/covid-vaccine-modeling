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

import numpy as np
from matplotlib import pyplot as plt


class SEIRVS:
    def __init__(self, beta, omega, fi, gamma, alpha, sigma, delta, theta, S0, E0, I0, R0, V0, D0, M0):
        """
        """

        if isinstance(omega, (float, int)):
            # Is number?
            self.omega = lambda t: omega
        elif callable(omega):
            self.omega = omega

        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta
        elif callable(beta):
            self.beta = beta

        if isinstance(fi, (float, int)):
            self.fi = lambda t: fi
        elif callable(fi):
            self.fi = fi

        if isinstance(gamma, (float, int)):
            self.gamma = lambda t: gamma
        elif callable(gamma):
            self.gamma = gamma

        if isinstance(alpha, (float, int)):
            self.alpha = lambda t: alpha
        elif callable(alpha):
            self.alpha = alpha

        if isinstance(sigma, (float, int)):
            self.sigma = lambda t: sigma
        elif callable(sigma):
            self.sigma = sigma

        if isinstance(delta, (float, int)):
            self.delta = lambda t: delta
        elif callable(delta):
            self.delta = delta

        if isinstance(theta, (float, int)):
            self.theta = lambda t: theta
        elif callable(delta):
            self.theta = theta

        self.initial_conditions = [S0, E0, I0, R0, V0, D0, M0]

    def solve_one_time_point(self, pure_variables, t):

        S, E, I, R, V, D, M = pure_variables

        S_1 = -self.beta(t)*(E + I)*S - self.omega(t)*S + self.fi(t)*M + self.theta(t) * V
        E_1 = self.beta(t)*(E + I)*S - (self.gamma(t) + self.alpha(t))*E
        I_1 = self.alpha(t)*E - (self.gamma(t) + self.sigma(t))*I
        R_1 = self.sigma(t)*I - self.gamma(t)*R - self.delta(t)*R
        V_1 = self.omega(t)*S - self.theta(t) * V
        D_1 = self.delta(t) * R
        M_1 = self.gamma(t) * (E + I + R) - self.fi(t) * M

        first_derivative = [S_1, E_1, I_1, R_1, V_1, D_1, M_1]

        S_2 = -self.beta(t) * (E_1 * S + S_1 * E + I_1 * S + S_1 * I) - self.omega(t) * S_1\
              + self.fi(t) * M_1 + self.theta(t) * V_1
        E_2 = self.beta(t) * (E_1 * S + S_1 * E + I_1 * S + S_1 * I) - \
              (self.gamma(t) + self.alpha(t)) * E_1
        I_2 = self.alpha(t) * E_1 - (self.gamma(t) + self.sigma(t)) * I_1
        R_2 = self.sigma(t) * I_1 - self.gamma(t) * R_1 - self.delta(t) * R_1
        V_2 = self.gamma(t) * (E_1 + I_1 + R_1) + self.omega(t) * S_1 - self.theta(t) * V_1
        D_2 = self.delta(t) * R_1
        M_2 = self.gamma(t) * (E_1 + I_1 + R_1) - self.fi(t) * M_1

        second_derivative = [S_2, E_2, I_2, R_2, V_2, D_2, M_2]

        result = []
        for i in range(7):
            result.append(pure_variables[i] + first_derivative[i] + 0.5 * second_derivative[i])

        return np.array(result)

    def solve(self, days):
        all_states = np.zeros((days, 7))
        curr_states = self.initial_conditions
        all_states[0, :] = curr_states

        for t in range(1, days):
            curr_states = self.solve_one_time_point(curr_states, t)
            print("DAY : ", t)
            print(curr_states)
            all_states[t, :] = curr_states

        return all_states



if __name__ == "__main__":

    fi = 1 / 120  # from M to S
    gamma = 1 / 21  # from EIR to M
    alpha = 0.2  # from E to I
    beta = 0.4  # from S to E == contact rate
    sigma = alpha  # from I to R
    omega = 0.002  # from S to V, кількість вакцинованих за день * якість вакцини
    delta = 1/50 # from R to D
    theta = 1/100 # from V to S, тривалість дії вакцини

    days = 600

    S0 = 0.9
    E0 = 0.1
    I0 = 0
    R0 = 0
    V0 = 0
    D0 = 0
    M0 = 0

    sir = SEIRVS(beta, omega, fi, gamma, alpha, sigma, delta, theta, S0, E0, I0, R0, V0, D0, M0)

    u = sir.solve(days)

    t = np.linspace(0, days, days)

    # for x in u:
    #     print(x)

    plt.plot(t, u[:, 0], label="S")
    plt.plot(t, u[:, 1], label="E")
    plt.plot(t, u[:, 2], label="I")
    plt.plot(t, u[:, 3], label="R")
    plt.plot(t, u[:, 4], label="V")
    plt.plot(t, u[:, 5], label="D")
    plt.plot(t, u[:, 6], label="M")
    plt.legend()
    plt.show()
