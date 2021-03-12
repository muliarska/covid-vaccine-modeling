"""
SEIRVS disease model

S' = -beta*(E + I)*S - w*S + fi*V
E' = beta*(E + I)*S - (gamma + alpha)E
I' = alpha*E - (gamma + sigma)I
R' = sigma*I - gamma*R
V' = gamma*(E + I + R) + w*S - fi*V
"""

import numpy as np
from ODESolver import ForwardEuler
from matplotlib import pyplot as plt


class SEIRVS:
    def __init__(self, beta, omega, fi, gamma, alpha, sigma, S0, E0, I0, R0, V0):
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

        self.initial_conditions = [S0, E0, I0, R0, V0]

    def __call__(self, u, t):

        S, E, I, R, V = u

        return np.asarray([
            -self.beta(t)*(E + I)*S - self.omega(t)*S + self.fi(t)*V,
            self.beta(t)*(E + I)*S - (self.gamma(t) + self.alpha(t))*E,
            self.alpha(t)*E - (self.gamma(t) + self.sigma(t))*I,
             self.sigma(t)*I - self.gamma(t)*R,
            self.gamma(t)*(E + I + R) + self.omega(t)*S - self.fi(t)*V
        ])


if __name__ == "__main__":

    fi = 1/120 # from V to S
    gamma = 1/14 # from EIR to V
    beta = 0.01 # from S to E
    alpha = 0.2 # from E to I
    sigma = alpha # from I to R
    omega = 0.008 # from S to V

    S0 = 100
    E0 = 1
    I0 = 0
    R0 = 0
    V0 = 0

    sir = SEIRVS(beta, omega, fi, gamma, alpha, sigma, S0, E0, I0, R0, V0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 200, 1001)
    print(time_steps)
    u, t = solver.solve(time_steps)

    plt.plot(t, u[:, 0], label="S")
    plt.plot(t, u[:, 1], label="E")
    plt.plot(t, u[:, 2], label="I")
    plt.plot(t, u[:, 3], label="R")
    plt.plot(t, u[:, 4], label="V")
    plt.legend()
    plt.show()
