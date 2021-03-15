"""
SEIRVS disease model

S' = -beta*(E + I)*S - w*S + fi*V
S = 540 E = 450 V = 1 I = 1
E' = beta*(E + I)*S - (gamma + alpha)E
I' = alpha*E - (gamma + sigma)I
R' = sigma*I - gamma*R - delta*R
V' = gamma*(E + I + R) + w*S - fi*V
D' = delta*R
"""

import numpy as np
from diff_equation.ODESolver.ODESolver import ForwardEuler
from matplotlib import pyplot as plt


class SEIRVS:
    def __init__(self, beta, omega, fi, gamma, alpha, sigma, delta, S0, E0, I0, R0, V0, D0):
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

        self.initial_conditions = [S0, E0, I0, R0, V0, D0]

    def __call__(self, u, t):

        S, E, I, R, V, D = u

        return np.asarray([
            -self.beta(t)*(E + I)*S - self.omega(t)*S + self.fi(t)*V,
            self.beta(t)*(E + I)*S - (self.gamma(t) + self.alpha(t))*E,
            self.alpha(t)*E - (self.gamma(t) + self.sigma(t))*I,
             self.sigma(t)*I - self.gamma(t)*R - self.delta(t)*R,
            self.gamma(t)*(E + I + R) + self.omega(t)*S - self.fi(t)*V,
            self.delta(t) * R
        ])


if __name__ == "__main__":

    fi = 1/120 # from V to S
    gamma = 1/14 # from EIR to V
    beta = 0.01 # from S to E
    alpha = 0.2 # from E to I
    sigma = alpha # from I to R
    omega = 0.008 # from S to V
    delta = 1/50 # from R to D

    S0 = 990
    E0 = 10
    I0 = 0
    R0 = 0
    V0 = 0
    D0 = 0

    sir = SEIRVS(beta, omega, fi, gamma, alpha, sigma, delta, S0, E0, I0, R0, V0, D0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 1001)

    u, t = solver.solve(time_steps)

    for x in u:
        print(x)
    print(t)

    plt.plot(t, u[:, 0], label="S")
    plt.plot(t, u[:, 1], label="E")
    plt.plot(t, u[:, 2], label="I")
    plt.plot(t, u[:, 3], label="R")
    plt.plot(t, u[:, 4], label="V")
    plt.plot(t, u[:, 5], label="D")
    plt.legend()
    plt.show()
