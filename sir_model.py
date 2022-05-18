import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


class SIR:
    def __init__(self):
        self.compartments = ["S", "I", "R"]

    def get_initial_values(self):
        iv = {
            "S": 0.8,
            "I": 0.2,

            "R": 0
             }
        return np.array([iv[comp] for comp in self.compartments]).flatten()

    def get_solution(self, t: np.ndarray, parameters: dict, initial_values: np.ndarray) -> np.ndarray:
        return np.array(odeint(self._get_model, initial_values, t, args=(parameters, )))

    def _get_model(self, xs: np.ndarray, _, ps: dict) -> np.ndarray:
        # the same order as in self.compartments!
        S = xs[0]
        I = xs[1]
        R = xs[2]

        model_eq_dict = {
            "S": - S * I * ps["beta"],  # S'(t)
            "I":  S * I * ps["beta"] - ps["gamma"] * I,  # I'(t)
            "R":  ps["gamma"] * I,  # R'(t)
        }

        model_eq = [model_eq_dict[comp] for comp in self.compartments]
        v = np.array(model_eq).flatten()
        return v


def main():
    ps = {
        "beta": 0.5,
        "gamma": 0.1
    }
    t = np.linspace(0, 20)
    p = SIR()
    initial_values = p.get_initial_values()
    y = p.get_solution(t=t, parameters=ps, initial_values=initial_values)
    s = y[:, 0]
    i = y[:, 1]
    r = y[:, 2]

    plt.figure()
    plt.plot(t, s, 'b', label='S(t)')
    plt.plot(t, i, 'r', label='I(t)')
    plt.plot(t, r, 'g', label='R(t)')
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
