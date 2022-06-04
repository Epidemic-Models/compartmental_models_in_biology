import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


class LVolterra:
    def __init__(self):
        self.compartments = ["x", "y"]

    def get_initial_values(self):
        iv = {
            "x": 100,
            "y": 8
             }
        return np.array([iv[comp] for comp in self.compartments]).flatten()

    def get_solution(self, t: np.ndarray, parameters: dict, initial_values: np.ndarray) -> np.ndarray:
        return np.array(odeint(self._get_model, initial_values, t, args=(parameters, )))

    def _get_model(self, xs: np.ndarray, _, ps: dict) -> np.ndarray:
        # the same order as in self.compartments!
        x = xs[0]
        y = xs[1]

        model_eq_dict = {
            "x": ps["alpha"] * x - ps["beta"] * x * y,  # x'(t)
            "y": ps["delta"] * x * y - ps["gamma"] * y,  # y'(t)
        }

        model_eq = [model_eq_dict[comp] for comp in self.compartments]
        v = np.array(model_eq).flatten()
        return v


def main():
    ps = {
        "alpha": 0.4,
        "beta": 0.4,
        "gamma": 2.0,
        "delta": 0.09
    }
    t = np.linspace(0, 50, 100)
    p = LVolterra()
    initial_values = p.get_initial_values()
    z = p.get_solution(t=t, parameters=ps, initial_values=initial_values)
    x = z[:, 0]
    y = z[:, 1]

    plt.figure()
    plt.plot(t, x, '-o', label='x(t)')
    plt.plot(t, y, '--', label='y(t)')
    plt.xlim([0, 50])
    plt.ylim([0, 140])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
