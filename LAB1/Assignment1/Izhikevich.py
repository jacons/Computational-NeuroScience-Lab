# Izhikevich model
import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros


class Izhikevich_simulation:
    def __init__(self, config: dict):
        self.hist_u, self.hist_v = None, None
        self.config = config

    def simulation(self):
        """
       Oss. I noticed that in some cases the coefficients are different, hence
       the take as default value 5 & 140, and if necessary, I modify them.
       We perform the simulation based on input current given by params.
       In the following code, I iterate each current value and I update the
       v & u variables.
       """
        v, u = self.config["v0"], self.config["u0"]
        a, b, c, d = self.config["params"]

        in_len = len(self.config["current_input"])
        self.hist_v, self.hist_u = zeros(in_len), zeros(in_len)  # I keep track of the dynamics

        for idx, curr_value in enumerate(self.config["current_input"]):

            # Differential equations "solved" by Euler's method
            v = (v +
                 self.config["tau"] *
                 (0.04 * np.power(v, 2) +
                  self.config["coefficients"][0] * v +
                  self.config["coefficients"][1] - u +
                  curr_value))

            u = u + self.config["tau"] * a * (b * v - u)

            if v > 30:
                self.hist_v[idx] = 30
                v = c
                u += d
            else:
                self.hist_v[idx] = v

            self.hist_u[idx] = u

        return self

    def show_charts(self, show_input: bool = True):
        fig, ax = plt.subplots(ncols=2, figsize=(30, 5))
        fig.suptitle(self.config["title"], fontsize=16)

        x_len = np.arange(0, self.config["length"] + self.config["tau"], self.config["tau"])

        ax[0].set_title("Membrane potential dynamics")
        ax[0].plot(x_len, self.hist_v, label="Membrane potential")
        ax[0].set_xlabel("Time (t)")
        ax[0].set_ylabel("Membrane potential u(t)")
        ax[0].grid()

        if show_input:
            t = -min(self.hist_v) + max(self.config["current_input"]) + 5
            ax[0].plot(x_len, np.array(self.config["current_input"]) - t, label="Input current")

        ax[0].legend()

        ax[1].set_title("Phase portrait")
        ax[1].plot(self.hist_v, self.hist_u)
        ax[1].set_xlabel("Membrane potential")
        ax[1].set_ylabel("Recovery variable")
        ax[1].grid()

        plt.show()


class Accommodation(Izhikevich_simulation):
    def simulation(self):
        """
       Oss. I noticed that in some cases the coefficients are different, hence
       the take as default value 5 & 140, and if necessary, I modify them.
       We perform the simulation based on input current given by params.
       In the following code, I iterate each current value and I update the
       v & u variables.
       """
        v, u = self.config["v0"], self.config["u0"]
        a, b, c, d = self.config["params"]

        in_len = len(self.config["current_input"])
        self.hist_v, self.hist_u = zeros(in_len), zeros(in_len)  # I keep track of the dynamics

        for idx, curr_value in enumerate(self.config["current_input"]):

            # Differential equations "solved" by Euler's method
            v = (v +
                 self.config["tau"] *
                 (0.04 * np.power(v, 2) +
                  self.config["coefficients"][0] * v +
                  self.config["coefficients"][1] - u +
                  curr_value))

            u = u + self.config["tau"] * a * (b * (v + 65))

            if v > 30:
                self.hist_v[idx] = 30
                v = c
                u += d
            else:
                self.hist_v[idx] = v

            self.hist_u[idx] = u

        return self
