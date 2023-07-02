import numpy as np
from numpy import ones, power, concatenate, ndarray, eye, zeros
from numpy.linalg import pinv
from numpy.random import rand


class LiquidStateMachine:

    def __init__(self,
                 units: int = 1000,
                 w_in_e: float = 5,
                 w_in_i: float = 2,
                 w_rec_e: float = 0.5,
                 w_rec_i: float = 1):
        """
        Oss. I build this class based on the matlab script provided in class

        Units: Number of neurons taken in consideration inside the liquid state

        We have two groups of hyperparameters; the former is composed by two parameters
        used for scaling the input connections, more precisely one is used for the excitation neurons
        and the other for inhibitory neurons. Likewise for the latter group used for
        recurrent connections.

        W_in_e: weights of excitatory neurons (input connections)
        w_in_i: weights of inhibitory neurons (input connections)

        W_rec_e: weights of excitatory neurons (recurrent connections)
        w_rec_i: weights of inhibitory neurons (recurrent connections)
        """

        # Number of Excitatory/Inhibitory neurons, I take always the same proportion
        # 80% excitatory and 20% inhibitory
        Ne, Ni = int(units * 0.8), int(units * 0.2)
        re, ri = rand(Ne), rand(Ni)

        # I precompute this three values, because they are used more than one time
        pow = power(re, 2)
        one_Ne, one_Ni = ones(Ne), ones(Ni)

        # Izhikevich parameters
        # Basically I'm representing the Izhikevich parameters for #neurons
        # a[i],b[i].. are the parameter for i-th neuron
        self.__a = concatenate((0.02 * one_Ne, 0.02 + 0.08 * ri))
        self.__b = concatenate((0.2 * one_Ne, 0.25 - 0.05 * ri))
        self.__c = concatenate((-65 + 15 * pow, -65 * one_Ni))
        self.__d = concatenate((8 - 6 * pow, 2 * one_Ni))

        # The "input" vector and the "recurrent" matrix
        self.__U = concatenate((w_in_e * one_Ne, w_in_i * one_Ni))
        self.__S = concatenate((w_rec_e * rand(units, Ne), -w_rec_i * rand(units, Ni)), axis=1)

        # Initial conditions of membrane potential and recovery variable
        self.__v0 = -65 * ones(units)  # Initial values of v
        self.__u0 = self.__b * self.__v0  # Initial values of u

        # Readout layer, initialized during the training
        self.__Wout = zeros(units)

    def __perform_state(self, time_series: ndarray) -> ndarray:
        # set initial condition
        u, v = self.__u0, self.__v0

        states = []  # List of states
        for element in time_series:
            I = element * self.__U
            fired = v >= 30

            v[fired] = self.__c[fired]
            u[fired] = u[fired] + self.__d[fired]

            I = I + np.sum(self.__S[:, fired], axis=1)

            v = v + 0.5 * (0.04 * power(v, 2) + 5 * v + 140 - u + I)  # for numerical stability
            v = v + 0.5 * (0.04 * power(v, 2) + 5 * v + 140 - u + I)  # we apply twice
            u = u + self.__a * (self.__b * v - u)

            states.append(v >= 30)

        return np.array(states, dtype=int)  # boolean to int

    def fit(self, time_series: ndarray, target: ndarray, lambd=0.1):
        # We apply the liquid state based on the input time series
        liquids_state = self.__perform_state(time_series)

        # Then we fit the readout (the only trained weights) in one-shot (direct approach)
        # based on pseudo-inverse applying the ridge regression

        identity = eye(liquids_state.shape[1])
        # I apply the tikhonov regularization, because it seems that the plain (just pinv)
        # approach tends to the over-fitting of a training set.
        self.__Wout = pinv(liquids_state.T @ liquids_state + lambd * identity) @ liquids_state.T @ target

    def __call__(self, time_series: ndarray) -> ndarray:
        """
        In order to predict the values in the time series,
        it's enough to calculate the liquid state and multiply them with the readout vector.
        """
        return self.__perform_state(time_series) @ self.__Wout
