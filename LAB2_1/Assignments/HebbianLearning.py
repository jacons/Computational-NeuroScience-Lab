from copy import copy

import numpy as np
from numpy import ndarray, random


class Hebbian_learning:
    def __init__(self,
                 source: ndarray,
                 epochs: int = 2,
                 lr: float = 0.1,
                 threshold: float = 1e-03):

        self.__source = source  # Dataset
        self.__epochs = epochs  # Number of max epochs to perform
        self.__lr = lr  # Learning rate
        self.__threshold = threshold  # Threshold for early stopping
        self.W = random.uniform(-1, 1, 2)  # weights of the network

        self.history_x1 = []
        self.history_x2 = []

    # Protected method that will be overridden to implement further rules
    def _hebbian_rule(self, u: ndarray) -> ndarray:
        return (self.W @ u) * u

    def __call__(self):

        current_epoch, norm = 0, 0
        flag, pred_W = False, None

        while current_epoch < self.__epochs and not flag:

            # iterate each element (composed by 2 values x1 & x2)
            for u in random.permutation(self.__source):
                # upgrade the weights based on hebbian rule
                # Oss. we are assuming that inside the constant "lr" there is also
                # the constant value "h" used to perform the differential equation
                # by the euler's method
                self.W += self.__lr * self._hebbian_rule(u)

                # we keep track the weights and the norm for each upgrade
                self.history_x1.append(self.W[0])
                self.history_x2.append(self.W[1])

            # we're also implementing the early stopping based on norm
            flag = True if pred_W is not None and np.linalg.norm(self.W - pred_W) < self.__threshold else False

            pred_W = copy(self.W)
            current_epoch += 1
