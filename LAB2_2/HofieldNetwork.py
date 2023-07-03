import numpy as np
from numpy import ndarray, random


class HopfieldNetworks:

    def __init__(self, all_source: ndarray):

        self.n_neurons: int = all_source.shape[1]  # Number of neurons
        self.sources: ndarray = all_source  # dataset of numbers

        # Weights, oss. It will be initialized in fit()
        self.W = None

    def fit(self):
        """
          Learning phase, we set the weights on a specified value
        """
        # summing among outer-product
        self.W = np.einsum('ij,ik->jk', self.sources, self.sources) / self.n_neurons

        # no recurrent connection
        np.fill_diagonal(self.W, 0)

    def __call__(self, init_state: ndarray, original: ndarray, epochs: int, bias=0.5):

        state = np.copy(init_state)
        n_neurons = self.n_neurons

        # Dynamic of overlapping and energy function
        h_overlap, h_energy = [], []
        evolution_state = [init_state]

        for _ in range(epochs):

            # return a permuted list [0..n_neurons]
            update_order = random.permutation(n_neurons)
            for idx in update_order:  # update the state, each neuron at time

                # perform the next state
                out = self.W[idx] @ state + bias

                # activation function
                state[idx] = np.where(out > 0, 1, -1)

                # perform overlapping function
                overlap = (self.sources @ state) / n_neurons

                # perform energy function (the function considers the bias!)
                energy = -0.5 * (np.sum((self.W @ state) * state)) - bias * np.sum(state)

                # saving
                h_overlap.append(np.copy(overlap))
                h_energy.append(np.copy(energy))
                evolution_state.append(np.copy(state))

                if np.array_equal(original, state):
                    return evolution_state, np.array(h_overlap), h_energy

        return evolution_state, np.array(h_overlap), h_energy


def distort_image(img: ndarray, alpha: float) -> ndarray:
    img = np.copy(img)

    length = img.shape[0]
    index = random.permutation(length)
    todist = index[:round(length * alpha)]
    img.flat[todist] = -img.flat[todist]

    return img
