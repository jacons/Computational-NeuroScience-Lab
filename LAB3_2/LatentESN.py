from typing import Tuple

import numpy as np
from numpy import ndarray, empty, zeros, tanh


class EchoStateNetwork:
    def __init__(self,
                 input_size: int,
                 hidden_dim: int,
                 omega: Tuple[float, float],
                 spectral_radius: float,
                 sparsity: float = 0.7):
        """
        Echo State Network implementation

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        :param sparsity: Percentage of sparsity
        """

        # Initialize the matrices
        Wx = (2 * np.random.random((hidden_dim, input_size))) - 1
        Wh = (2 * np.random.random((hidden_dim, hidden_dim))) - 1
        bh = (2 * np.random.random(hidden_dim)) - 1

        # Building the mask for the sparsity
        mask = (np.random.uniform(0, 1, (hidden_dim, hidden_dim)) < sparsity).astype(float)
        Wh = (Wh * mask)

        # Preprocessing to satisfy ESP
        self.Wx = omega[0] * Wx
        self.bh = omega[1] * bh
        self.Wh = spectral_radius * (Wh / self.spectral_radius(Wh))

        self.hidden_dim = hidden_dim
        return

    def __call__(self, x: ndarray, h0: ndarray = None) -> ndarray:

        # Initialize a stacked hidden states
        h_stack = empty((x.shape[0], self.hidden_dim))  # steps, input_size

        # The Default initially hidden state is equal to 0
        if h0 is None:
            h0 = zeros(self.hidden_dim)  # steps, hidden_size

        h = h0
        for i, x_ in enumerate(x):  # iterate on steps: x_ =  [input_size]
            z = (self.Wx @ x_.T + self.Wh @ h.T + self.bh).T  # [steps, hidden_size]
            h_stack[i] = tanh(z)

        return h_stack  # [steps, input_size]

    @staticmethod
    def spectral_radius(matrix) -> float:
        eigenvalues = np.linalg.eigvals(matrix)
        max_magnitude = np.max(np.abs(eigenvalues))
        return max_magnitude
