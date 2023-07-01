import numpy as np
from numpy import ndarray, empty, zeros, tanh
from numpy.linalg import pinv, eigvals
from numpy.random import random


class LatentESN_numpy:
    def __init__(self, input_size: int, hidden_dim: int, omega: float,
                 spectral_radius: float):
        """
        Echo State Network implementation

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        """
        self.hidden_dim = hidden_dim

        # Initialize the matrices
        Wx = (2 * random((hidden_dim, input_size))) - 1
        Wh = (2 * random((hidden_dim, hidden_dim))) - 1
        bh = (2 * random(hidden_dim)) - 1

        # Preprocessing to satisfy ESP
        self.Wx = omega * Wx
        self.bh = omega * bh
        self.Wh = spectral_radius * (Wh / self.spectral_radius(Wh))

    @staticmethod
    def spectral_radius(matrix) -> float:
        eigenvalues = eigvals(matrix)
        max_magnitude = np.max(np.abs(eigenvalues))
        return max_magnitude

    def reservoir(self, seq: ndarray, h0: ndarray = None) -> ndarray:
        # Oss. It doesn't handle batch operations

        # Initialize a stacked hidden states [steps + 1, input_size]
        h_stack = empty((seq.shape[0] + 1, self.hidden_dim))

        # The Default initially hidden state is equal to 0 [steps, hidden_size]
        h_stack[0] = zeros(self.hidden_dim) if h0 is None else h0.copy()

        for step, x in enumerate(seq, 1):  # iterate on steps: x_ =  [input_size]
            z = self.Wx @ x.T + self.Wh @ h_stack[step - 1].T + self.bh  # [steps, hidden_size]
            h_stack[step] = tanh(z.T)

        return h_stack[1:]  # [steps, input_size]


class ESN_seq2seq(LatentESN_numpy):

    def __init__(self, input_size: int, hidden_dim: int, omega: float,
                 spectral_radius: float, tikhonov: float):
        """
        Model based on Echo State network used into "Sequence to Sequence" scenario.

        :param input_size: Input dimension
        :param hidden_dim: hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        :param tikhonov: Tikhonov regularization parameter
        """
        # Latent Echo state network (untrained)
        super().__init__(input_size, hidden_dim, omega, spectral_radius)
        # Readout (trained)
        self.Wo = None

        self.tikhonov = tikhonov  # Tikhonov regularization

    def fit(self, x: ndarray, y: ndarray, transient: int):
        """
        Fit the model using the input, the target.
        First, it calculates the hidden states, which are used to fit the readout;
        finally, we return the loss between the output and target with a trained model
        and the last hidden state.
        """

        # Perform the hidden states (without a given initial state)
        h_states = self.reservoir(seq=x)  # [steps, input_size]
        # Discard the transient
        h_states, y = h_states[transient:], y[transient:]

        # Fit directly the readout
        self.Wo = pinv(
            h_states.T @ h_states + self.tikhonov * np.eye(h_states.shape[1])) @ h_states.T @ y

        y_pred = h_states @ self.Wo
        return self.MSE(y, y_pred), h_states[-1]

    @staticmethod
    def MSE(y: ndarray, y_pred: ndarray) -> float:
        """
        Mean square error
        :param y: Target
        :param y_pred: Predicted target
        """
        return np.power((y - y_pred), 2).mean()

    def predict(self, x: ndarray, y: ndarray = None, h0: ndarray = None):
        """
        Perform the forward pass with initially hidden state, if provided.
        If it provided the target, it performs also the loss.
        :param x: Input signal
        :param y: Target signal
        :param h0: Initially hidden state
        """

        # Perform the hidden states
        h_states = self.reservoir(seq=x, h0=h0)  # [steps, input_size]
        # Output signal
        y_pred = h_states @ self.Wo

        loss = None
        if y is not None:
            loss = self.MSE(y, y_pred)

        output = h_states[-1], y_pred
        return ((loss,) + output) if loss is not None else output

    def validate(self, x: ndarray, y: ndarray, h0: ndarray = None) -> tuple[float, ndarray]:
        loss, last_h, _ = self.predict(x, y, h0)
        return loss, last_h
