import numpy as np
import torch
from numpy import ndarray
from numpy.linalg import eigvals
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from torch import Tensor, empty, zeros, tanh
from torch.linalg import pinv

from Utils.utils import compute_acc


class LatentESN_torch:
    def __init__(self, input_size: int, hidden_dim: int, omega: float, spectral_radius: float,
                 leakage_rate: float, device: str = "cpu"):
        """
        Latent Echo State Network implementation

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        :param leakage_rate: Leaky Integrator constant
        :param device: matrices' device
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.leakage_rate = leakage_rate

        # Initialize the matrices
        Wx = (2 * torch.rand((hidden_dim, input_size), device=device)) - 1
        Wh = (2 * np.random.random((hidden_dim, hidden_dim))) - 1
        bh = (2 * torch.rand((hidden_dim, 1), device=device)) - 1

        # Preprocessing to satisfy ESP
        self.Wx = omega * Wx
        self.bh = omega * bh
        Wh = spectral_radius * (Wh / self.spectral_radius(Wh))
        self.Wh = torch.from_numpy(Wh).float().to(device)

    @staticmethod
    def spectral_radius(matrix: ndarray) -> float:
        eigenvalues = eigvals(matrix)
        max_magnitude = np.max(np.abs(eigenvalues))
        return max_magnitude

    @staticmethod
    def MSE(y: Tensor, y_pred: Tensor) -> Tensor:
        """
        Mean square error
        :param y: Target
        :param y_pred: Predicted target
        """
        return torch.pow((y - y_pred), 2).mean()

    def reservoir(self, seq: Tensor, h0: Tensor = None) -> Tensor:
        # Oss. Implemented with batch operations

        # Initialize the stacked hidden states [steps + 1, batch, input_size]
        h_stack = empty((seq.shape[0] + 1, seq.shape[1], self.hidden_dim), device=self.device)

        # The Default initially hidden state is equal to 0 [batches, hidden_size]
        h_stack[0] = zeros((seq.shape[1], self.hidden_dim), device=self.device) if h0 is None else h0

        for step, x in enumerate(seq, 1):  # iterate on steps: x_ =  [input_size]
            z = self.Wx @ x.T + self.Wh @ h_stack[step - 1].T + self.bh  # [bach, hidden_size]
            h_stack[step] = tanh(z.T)

        return h_stack[1:]  # [steps, input_size]

    def reservoir_last(self, seq: Tensor, h0: Tensor = None) -> Tensor:
        # Oss. Implemented with batch operations

        # The Default initially hidden state is equal to 0 [batches, hidden_size]
        hidden = zeros((seq.shape[1], self.hidden_dim), device=self.device) if h0 is None else h0

        for x in seq:  # iterate on steps: x_ = [input_size]
            z = self.Wx @ x.T + self.Wh @ hidden.T + self.bh  # [bach, hidden_size]
            hidden = (1 - self.leakage_rate) * hidden + self.leakage_rate * tanh(z.T)

        return hidden  # [steps, input_size]


class SGD_Seq2Seq(LatentESN_torch):
    def __init__(self, input_size: int, hidden_dim: int, omega: float,
                 spectral_radius: float, leakage_rate: float, max_iter: int, device="cpu"):
        """
        Sequence 2 Sequence model based on Echo state network

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        :param leakage_rate: Leaky Integrator constant
        :param max_iter: Max number of iteration
        :param device: matrices' device
        """
        super().__init__(input_size, hidden_dim, omega, spectral_radius, leakage_rate, device)

        # Stochastic Gradient Descent as readout
        self.regressor = SGDRegressor(max_iter=max_iter, learning_rate="adaptive")

    def fit(self, x: Tensor, y: Tensor, transient: int) -> Tensor:
        """
        Fit the model using the input, the target.
        First, it calculates the hidden states, which are used to fit the readout;
        finally, we return the loss between the output and target with a trained model
        and the last hidden state.
        """

        # Perform the hidden states (without a given initial state)
        h_stack = self.reservoir(seq=x)
        # Discard the transient
        h_stack, y = h_stack[transient:].squeeze(), y[transient:].squeeze()

        # Fit directly the readout
        self.regressor.fit(h_stack, y)
        y_pred = self.regressor.predict(h_stack)

        return self.MSE(y, y_pred)

    def predict(self, x: Tensor, y: Tensor, h0: Tensor = None):
        """
        Perform the forward pass with initially hidden state, if provided.
        If it provided the target, it performs also the loss.
        :param x: Input signal
        :param y: Target signal
        :param h0: Initially hidden state
        """

        # Perform the hidden states
        h_stack = self.reservoir(seq=x, h0=h0).squeeze()  # [steps, input_size]
        # Output signal
        y_pred = self.regressor.predict(h_stack)

        loss = None
        if y is not None:
            loss = self.MSE(y, y_pred)

        output = h_stack[-1], y_pred
        return ((loss,) + output) if loss is not None else output


class sMNISTEsnClassifier(LatentESN_torch):
    def __init__(self, input_size: int, hidden_dim: int, omega: float, spectral_radius: float,
                 leakage_rate: float, tikhonov: float, device: str = "cpu"):
        """
        Model based on Echo State network used into "Classification" scenario.

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
        :param leakage_rate: Leaky Integrator constant
        :param tikhonov: Tikhonov regularization parameter
        :param device: matrices' device

        """
        # Latent Echo state network (untrained)
        super().__init__(input_size, hidden_dim, omega, spectral_radius, leakage_rate, device)

        # Readout (trained)
        self.Wo = None

        self.tikhonov = tikhonov  # Tikhonov regularization
        self.device = device

    def fit(self, x: Tensor, y: Tensor = None):
        """
        Fit the model using the input, the target.
        First, it calculates the hidden states, which are used to fit the readout;
        (In particular, the last one) finally, we return the loss between the output
        and target with a trained model.
        """

        # Perform the LAST hidden states
        h_last = self.reservoir_last(seq=x)  # [batch, hidden_dim]

        # Fit directly the readout
        ID = torch.eye(h_last.shape[1], device=self.device)
        self.Wo = pinv(
            h_last.T @ h_last + self.tikhonov * ID) @ h_last.T @ y

        y_pred = h_last @ self.Wo  # [batch, 10]
        return compute_acc(predicted=y_pred, labels=y.argmax(-1))

    def predict(self, x: Tensor, y: Tensor = None):
        """
        Perform the forward pass.
        If it provided the target,
        it performs also the loss.
        :param x: Input signal [steps, batch, input_dim]
        :param y: Target signal [batch, 10]
        """

        # Perform the LAST hidden states
        h_last = self.reservoir_last(seq=x)  # [batch. hidden_dim]

        # Output signal
        y_pred = h_last @ self.Wo  # [batch, 10]

        acc = None
        if y is not None:
            acc = compute_acc(predicted=y_pred, labels=y.argmax(-1))

        return (acc, y_pred) if acc is not None else y_pred


class EsnSvmClassifier(LatentESN_torch):
    def __init__(self, input_size: int, hidden_dim: int, omega: float, spectral_radius: float,
                 leakage_rate: float, device: str = "cpu"):
        super().__init__(input_size, hidden_dim, omega, spectral_radius, leakage_rate, device)

        self.clf = linear_model.SGDClassifier(loss="squared_error", n_jobs=-1, verbose=1)

    def fit(self, x: Tensor, y: Tensor = None) -> Tensor:
        # Perform the LAST hidden states
        h_last = self.reservoir_last(seq=x).cpu()  # [batch. hidden_dim]

        print("Fitting...")
        self.clf.fit(h_last, y, )
        print("Predict..")
        y_pred = self.clf.predict(h_last)

        return self.MSE(y, y_pred)

    def predict(self, x: Tensor, y: Tensor = None):
        """
        Perform the forward pass.
        If it provided the target,
        it performs also the loss.
        :param x: Input signal
        :param y: Target signal
        """

        # Perform the LAST hidden states
        h_last = self.reservoir_last(seq=x).cpu()  # [batch. hidden_dim]

        # Output signal
        print("Predict..")
        y_pred = self.clf.predict(h_last)

        acc = None
        if y is not None:
            acc = compute_acc(y, y_pred)

        return (acc, y_pred) if acc is not None else y_pred
