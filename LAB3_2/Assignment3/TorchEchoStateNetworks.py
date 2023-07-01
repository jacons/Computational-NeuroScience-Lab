import numpy as np
import torch
from numpy import ndarray
from numpy.linalg import eigvals
from torch import Tensor, empty, zeros, tanh


class LatentESN_torch:
    def __init__(self, input_size: int, hidden_dim: int, omega: float, spectral_radius: float,
                 leakage_rate: float, device: str = "cpu"):
        """
        Echo State Network implementation

        :param input_size: Input dimension
        :param hidden_dim: Hidden dimension
        :param omega: Scaling factor of input matrix and bias
        :param spectral_radius: Desiderata spectral radius
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
