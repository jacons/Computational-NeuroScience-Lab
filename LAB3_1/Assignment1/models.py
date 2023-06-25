from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh, MSELoss, RNN


class TimeDelayNN(Module):  # Time delay neural network model
    def __init__(self, hidden: int = 10, back_steps: int = 10):
        super(TimeDelayNN, self).__init__()

        self.feedforward = Sequential(
            Linear(back_steps + 1, hidden),  # Linear layer
            Tanh(),  # Activation function
            Linear(hidden, 1))  # Readout

        self.criteria = MSELoss()  # Mean square error loss

    def forward(self, x: Tensor, y: Tensor = None):
        y_pred = self.feedforward(x)

        loss = None
        if y is not None:
            loss = self.criteria(y_pred, y)
        return (loss, y_pred) if loss is not None else y_pred


class RecurrentNN(Module):  # Recurrent Neural network
    def __init__(self, hidden: int, layers: int, no_linearity: str = "relu"):
        super(RecurrentNN, self).__init__()

        self.hidden_size = hidden
        self.layers = layers

        self.rnn = RNN(1, hidden_size=hidden, num_layers=layers,
                       nonlinearity=no_linearity, batch_first=True)

        self.read_out = Sequential(Linear(hidden, 1))
        self.criteria = MSELoss()  # Mean square error loss

        self.last_hidden = None

    def forward(self, x: Tensor, y: Tensor = None, save_state: bool = False):
        self.rnn.flatten_parameters()
        output, hn = self.rnn(x, self.last_hidden)
        y_pred = self.read_out(output)

        if save_state:
            self.last_hidden = hn.detach()

        loss = None
        if y is not None:
            loss = self.criteria(y_pred, y)
        return (loss, y_pred) if loss is not None else y_pred
