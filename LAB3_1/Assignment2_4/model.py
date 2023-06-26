from torch import Tensor
from torch.nn import Module, Linear, CrossEntropyLoss


class sMNIST_model(Module):  # Recurrent Neural network
    def __init__(self, rnn_model: Module, hidden: int, bi: bool):
        super(sMNIST_model, self).__init__()

        self.rnn = rnn_model

        B = 2 if bi else 1
        self.readout = Linear(B * hidden, 10)
        self.criteria = CrossEntropyLoss()

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        y_pred = self.readout(out[:, -1, :])  # take only the last hidden state as (cumulative knowledge)

        loss = None
        if y is not None:
            loss = self.criteria(y_pred, y)
        return (loss, y_pred) if loss is not None else y_pred
