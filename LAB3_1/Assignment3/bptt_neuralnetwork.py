import numpy as np
from numpy import ndarray, zeros, sqrt
from numpy.random import uniform


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return e_x / np.sum(e_x, axis=0)


def d_softmax(x):
    x = softmax(x)
    return x * (1 - x)


def CELoss(y_pred: ndarray, y: ndarray):
    y_hoe = np.zeros_like(y_pred)  # [batch, out_dim]
    y_hoe[range(0, len(y)), y] = 1

    return -(y_hoe * np.log(y_pred)).sum()


def d_CELoss(y_pred: ndarray, y: ndarray):
    """

    :param y_pred: [batch, out_dim]
    :param y: [batch]
    :return:
    """
    y_hoe = np.zeros_like(y_pred)  # [batch, out_dim]
    y_hoe[range(0, len(y)), y] = 1

    return - (y_hoe / y_pred) + (1 - y_hoe) / (1 - y_pred)


class HandMadeRNN:
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, lr: float, clip: float):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        k = sqrt(1 / hidden_dim)
        self.w_x = uniform(-k, k, (hidden_dim, input_dim))
        self.w_h = uniform(-k, k, (hidden_dim, hidden_dim))
        self.w_o = uniform(-k, k, (hidden_dim, output_dim))

        self.b_h = uniform(-k, k, (hidden_dim, 1))
        self.b_o = uniform(-k, k, (output_dim, 1))

        self.grad_step = lambda m: lr * clip * (m / np.linalg.norm(m))

    def __call__(self, x: ndarray, y: ndarray = None, h0: ndarray = None):
        """
        :param x: [batch, steps, input_size]
        :param y: [batch, 1]
        :param h0: [batch, hidden_dim ]
        :return:
        """
        z_stack = zeros((x.shape[1] + 1, x.shape[0], self.hidden_dim))  # [steps, batch, hidden_size]
        h_stack = zeros((x.shape[1] + 1, x.shape[0], self.hidden_dim))  # [steps, batch, hidden_size]

        # ---------------- Forward ----------------
        h_stack[0] = zeros((x.shape[0], self.hidden_dim)) if h0 is None else h0  # [batch, hidden_size]
        for step, x_t in enumerate(np.rollaxis(x, 1), 1):  # x_t [ batch, input_size]
            z = (self.w_x @ x_t.T + self.w_h @ h_stack[step - 1].T + self.b_h).T
            h_stack[step] = np.tanh(z)
            z_stack[step] = z.__copy__()

        # output for the hidden layer
        y_pred = softmax((h_stack[-1] @ self.w_o + self.b_o.T).T).T
        # ---------------- Forward ----------------

        loss = None
        if y is not None:

            g_wh = np.zeros_like(self.w_h)
            g_wx = np.zeros_like(self.w_x)
            g_bh = np.zeros_like(self.b_h)

            d_activ = np.vectorize(lambda x_: 1 - np.power(np.tanh(x_), 2))

            # ---------------- Backward ----------------
            # Take only the last hidden state and apply the softmax return a distribution of probability
            loss = CELoss(y_pred, y)
            dLdo = d_CELoss(d_softmax(y_pred), y)

            g_wo = h_stack[-1].T @ dLdo  # gradient for W_o
            g_bo = dLdo.T.sum(axis=1, keepdims=1)  # gradient

            tmp = (dLdo @ self.w_o.T) * d_activ(z_stack[-1])
            for i in reversed(range(x.shape[1])):
                g_wh += tmp.T @ h_stack[i]
                g_wx += tmp.T @ x[:, i, :]
                g_bh += tmp.T.sum(axis=1, keepdims=1)
                tmp = (tmp @ self.w_h.T) * (d_activ(z_stack[i]))
            # ---------------- Backward ----------------

            self.w_x -= self.grad_step(g_wx)
            self.w_h -= self.grad_step(g_wh)
            self.w_o -= self.grad_step(g_wo)
            self.b_h -= self.grad_step(g_bh)
            self.b_o -= self.grad_step(g_bo)

        return (y_pred, loss) if loss is not None else y_pred
