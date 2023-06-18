from typing import Tuple

import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor, zeros, eq
from torch.nn import Flatten
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


def show_split(tr: ndarray, dev: ndarray, ts: ndarray):
    plt.figure(figsize=(28, 4))
    plt.plot([*range(0, 4000)], tr[:, 1], label="Train")
    plt.plot([*range(4000, 5000)], dev[:, 1], label="Validation")
    plt.plot([*range(5000, 10000)], ts[:, 1], label="Test")
    plt.title("Hold-out")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.grid()
    plt.legend()
    plt.xlim([-10, 10010])
    plt.show()


def show_loss(tr_loss: Tensor):
    plt.figure(figsize=(5, 4))
    plt.title("Training loss")
    plt.plot(tr_loss, label="Tr loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Overall epochs")
    plt.ylabel("Loss")
    plt.show()


def show_result(tr_y_pred: Tensor, tr_y: Tensor, ts_y_pred: Tensor, ts_y: Tensor):
    tr_y_pred, tr_y, ts_y_pred, ts_y = tr_y_pred.cpu(), tr_y.cpu(), ts_y_pred.cpu(), ts_y.cpu()
    f, axs = plt.subplots(ncols=1, nrows=2, figsize=(28, 7))

    axs[0].plot(tr_y_pred[-1000:], label="Tr output")
    axs[0].plot(tr_y[-1000:], label="Target output")
    axs[0].set_title("Comparison between Timeseries on Traning set (last 1k elements)")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(ts_y_pred[-1000:], label="Tr output")
    axs[1].plot(ts_y[-1000:], label="Target output")
    axs[1].set_title("Comparison between Timeseries on Test set (last 1k elements)")
    axs[1].grid()
    axs[1].legend()

    plt.show()


def make_sequence(df: ndarray, steps: int, dt: str, unsqueeze: bool = False) -> Tuple[Tensor, Tensor]:
    # Build a tensor used for the training

    len_df = df.shape[0]  # Length of dataset

    df_x = zeros(len_df - steps, steps)
    df_y = zeros(len_df - steps, 1)

    for idx, i in enumerate(range(steps, len_df)):
        x = df[i - steps:i, 0]
        y = df[i - 1, 1] if dt == "NARMA10" else df[i, 0]

        df_x[idx, :] = Tensor(x)
        df_y[idx, :] = y

    if unsqueeze:
        df_x = df_x.unsqueeze(-1)

    return df_x, df_y


class Sequential_mnist(Dataset):

    def __init__(self, hold_out: str):
        super(Sequential_mnist, self).__init__()

        if hold_out == "train":
            original_mist = MNIST(root="sources", download=True, train=True)
            data = original_mist.data[:50000]
            target = original_mist.targets[:50000]

        elif hold_out == "dev":
            original_mist = MNIST(root="sources", download=True, train=True)
            data = original_mist.data[50000:]
            target = original_mist.targets[50000:]

        else:
            original_mist = MNIST(root="sources", download=True, train=False)
            data = original_mist.data
            target = original_mist.targets

        self.data = data.reshape(-1, 28 * 28, 1).contiguous().type(torch.float32) / 255
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def compute_acc(predicted: Tensor, labels: Tensor):
    """
    Compute the accuracy of model both for real and fake images
    :param predicted: label predicted by discriminator
    :param labels:  true label
    :return:
    """
    correct = eq(predicted.argmax(dim=-1), labels).sum().item()
    return float(correct) / float(labels.size(0))