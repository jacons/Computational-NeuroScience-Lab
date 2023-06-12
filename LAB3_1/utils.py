from typing import Tuple

from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor, zeros


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


def show_result(tr_out, final_tr_y, test_out, ts_y):
    f, axs = plt.subplots(ncols=1, nrows=2, figsize=(28, 7))

    axs[0].plot(tr_out[-1000:], label="Tr output")
    axs[0].plot(final_tr_y[-1000:], label="Target output")
    axs[0].set_title("Comparison between Timeseries on Traning set (last 1k elements)")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(test_out[-1000:], label="Tr output")
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
        df_x = df_x.unsqueeze(2)

    return df_x, df_y
