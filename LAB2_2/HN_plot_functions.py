import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


def plot_results(evolution: list,
                 overlap: ndarray,
                 energy: ndarray,
                 title: str,
                 filename=None):
    fig = plt.figure(layout='constrained', figsize=(10, 7))

    sub_figs = fig.subfigures(2)

    ax_img = sub_figs[0].subplots(ncols=15, nrows=3)
    offset = int(len(evolution) / 45)
    for idx in range(45):
        ax_img[int(idx / 15), idx % 15].imshow(evolution[offset * idx].reshape(32, -1).T)
        ax_img[int(idx / 15), idx % 15].axis('off')
    sub_figs[0].suptitle('Evolution of state')

    ax_line = sub_figs[1].subplots(ncols=2, sharex=True)
    sub_figs[1].suptitle("Overlap and Energy functions")

    ax_line[0].plot(overlap[:, 0], label="Number 0")
    ax_line[0].plot(overlap[:, 1], label="Number 1")
    ax_line[0].plot(overlap[:, 2], label="Number 2")
    ax_line[0].set_xlabel("Updates")
    ax_line[0].set_ylabel("Percentage of overlapping")
    ax_line[0].grid()
    ax_line[0].legend()
    ax_line[0].set_ylim(-1.1, 1.1)

    ax_line[1].plot(energy, label="Energy function")
    ax_line[1].set_xlabel("Updates")
    ax_line[1].set_ylabel("Energy value")
    ax_line[1].grid()
    ax_line[1].legend()

    plt.suptitle(title)

    if filename is not None:
        plt.savefig("imgs/" + filename)

    plt.show()


def plot_results2(evolution: list,
                  overlap: ndarray,
                  energy: ndarray,
                  title: str,
                  filename=None):
    fig = plt.figure(layout='constrained', figsize=(15, 3))
    sub_figs = fig.subfigures(ncols=2)

    axs_evl = sub_figs[0].subplots(ncols=len(evolution))
    for idx, state in enumerate(evolution):
        axs_evl[idx].imshow(np.expand_dims(state, 1))
        axs_evl[idx].axis("off")
        axs_evl[idx].set_title(str(idx))
    sub_figs[0].suptitle("Evolution of the state during the updates")

    ax_line = sub_figs[1].subplots(ncols=2, sharex=True)
    ax_line[0].plot(overlap[:, 0], label="Pattern 0")
    ax_line[0].plot(overlap[:, 1], label="Pattern 1")
    ax_line[0].plot(overlap[:, 2], label="Pattern 2")
    ax_line[0].set_xlabel("Updates")
    ax_line[0].set_ylabel("Percentage of overlapping")
    ax_line[0].grid()
    ax_line[0].legend()
    ax_line[0].set_ylim(-1.1, 1.1)

    ax_line[1].plot(energy, label="Energy function")
    ax_line[1].set_xlabel("Updates")
    ax_line[1].set_ylabel("Energy value")
    ax_line[1].grid()
    ax_line[1].legend()

    sub_figs[1].suptitle("Overlap & Energy function")

    plt.suptitle("Hello")
    plt.show()
