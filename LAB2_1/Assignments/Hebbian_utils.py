from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, corrcoef, argsort
from numpy.linalg import eig
from numpy.linalg.linalg import norm
from pandas import DataFrame


def principal_components(dataset: ndarray) -> Tuple[ndarray, ndarray]:
    # We perform the correlation matrix
    corr_matrix = corrcoef(dataset)
    # Perform the eigenvector and eigenvalues
    eig_values, eig_vectors = eig(corr_matrix)
    idx_1 = argsort(eig_values)[-1]  # first principal component
    idx_2 = argsort(eig_values)[-2]  # second principal component

    first_pc = eig_vectors[:, idx_1]
    second_pc = eig_vectors[:, idx_2]

    return first_pc, second_pc


# @title Plotting functions

# Oss. We suppose to have in "memory" first_pc & second_pc related to a first database
def plot_results(
        title: str,
        dataset: DataFrame,
        untr_W: ndarray,
        tr_W: ndarray,
        history_x1: list,
        history_x2: list,
        first_pc: tuple = None,
        second_pc: tuple = None,
        second_component: bool = False,
        filename: str = None):
    """
    :param title: title of the image
    :param dataset:
    :param untr_W: untrained weight
    :param tr_W: trained weight
    :param history_x1: evolution of weight x1 (during the updates)
    :param history_x2: evolution of weight x2 (during the updates)
    :param first_pc: First principal component
    :param second_pc: Second principal component
    :param second_component: True if we want to display the second component
    :param filename: if not none, it corresponds to the name of the image
    :return:
    """
    fig = plt.figure(figsize=(25, 17), constrained_layout=True)
    fig.suptitle(title)
    gs = fig.add_gridspec(3, 6)

    ax0 = fig.add_subplot(gs[0, 1:3])
    ax0.set_title('Scatter-plot of dataset with untrained vector and First PC')
    ax0.scatter(*dataset)
    ax0.axline((0, 0), slope=np.arctan(first_pc[1] / first_pc[0]), color='purple', lw=0.8, ls="--",
               label="Direction of First pc")
    ax0.quiver(0, 0, first_pc[0], first_pc[1], label="First principal components", color="green")

    if second_component:
        ax0.quiver(0, 0, second_pc[0], second_pc[1], label="Second principal components", color="red")
        ax0.axline((0, 0), slope=np.arctan(second_pc[1] / second_pc[0]), color='orange', lw=0.8, ls="--",
                   label="Direction of Second pc")

    ax0.quiver(0, 0, untr_W[0], untr_W[1], label="Untrained weights", color="blue")
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.axis("equal")
    ax0.grid()
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 3:5])
    ax1.set_title('Scatter-plot of dataset with trained vector and First PC')
    ax1.scatter(*dataset)
    ax1.axline((0, 0), slope=np.arctan(first_pc[1] / first_pc[0]), color='purple', lw=0.8, ls="--",
               label="Direction of First pc")
    ax1.quiver(0, 0, first_pc[0], first_pc[1], label="First principal components", color="green")

    if second_component:
        ax1.quiver(0, 0, second_pc[0], second_pc[1], label="Second principal components", color="red")
        ax1.axline((0, 0), slope=np.arctan(second_pc[1] / second_pc[0]), color='orange', lw=0.8, ls="--",
                   label="Direction of Second pc")

    ax1.quiver(0, 0, tr_W[0], tr_W[1], label="Trained weights", color="blue")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.axis("equal")
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.set_title('Evolution of the weight X1 through the upgrades')
    ax2.plot(history_x1, label="x1")
    ax2.set_xlabel("Upgrades")
    ax2.set_ylabel("Values")
    ax2.grid()
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.set_title('Evolution of the weight X2 through the upgrades')
    ax3.plot(history_x2, label="x2")
    ax3.set_xlabel("Upgrades")
    ax3.set_ylabel("Values")
    ax3.grid()
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 4:6])
    ax4.set_title('Evolution of the norm of weight through the upgrades')
    ax4.plot([np.linalg.norm(np.array([a, b])) for (a, b) in zip(history_x1, history_x2)],
             label="norm(W)")
    ax4.set_xlabel("Upgrades")
    ax4.set_ylabel("Values")
    ax4.grid()
    ax4.legend()

    if filename is not None:
        plt.savefig("imgs/" + filename)

    plt.show()


# Oss. We suppose to have in "memory" first_pc & second_pc related to second database
def plot_results2(
        title: str,
        dataset: DataFrame,
        untr_W: ndarray,
        tr_W: ndarray,
        history_x1: list,
        history_x2: list,
        first_pc2: tuple = None,
        second_pc2: tuple = None,
        second_component: bool = False,
        filename: str = None):
    """
    :param title: title of the image
    :param dataset:
    :param untr_W: untrained weight
    :param tr_W: trained weight
    :param history_x1: evolution of weight x1 (during the updates)
    :param history_x2: evolution of weight x2 (during the updates)
    :param first_pc2: First principal component
    :param second_pc2: Second principal component
    :param second_component: True if we want to display the second component
    :param filename: if not none, it corresponds to the name of the image
    """
    fig = plt.figure(figsize=(25, 17), constrained_layout=True)
    fig.suptitle(title)
    gs = fig.add_gridspec(3, 6)

    origin_x, origin_y = 10, 2000

    ax0 = fig.add_subplot(gs[0, 1:3])
    ax0.set_title('Scatter-plot of dataset with untrained vector and First PC')
    ax0.scatter(*dataset)
    ax0.quiver(origin_x, origin_y, first_pc2[0], first_pc2[1], label="First principal components", color="green")

    if second_component:
        ax0.quiver(origin_x, origin_y, second_pc2[0], second_pc2[1], label="Second principal components", color="red")

    ax0.quiver(origin_x, origin_y, untr_W[0], untr_W[1], label="Untrained weights", color="blue")
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.grid()
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 3:5])
    ax1.set_title('Scatter-plot of dataset with trained vector and First PC')
    ax1.scatter(*dataset)
    ax1.quiver(origin_x, origin_y, first_pc2[0], first_pc2[1], label="First principal components", color="green")

    if second_component:
        ax1.quiver(origin_x, origin_y, second_pc2[0], second_pc2[1], label="Second principal components", color="red")

    ax1.quiver(origin_x, origin_y, tr_W[0], tr_W[1], label="Trained weights", color="blue")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.set_title('Evolution of the weight X1 through the upgrades')
    ax2.plot(history_x1, label="x1")
    ax2.set_xlabel("Upgrades")
    ax2.set_ylabel("Values")
    ax2.grid()
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.set_title('Evolution of the weight X2 through the upgrades')
    ax3.plot(history_x2, label="x2")
    ax3.set_xlabel("Upgrades")
    ax3.set_ylabel("Values")
    ax3.grid()
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 4:6])
    ax4.set_title('Evolution of the norm of weight through the upgrades')
    ax4.plot([norm(np.array([a, b])) for (a, b) in zip(history_x1, history_x2)],
             label="norm(W)")
    ax4.set_xlabel("Upgrades")
    ax4.set_ylabel("Values")
    ax4.grid()
    ax4.legend()

    if filename is not None:
        plt.savefig("imgs/" + filename)

    plt.show()
