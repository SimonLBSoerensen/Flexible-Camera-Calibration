import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def maha_dist(x, mean, cov):
    """
    calgulates the malahonobis distance

    :param x: The point
    :param mean: The mean vector
    :param cov:  The covarian matrix
    :return: The malahonobis distance
    """
    return np.matmul(np.matmul((np.array(x) - mean).transpose(), np.linalg.inv(cov)), (np.array(x) - mean))

def maha_dist_arr(arr, mean, cov):
    """
    calgulates the malahonobis distance

    :param arr: The array of points
    :param mean: The mean vector
    :param cov:  The covarian matrix
    :return: The malahonobis distance
    """
    return [maha_dist(x, mean, cov) for x in arr]


def mean_cov(arr):
    """
    Calgulates the mean vector and covarian matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return mean: The mean vector
    :return cov: The covarian matrix
    """
    cov = np.cov(arr.T)
    mean = np.mean(arr, axis=0)
    return mean, cov


def plot_3DMVN(mean, cov, start, end, res):
    """
    Plots a 3D MVN. Only works for 1x2 mean and 2x2 cov
    :param mean: The mean vector
    :param cov: The covarian matrix
    :param start: The sample space start
    :param end: The sample space end
    :param res: The sample space resulusion
    :return: The axis for the plot
    """
    assert len(mean) == 2, "plot_3DMVN can only handle p=2 MVN's"

    x = np.linspace(start, end, res)
    y = np.linspace(start, end, res)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal(mean, cov)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    return ax


def plot_qqplot(arr, ax=None):
    """
    Makes a QQ-Plot in relations to chi^2
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return: The axis for the plot
    """
    p = arr.shape[1]
    if ax is None:
        plt.figure()
        ax = plt.gca()

    mean, cov = mean_cov(arr)
    dists = maha_dist_arr(arr, mean, cov)
    (osm, osr), (slope, intercept, r) = stats.probplot(x=dists, dist=stats.chi2(p), plot=ax)
    ax.set_title(
        "qq-plot for $d_j^2$ versus ${\chi}_" + str(arr.shape[1]) + "^2$ distribution, $R^2$=" + "{:.04f}".format(
            r ** 2))
    ax.set_xlabel("Quantiles for $d_j^2$")
    ax.set_ylabel("Quantiles for ${\chi}_" + str(arr.shape[1]) + "^2$ distribution")
    return ax


def plot_box(arr, ax=None):
    """
    Makes a Box plot
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return: The axis for the plot
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.boxplot(arr)
    ax.set_title("Boxplots")
    ax.set_xlabel("Boxplot for the i'th dimensional variable")
    return ax


def plot_scatter_matrix(arr, ax_info=None):
    """
    Makes a Scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :param ax_info: [ax, ax_pos], where ax is the axis to plot on and ax_pos is which plot to make (i vs j demisions)
    :return: The axis for the plot
    """
    p = arr.shape[1]

    if ax_info is None:
        fig, axs = plt.subplots(p, p)
        for i in range(0, p):
            for j in range(0, p):
                data_i = arr[:, i]
                data_j = arr[:, j]

                if i == j:
                    axs[i, j].hist(data_i)
                else:
                    axs[i, j].scatter(data_i, data_j)

        return axs
    else:
        ax, ax_pos = ax_info
        i, j = ax_pos
        data_i = arr[:, i]
        data_j = arr[:, j]
        if i == j:
            ax.hist(data_i)
        else:
            ax.scatter(data_i, data_j)
        return ax


def _plot_model_check_woscatter(arr):
    """
    Makes a model check plot with out a scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return: The axis for the plot
    """
    fig, axs = plt.subplots(2, 1)
    plot_box(arr, axs[0])
    plot_qqplot(arr, axs[1])
    plt.tight_layout()
    return axs


def _plot_model_check_wscatter(arr):
    """
    Makes a model check plot with a scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return: The axis for the plot
    """
    p = arr.shape[1]

    rows = p if p % 2 == 0 else p + 1
    cols = p * 2

    box_pos = (0, p)
    box_rowspan = rows // 2
    box_colspan = p

    qq_pos = (box_rowspan, p)
    qq_rowspan = rows // 2
    qq_colspan = p

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(rows, cols)
    axs = []

    for i in range(p):
        for j in range(p):
            ax = fig.add_subplot(gs[i, j])
            ax = plot_scatter_matrix(arr, [ax, [i, j]])
            axs.append(ax)

    ax = fig.add_subplot(gs[box_pos[0]:box_pos[0] + box_rowspan, box_pos[1]:box_pos[1] + box_colspan])
    ax = plot_box(arr, ax)
    axs.append(ax)

    ax = fig.add_subplot(gs[qq_pos[0]:qq_pos[0] + qq_rowspan, qq_pos[1]:qq_pos[1] + qq_colspan])
    ax = plot_qqplot(arr, ax)
    axs.append(ax)

    return axs


def plot_model_check(arr, with_scatter=True):
    """
    Makes a model check plot
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :param  with_scatter: Indicats if a scatter matrix shoud be plottet in the model check
    :return: The axis for the plot
    """
    if with_scatter:
        axs = _plot_model_check_wscatter(arr)
    else:
        axs = _plot_model_check_woscatter(arr)
    return axs


def one_sample_hotelling_t2(arr, mean_to_test):
    """
    Makes a one-sample hotelling T^2 test
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :param mean_to_test: The mean vector to test for
    :return p_value: The p_value for the test
    """
    # TODO: anden person gennem gå enninghed i matematikken

    n = arr.shape[0]
    p = arr.shape[1]

    mean, cov = mean_cov(arr)
    mean_diff = (mean - mean_to_test)

    temp = np.matmul(mean_diff.T, np.linalg.inv(cov / n))
    test_statistic = np.matmul(temp, mean_diff)

    scale = (n - p) / (p * (n - 1))
    p_value = 1 - stats.f.cdf(scale * test_statistic, p, n - p)  # CDF = P(X <= x) -> 1-CDF = P(X > x)
    return p_value


def no_systematic_error_test(arr):
    """
    Tests if there is a systematic error. By makeing a one-sample hotelling T^2 test vs a mean vector with zeros
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of diamations)
    :return p_value: The p_value for the test
    """
    return one_sample_hotelling_t2(arr, mean_to_test=0.0)
