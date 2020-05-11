import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import itertools


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
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
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
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
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


def plot_box(arr, ax=None, violinplot=False):
    """
    Makes a Box plot
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :return: The axis for the plot
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if not violinplot:
        ax.boxplot(arr)
        ax.set_title("Boxplots")
        ax.set_xlabel("Boxplot for the i'th dimensional variable")
    else:
        ax.violinplot(arr, showmedians=True)
        ax.set_title("Violinplot")
        ax.set_xlabel("Violinplot for the i'th dimensional variable")

    return ax


def plot_seaborn_scatter_matrix(arr, kde=True, names=None):
    """
    Makes a seaborn based scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :param kde: Rather or not to ues kernel density estimation in the diagronal plots
    :param names: A list of the names of the p dimensions. If None will be named "Dimension i"
    :return: The axis for the plot
    """
    axs = sns.pairplot(pd.DataFrame(arr, columns=["Dimension {}".format(i) for i in
                                                  range(1, arr.shape[1] + 1)] if names is None else names),
                       diag_kind="kde" if kde else "hist", kind="reg", markers="+")
    return axs


def plot_matplotlib_scatter_matrix(arr, ax_info=None):
    """
    Makes a matplotlib based scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
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


def _plot_model_check_in_one_wscatter(arr, violinplot=False):
    """
    Makes a model check plot with a scatter matrix
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
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
            ax = plot_matplotlib_scatter_matrix(arr, [ax, [i, j]])
            axs.append(ax)

    ax = fig.add_subplot(gs[box_pos[0]:box_pos[0] + box_rowspan, box_pos[1]:box_pos[1] + box_colspan])
    ax = plot_box(arr, ax, violinplot=violinplot)
    axs.append(ax)

    ax = fig.add_subplot(gs[qq_pos[0]:qq_pos[0] + qq_rowspan, qq_pos[1]:qq_pos[1] + qq_colspan])
    ax = plot_qqplot(arr, ax)
    axs.append(ax)

    return axs


def plot_model_check(arr, in_one=False, names=None, violinplot=False, save_f_string=None, **saveargs):
    """
    Makes a model check plot
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :param names: A list of the names of the p dimensions. If None will be named "Dimension i". Only in use when in_one=False
    """
    if in_one:
        _plot_model_check_in_one_wscatter(arr, violinplot=violinplot)
    else:
        plot_seaborn_scatter_matrix(arr, names=names)
        if save_f_string:
            plt.savefig(save_f_string.format("scatter_matrix"), **saveargs)
        plt.show()

        plt.figure()
        ax = plt.gca()
        plot_box(arr, ax, violinplot=violinplot)
        if save_f_string:
            plt.savefig(save_f_string.format("box"), **saveargs)
        plt.show()

        plt.figure()
        ax = plt.gca()
        plot_qqplot(arr, ax)
        if save_f_string:
            plt.savefig(save_f_string.format("qqplot"), **saveargs)
        plt.show()


def one_sample_hotelling_t2(arr, mean_to_test):
    """
    Makes a one-sample hotelling T^2 test
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :param mean_to_test: The mean vector to test for
    :return p_value: The p_value for the test
    """
    n = arr.shape[0]
    p = arr.shape[1]

    mean, cov = mean_cov(arr)
    mean_diff = (mean - mean_to_test)

    temp = np.matmul(mean_diff.T, np.linalg.inv(cov / n))
    test_statistic = np.matmul(temp, mean_diff)

    scale = (n - p) / (p * (n - 1))
    p_value = 1 - stats.f.cdf(scale * test_statistic, p, n - p)  # CDF = P(X <= x) -> 1-CDF = P(X > x)
    return p_value


def zero_mean_test(arr):
    """
    Tests if there is a systematic error. By makeing a one-sample hotelling T^2 test vs a mean vector with zeros
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :return p_value: The p_value for the test
    """
    return one_sample_hotelling_t2(arr, mean_to_test=0.0)


def plot_p_values(p_values, critical_value=None, x_labels=None, ax=None):
    """
    Plots p-values with a stem plot
    :param p_values: The p-values to plot
    :param critical_value: If given a horizontal line is made at the critical_value
    :param x_labels: What the p-values describes. If None 1 to len(p_values) is used
    :return p_value: The p_value for the test
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    p_values_x = np.arange(len(p_values))
    y_ticks = list(np.arange(0, 1.1, 0.1))
    container_stem = ax.stem(p_values_x, p_values, use_line_collection=True, bottom=0, label="p-values")

    if critical_value is not None:
        left, right = ax.get_xlim()
        ax.hlines(critical_value, left, right, label="Critical value: {}".format(critical_value))
        ax.set_xlim(left, right)
        y_ticks += [critical_value]

    if x_labels is not None:
        left, right = ax.get_xlim()
        locs, labels = plt.xticks()
        x_labels = [""] + list(x_labels) + [""]
        plt.xticks(ticks=locs, labels=x_labels, rotation=45)
        ax.set_xlim(left, right)

    ax.set_ylabel("p-value")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.set_yticks(y_ticks)
    plt.grid(True)
    return ax


def correlation_test(a, b, nonparametric):
    """
    Test for correlation. If nonparametric Spearman correlation test else Pearson correlation test
    :param a: First data arrray (one diamation)
    :param b: Secend data array (one diamation)
    :param nonparametric: Reather or not a nonparametric test has to be used. If not assumption that each dataset is normally distributed is made.
    :return p_value: The p_values
    """
    if nonparametric:
        correlation_coefficient, p_value = spearmanr(a, b)  # nonparametric
    else:
        correlation_coefficient, p_value = pearsonr(a, b)  # parametric: normally distributed
    return correlation_coefficient, p_value


def arr_correlation_test(arr, nonparametric, plot=False, plot_critical_value=None, plot_ax=None):
    """
    Test for correlation. If nonparametric Spearman correlation test else Pearson correlation test
    :param arr: The input data in the shape of (n, p) (n the number of samples, p the number of dimensions)
    :param nonparametric: Reather or not a nonparametric test has to be used. If not assumption that each dataset is normally distributed is made.
    :param plot: Wather or not to plot the p-values
    :param critical_value: If given a horizontal line is made at the critical_value
    :param plot_ax: The axis for the plot if None a new plot axis is made
    :return p_value: The p_values
    :return ax: The axis for the plot. Is only returnt if plot=True
    """
    p = arr.shape[1]

    all_combos_idx = list(itertools.combinations(np.arange(p), 2))
    res = {}

    for a_i, b_i in all_combos_idx:
        _, p_value = correlation_test(arr[:, a_i], arr[:, b_i], nonparametric)
        res[(a_i, b_i)] = p_value

    if plot:
        p_values = list(res.values())
        x_labels = list(map(str, list(res.keys())))
        ax = plot_p_values(p_values, plot_critical_value, x_labels, ax=plot_ax)
        ax.set_title("Correlation test")
        ax.set_xlabel("p-value for correlation between (i'th, j'th) dimensions")
        return (res, ax)
    else:
        return (res)
