import numpy as np
from collections.abc import Callable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_distort(undistort_func, image_size, step=10, contour_n_levels=None, save_f_string=None):
    """
    Plots the distorsion discrept width distCoeffs
    Arguments:
        undistort_func (function): A function with argument point coordinates, Nx2 floats. It has to return the undistoret points as a Nx2 array with the same order of points
        image_size (tuple): The size of the image from the camera of length 2
        step (int): Grid step size for undistorted calculations over the image
        contour_n_levels (int/array): Determines the number and positions of the contour lines / regions. If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen. If array-like, draw contour lines at the specified levels. The values must be in increasing order.
        save_f_string (str): The f string to use when saving
    """
    assert len(image_size) == 2 and isinstance(image_size, tuple), "image_size has to be a tuple of length 2"
    assert isinstance(undistort_func, Callable), "undistortFunc has to be callable"

    # Make a grid of point there spread over the image
    gridrange = (np.arange(image_size[0], step=step), np.arange(image_size[1], step=step))
    points = np.array(np.meshgrid(*gridrange)).transpose().reshape(-1, 2).astype(np.float32)

    # Calculate the undistorted points
    undistort_points = undistort_func(points)

    # Calculate the difference between the points and there undistorted counter part
    diff = points - undistort_points

    # Calculate the euclidean for the difference
    errors = np.linalg.norm(diff, axis=1)

    # Reshape the errors to the image grid size
    X, Y = np.mgrid[0:len(gridrange[0]), 0:len(gridrange[1])]
    errors_grid = errors.reshape(X.shape)

    # Plot the euclidean distance in pixels between distorted and undistorted
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.title("Euclidean distance in pixels between distorted and undistorted")
    im = plt.imshow(errors.reshape([len(g) for g in gridrange]))
    plt.xticks([], [])
    plt.yticks([], [])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 2, 2)
    plt.title("Euclidean distance in pixels between distorted and undistorted contour")
    CS = plt.contour(Y, X, errors_grid, levels=contour_n_levels)
    plt.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')

    plt.xticks([], [])
    plt.yticks([], [])

    asp = np.diff(CS.ax.get_xlim())[0] / np.diff(CS.ax.get_ylim())[0]
    asp /= np.abs(np.diff(im.axes.get_xlim())[0] / np.diff(im.axes.get_ylim())[0])
    CS.ax.set_aspect(asp)

    #plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    if save_f_string is not None:
        plt.savefig(save_f_string.format("distance"))
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(20, 10))

    # Find the angels the distorted points are moved with
    angels = np.array([np.angle(el[0] + el[1] * 1j, deg=True) for el in diff])

    # Reshape the angels to the image grid size
    X, Y = np.mgrid[0:len(gridrange[0]), 0:len(gridrange[1])]
    angels_grid = angels.reshape(X.shape)

    # Plot the angle between distorted and undistorted
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.title("Angle between distorted and undistorted [deg]")
    im = plt.imshow(angels.reshape([len(g) for g in gridrange]), cmap="hsv")
    plt.xticks([], [])
    plt.yticks([], [])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 2, 2)
    plt.title("Angle between distorted and undistorted contour [deg]")
    CS = plt.contour(Y, X, angels_grid, levels=contour_n_levels)
    plt.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')

    plt.xticks([], [])
    plt.yticks([], [])

    asp = np.diff(CS.ax.get_xlim())[0] / np.diff(CS.ax.get_ylim())[0]
    asp /= np.abs(np.diff(im.axes.get_xlim())[0] / np.diff(im.axes.get_ylim())[0])
    CS.ax.set_aspect(asp)

    plt.gca().invert_yaxis()

    if save_f_string is not None:
        plt.savefig(save_f_string.format("angle"))
        plt.close()
    else:
        plt.show()