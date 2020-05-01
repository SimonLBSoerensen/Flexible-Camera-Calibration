from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import cv2

def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Taken from: https://gist.github.com/pv/8036995 (Author: Pauli Virtanen)

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _color_angle(values, deg=True):
    if not deg:
        values = np.rad2deg(values)

    if not isinstance(values, Iterable):
        values = [values]

    if len(values) == 1:
        values = [values[0], 360.0, 360.0]
    if len(values) == 2:
        values = [values[0], values[1], 360.0]

    assert len(values) == 3, "The len(values) has to be 1<=len(values)<=3"
    values = np.array(values) / np.array([1.0, 360.0, 360.0])
    H, V, S = values

    Hm = H / 60

    C = V * S
    X = C * (1 - abs(Hm % 2 - 1))

    if 0 <= Hm <= 1:
        ms = [C, X, 0]
    elif 1 < Hm <= 2:
        ms = [X, C, 0]
    elif 2 < Hm <= 3:
        ms = [0, C, X]
    elif 3 < Hm <= 4:
        ms = [0, X, C]
    elif 4 < Hm <= 5:
        ms = [X, 0, C]
    elif 5 < Hm <= 6:
        ms = [C, 0, X]
    else:
        ms = [0, 0, 0]

    Rm, Gm, Bm = ms

    m = V - C

    RGB = np.array([Rm + m, Gm + m, Bm + m])
    RGB = np.round(RGB * 255).astype(int)

    return RGB


def cal_angles_and_mag(image_points, obj_points, rvecs, tvecs, cameraMatrix, distCoeffs):
    """
    Calgulates the angels and magnitude for the projectet points in relation to the feature points
    :param image_points: The points object points are found in the image
    :param obj_points: The object points
    :param rvecs (ndarray): Rotation vectors estimated for each pattern view
    :param tvecs (ndarray): Translation vectors estimated for each pattern view
    :param cameraMatrix: 3x3 floating-point camera matrix
            [[fx, 0,  cx],
             [0,  fy, cy],
    :param distCoeffs: vector of distortion coefficients of 4, 5, 8, 12 or 14 elements
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    :return img_points: The image points in a flat list
    :return diff: The difrence for the points as a (nx2) list
    :return angels: The angels for each of the points
    :return mag: The L2 magnitute for each of the points
    """
    img_points_all = []
    angels_all = []
    diff_all = []
    mag_all = []

    for img_ps, obj_ps, rvec, tvec in zip(image_points, obj_points, rvecs, tvecs):
        real_img_points = img_ps.reshape(-1, 2)

        repor_img_points = cv2.projectPoints(obj_ps, rvec, tvec, cameraMatrix, distCoeffs)[0].reshape(-1, 2)
        diff = real_img_points - repor_img_points

        angels = np.array([np.angle(el[0] + el[1] * 1j, deg=True) for el in diff])
        neg_angels = np.where(angels < 0)[0]
        angels[neg_angels] = 360 + angels[neg_angels]

        mag = np.linalg.norm(diff, axis=1)

        img_points_all.append(real_img_points)
        diff_all.append(diff)
        angels_all.append(angels)
        mag_all.append(mag)

    diff_all = np.concatenate(diff_all).reshape(-1, 2)
    img_points_all = np.concatenate(img_points_all).reshape(-1, 2)
    angels_all = np.concatenate(angels_all).flatten()
    mag_all = np.concatenate(mag_all).flatten()
    return img_points_all, diff_all, angels_all, mag_all


def plot_voronoi(points, points_angle, deg=True, ax=None, xy_lim=True, radius=None, plot_points=False):
    """
    Plots voronoi diagram for the given points
    """
    if ax is None:
        ax = plt.gca()

    if plot_points:
        ax.plot(points[:, 0], points[:, 1], "o")

    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=radius)

    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), color=_color_angle(points_angle[i], deg=deg) / 255)

    if xy_lim:
        ax.set_xlim([vor.min_bound[0], vor.max_bound[0]])
        ax.set_ylim([vor.min_bound[1], vor.max_bound[1]])

    return ax
