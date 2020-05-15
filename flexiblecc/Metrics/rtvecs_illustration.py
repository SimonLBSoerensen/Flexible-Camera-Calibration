import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def Rt_from_rvectvec(rvec, tvec):
    R, *_  = cv2.Rodrigues(rvec)

    Rt = np.zeros(shape=(4,4))
    Rt[:3, :3] = R
    Rt[:-1, 3] = tvec.flatten()
    Rt[-1,-1] = 1
    return Rt

def _do_homogeneous_Rt(p, Rt):
    if p.shape[0] == 3:
        temp = np.ones(4)
        temp[:3] = p
        p = temp
    assert p.shape[0] == 4, "The shape pf p has to be 3 or 4 for each point"

    new_p = np.matmul(Rt,p)

    new_p_uh = new_p[:3]
    new_p_uh = new_p_uh / new_p[-1]

    return new_p_uh

def do_homogeneous_Rt(p, Rt):
    if len(p.shape) == 1:
        new_ps = _do_homogeneous_Rt(p, Rt)
    else:
        new_ps = np.array([_do_homogeneous_Rt(p_el, Rt) for p_el in p])
    return new_ps


def draw_rtvecs(rvecs, tvecs, obj_points_all, draw_points=True, draw_surfaces=False, ax = None):
    if ax is None:
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    n_images = len(obj_points_all)

    transformed_points = []

    ax.scatter([0], [0], [0], marker='x', c='r')

    for i in range(n_images):
        Rt = Rt_from_rvectvec(rvecs[i], tvecs[i])

        transformed_points.append(do_homogeneous_Rt(obj_points_all[i], Rt))
        
        x = transformed_points[i][:,0]
        y = transformed_points[i][:,1]
        z = -transformed_points[i][:,2]

        if draw_points and draw_surfaces:
            pts = ax.scatter(x, y, z)
            c = pts.get_facecolor()[0]
            ax.plot_trisurf(x, y, z, color=c)

        elif draw_points:
            pts = ax.scatter(x, y, z)
        
        elif draw_surfaces:
            ax.plot_trisurf(x, y, z)

    return ax
