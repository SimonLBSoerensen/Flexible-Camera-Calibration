import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import os
import flexiblecc as fcc

datasetpath = "../../CalImgs/cam_0*"


import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

'''
color_cal_imgs_files = glob.glob(datasetpath)
color_cal_imgs = np.array([cv2.imread(f) for f in tqdm(color_cal_imgs_files)])
gray_cal_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in tqdm(color_cal_imgs)]


pattern_size = (12, 12)


retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, points_3D, points_2D = fcc.OldSchoolCC.calibrate_camera(gray_cal_imgs, pattern_size, verbose=1)

np.savez("savedOldSchool", rvecs, tvecs, points_3D, points_2D)
'''

rvecs, tvecs, points_3D, points_2D = dict(np.load("savedOldSchool.npz")).values()

bad_images = np.array([4,5,6,7,8,17,18,19,20,21,22,23,35,36,37,38,40,41,42,43,44,53,54,55,56,61,62,63,64,68,73,74,75,81,82,83,84,85,93,94,39])
bad_images = bad_images - 1

rvecs = np.array(rvecs)
tvecs = np.array(tvecs)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

cam1_points_3D = np.reshape(points_3D[0], (-1,3))

ax.plot3D(cam1_points_3D[:,0], cam1_points_3D[:,1], cam1_points_3D[:,2], 'bo')
for i in range(len(points_3D)):

    #if i in bad_images:
    #ax.plot3D(tvecs[i][0], tvecs[i][1], tvecs[i][2], 'ro')
    if(i == 38):
        ax.plot3D(-tvecs[i][0], -tvecs[i][1], -tvecs[i][2], 'gx')
    else:
        ax.plot3D(-tvecs[i][0], -tvecs[i][1], -tvecs[i][2], 'rx')

plt.show()


n_images = 94
grid_width = 12
grid_height = 12
grid_size = grid_width * grid_height

grid = np.zeros((grid_size, 3))
meshgrid = np.transpose(np.meshgrid(np.arange(0,12),np.arange(0,12)))

meshgrid = np.reshape(meshgrid, (grid_size, 2))

grid[:, :-1] = meshgrid
grid = np.reshape(grid, grid.shape+(1,))

forward_projected_rays = np.ndarray((n_images, grid_size, 3, 1))
for i,(r,t) in enumerate(zip(rvecs, tvecs)):
    rotation_matrix = cv2.Rodrigues(r)[0]
    for j in range(grid_size):
        forward_projected_rays[i,j] = np.dot(rotation_matrix.T, grid[j]-t)



ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


ax.plot3D([0],[0],[0], 'rx')
#for i, c in enumerate(["b", "y", "g", "r", "c"]):
for i in range(len(forward_projected_rays)):

    #if i in bad_images:

    #draw normals
    vector1 = np.reshape(forward_projected_rays[i][12*5+5] - forward_projected_rays[i][12*5+6], (3,))
    vector2 = np.reshape(forward_projected_rays[i][12*5+5] - forward_projected_rays[i][12*6+5], (3,))
    normalvector = np.cross(vector1, vector2)
    pa = forward_projected_rays[i][12*5+5].reshape((3,))
    pb = pa - normalvector * 100
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],zs=[pa[2], pb[2]])

    #draw planes
    cam1_points_3D = np.reshape(forward_projected_rays[i], (-1,3))
    ax.plot_trisurf(cam1_points_3D[:,0], cam1_points_3D[:,1], cam1_points_3D[:,2])
    
axisEqual3D(ax)
plt.show()