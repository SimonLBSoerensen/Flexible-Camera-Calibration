import numpy as np
import cv2
from scipy.optimize import least_squares


def calc_residuals( params, rvecs, tvecs, n_params, n_cams, points_2D ):
    #backward_projected_rays = b_spline_object.getRays(points_2D)
    
    points_3D = params[n_params*n_cams : ].reshape(-1,3)

    grid = np.zeros((12, 12, 3))
    meshgrid = np.transpose(np.meshgrid(np.arange(0,12),np.arange(0,12)))

    grid[:, :, 1:] = meshgrid


    forward_projected_rays = np.ndarray((94, 144))
    for r, t in zip(rvecs, tvecs):
        rotation_matrix = cv2.rodrigues(r)
        forward_projected_rays = grid * rotation_matrix + t

    forward_projected_rays = cam_extrinsic * grid
    print(forward_projected_rays)

'''
def bundle_adjustment( b_spline_object, cam_extrinsic, points_2D, points_3D ):
    """ Optimizes camera parameters using bundle adjustment

    Arguments:
        cam_params: camera parameters for all cameras.
        points_3D: all 3D points.
        cam_indexs: index for the camera resulting in 2D point from the 3D point described by point_3D_indexs.
        point_3D_indexs: index for the 3D point resulting in the correct 2D point in the camera described by cam_indexs.
        points_2D: 2D points from all cameras.
    """
    params = np.hstack((b_spline_object.params.ravel(), points_3D.ravel()))
    residual_init = calc_residuals(params, b_spline_object.params.shape[0], points_3D.shape[0], points_2D)

    res = least_squares(calc_residuals, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(cam_params.shape[0], points_3D.shape[0], cam_indexs, point_3D_indexs, points_2D))
                    
    return residual_init, res.fun, res.x
'''

grid = np.zeros((12, 12, 3))
meshgrid = np.transpose(np.meshgrid(np.arange(0,12),np.arange(0,12)))

grid[:, :, 1:] = meshgrid

print(grid)
'''
for i in range(0,12):
    for j in range(0,12):
        grid = np.concatenate(grid, [0, i, j])

print(grid.shape)
forward_projected_rays = np.multiply(np.array([[0,1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1,0]]), grid)
print(forward_projected_rays)'''