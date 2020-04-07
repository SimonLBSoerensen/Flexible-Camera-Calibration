import numpy as np
from scipy.optimize import least_squares

def project(points_3D, cam_params):
    pass


def objective(params, n_params, n_cams, n_points_3D, cam_indexs, point_3D_indexs, points_2D):
    """ Computes residuals.
    Arguments:
        params: contains camera parameters and 3-D coordinates.
        n_params: parameters used to describe camera.
        n_cams: amount of cameras.
        n_points_3D: amount of points (144 for 12x12 checkerboard).
        cam_indexs: index for the camera resulting in 2D point from the 3D point described by point_3D_indexs.
        point_3D_indexs: index for the 3D point resulting in the correct 2D point in the camera described by cam_indexs.
        points_2D: 2D points from all cameras.
    """
    cam_params = params[:n_cams * n_params].reshape((n_cams, n_params))
    points_3D = params[n_cams * n_params:].reshape((n_points_3D, 3))

    # How I do dis?? Dunno.. Nobody knows
    points_2D_proj = project(points_3D[point_3D_indexs], cam_params[cam_indexs])
    
    residual = (points_2D_proj - points_2D).ravel()
    return residual

def bundle_adjustment(cam_params, points_3D, cam_indexs, point_3D_indexs, points_2D):
    """ Optimizes camera parameters using bundle adjustment

    Arguments:
        cam_params: camera parameters for all cameras.
        points_3D: all 3D points.
        cam_indexs: index for the camera resulting in 2D point from the 3D point described by point_3D_indexs.
        point_3D_indexs: index for the 3D point resulting in the correct 2D point in the camera described by cam_indexs.
        points_2D: 2D points from all cameras.
    """
    params = np.hstack((cam_params.ravel(), points_3D.ravel()))
    residual_init = objective(params, cam_params.shape[0], points_3D.shape[0], cam_indexs, point_3D_indexs, points_2D)

    res = least_squares(objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(cam_params.shape[0], points_3D.shape[0], cam_indexs, point_3D_indexs, points_2D))
                    
    return residual_init, res.fun, res.x