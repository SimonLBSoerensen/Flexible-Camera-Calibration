import numpy as np
import cv2
from central_model import CentralModel, grid_creation
from scipy.optimize import least_squares
from tqdm import tqdm


def forward_project(n_images, rvecs, tvecs, grid_width, grid_height, n_grid_points):

    grid = np.zeros((n_grid_points, 3))
    meshgrid = np.transpose(np.meshgrid(np.arange(grid_width),np.arange(grid_height)))

    meshgrid = np.reshape(meshgrid, (n_grid_points, 2))

    grid[:, :-1] = meshgrid
    grid = np.reshape(grid, grid.shape+(1,))

    rays = np.ndarray((n_images, n_grid_points, 3, 1))
    for i, r, t in zip(range(n_images), rvecs, tvecs):
        rotation_matrix = cv2.Rodrigues(r)[0]
        for j in range(n_grid_points):
            rays[i,j] = np.dot(rotation_matrix.T, grid[j]-t)

    return rays

current_iteration = 0
pbar = None
def calc_residuals( params, n_images, points_2D, grid_width, grid_height, n_grid_points, image_size):

    global pbar, current_iteration
    n_params = params.shape[0]
    current_iteration = (current_iteration + 1) % n_params
    if current_iteration == 1:
        if pbar:
            pbar.close()
        pbar = tqdm(total=params.shape[0])
    if current_iteration % 20 == 0:
        pbar.update(20)

    spline_grid = params[:n_grid_points*3].reshape((grid_width, grid_height,3))
    rvecs = params[n_grid_points*3:][:n_images*3].reshape((94,3,1))
    tvecs = params[n_grid_points*3 + n_images*3 :].reshape((94,3,1))

    b_spline_object = CentralModel(image_size, image_size, spline_grid, 2)

    #Calculate forward projected rays
    forward_projected_rays = forward_project(n_images, rvecs, tvecs, grid_width, grid_height, n_grid_points)

    #Calculate backward projected rays (b_spline)
    model_rays = np.ndarray((n_images, n_grid_points, 3))

    for i, image_points in zip(range(n_images), points_2D):
        for j, point_2D in zip(range(n_grid_points), image_points):
            model_rays[i,j] = b_spline_object.sample(point_2D[0,0], point_2D[0,1])

    model_rays = model_rays.reshape(model_rays.shape+(1,))
    return (forward_projected_rays - model_rays).ravel()


def bundle_adjustment( rvecs, tvecs, n_images, points_2D, grid_width, grid_height, image_size):
    """ Optimizes camera parameters using bundle adjustment

    Arguments:
        cam_params: camera parameters for all cameras.
        points_3D: all 3D points.
        cam_indexs: index for the camera resulting in 2D point from the 3D point described by point_3D_indexs.
        point_3D_indexs: index for the 3D point resulting in the correct 2D point in the camera described by cam_indexs.
        points_2D: 2D points from all cameras.
    """

    n_grid_points = grid_width * grid_height

    start_grid = grid_creation((grid_width, grid_height), image_size)
    
    params = np.hstack((start_grid.ravel(), rvecs.ravel(), tvecs.ravel()))

    residual_init = calc_residuals(params, n_images, points_2D, grid_width, grid_height, n_grid_points, image_size)

    res = least_squares(calc_residuals, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_images, points_2D, grid_width, grid_height, n_grid_points, image_size))
                    
    return residual_init, res.fun, res.x


rvecs, tvecs, points_3D, points_2D = dict(np.load("savedOldSchool.npz")).values()

grid_width = 4
grid_height = 3
n_images = 94

image_size = (1600, 1200)

bundle_adjustment(rvecs, tvecs, n_images, points_2D, grid_width, grid_height, image_size)