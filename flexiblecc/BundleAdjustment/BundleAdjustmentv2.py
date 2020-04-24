import numpy as np
import cv2
from central_model import CentralModel
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tqdm import tqdm


# FIX DUMB STUFF WITH GRID SIZE (CHECKERBOARD ALWAYS 12X12, BUT NOT B-SPLINE)

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
            
            #Normalize each ray
            rays[i,j] = np.divide(rays[i,j], np.linalg.norm(rays[i,j])) 

    return rays

current_iteration = 0
pbar = None
def calc_residuals( params, n_images, points_2D, grid_width, grid_height, n_grid_points, image_size, spline_grid_shape):

    global pbar, current_iteration
    n_params = params.shape[0]
    current_iteration = (current_iteration + 1) % n_params
    if current_iteration == 1:
        if pbar:
            pbar.close()
        pbar = tqdm(total=params.shape[0])
    if current_iteration % 5 == 0:
        pbar.update(5)


    spline_grid = params[:np.prod(spline_grid_shape)].reshape(spline_grid_shape)
    rvecs = params[np.prod(spline_grid_shape):][:n_images*3].reshape((94,3,1))
    tvecs = params[np.prod(spline_grid_shape) + n_images*3 :].reshape((94,3,1))

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


def grid_creation():

    step = 400

    border = 0
    image_width = 1600
    image_height = 1200


    x = np.arange(
        start=-border, 
        stop=image_width + border + step, 
        step=step,
        dtype=float)

    y = np.arange(
        start=-border, 
        stop=image_height + border + step, 
        step=step,
        dtype=float)

    src = np.transpose(np.meshgrid(x, y)).reshape((-1, 2))

    fx = 2.89099220e+03
    fy = 2.88229115e+03
    cx = 8.24326073e+02
    cy = 5.63283448e+02

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,τx,τy = 2.03288270e+00,  4.52548448e+01, -7.73226736e-03, 1.80779861e-02, -1.47886929e+02,  2.09683329e+00,  4.65342697e+01, -1.51419436e+02, -1.83082177e-02, -1.92297600e-03, 9.86103858e-03, 7.83822797e-03, 6.18311558e-03, 3.67951499e-02

    distCoeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([0, 0, 0])

    Rt = np.ndarray((3,4))
    Rt[:,:3] = R
    Rt[:,3] = t

    P = np.dot(K, Rt)

    dst = cv2.undistortPoints(
        src=src,
        cameraMatrix=K,
        distCoeffs=distCoeffs,
        P=P)[:,0,:]
        
    Kinv = np.linalg.inv(K)

    projected_pts = np.array([np.dot(Kinv, np.array([[pt[0],], [pt[1],], [1,]])) for pt in dst]).reshape((-1,3))

    return projected_pts


def get_sparsity_matrix(n_images, n_points, b_spline_shape):
    m = n_images * n_points * 3                     # number of residuals
    n = n_images * 6                                # number of parameters
    A = lil_matrix((m, n), dtype=int)

    residuals = np.arange(n_points*3)
    params = np.arange(6)

    for i in range(n):
        A[n_points*3*int(i/6)+residuals, i] = 1

    return A


def bundle_adjustment( rvecs, tvecs, n_images, points_2D, grid_width, grid_height, image_size, spline_grid_shape, sparsity=True):
    """ Optimizes camera parameters using bundle adjustment

    Arguments:
        cam_params: camera parameters for all cameras.
        points_3D: all 3D points.
        cam_indexs: index for the camera resulting in 2D point from the 3D point described by point_3D_indexs.
        point_3D_indexs: index for the 3D point resulting in the correct 2D point in the camera described by cam_indexs.
        points_2D: 2D points from all cameras.
    """
 
    n_grid_points = grid_width * grid_height

    start_grid = grid_creation()

    #Normalize start guess
    start_grid = np.divide(start_grid, np.linalg.norm(start_grid, axis=1).reshape((start_grid.shape[0],1)))

    params = np.hstack((start_grid.ravel(), rvecs.ravel(), tvecs.ravel()))

    residual_init = calc_residuals(params, n_images, points_2D, grid_width, grid_height, n_grid_points, image_size, spline_grid_shape)

    if sparsity:
        A = get_sparsity_matrix(n_images, points_2D.shape[1], spline_grid_shape)

        res = least_squares(calc_residuals, params, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                args=(n_images, points_2D, grid_width, grid_height, n_grid_points, image_size, spline_grid_shape))

    else:
        res = least_squares(calc_residuals, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                args=(n_images, points_2D, grid_width, grid_height, n_grid_points, image_size, spline_grid_shape))


                    
    return residual_init, res.fun, res.x



rvecs, tvecs, points_3D, points_2D = dict(np.load("savedOldSchool.npz")).values()

grid_width = 12
grid_height = 12
n_images = 94

image_size = (1600, 1200)

spline_grid_shape = (5, 4, 3)

res_int, res_fun, res_x = bundle_adjustment(rvecs, tvecs, n_images, points_2D, grid_width, grid_height, image_size, spline_grid_shape)

print("residual_init", res_int)

print("res_fun", res_fun)

print("res_x", res_x)

