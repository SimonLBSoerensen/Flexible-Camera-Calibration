import numpy as np
import cv2
import time, datetime
import json
import pickle
from matplotlib import pyplot as plt
from flexiblecc.BSpline.central_model import CentralModel, fit_central_model
from flexiblecc.BundleAdjustment import initialization
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from socket import gethostname
from tqdm import tqdm
from sys import argv


def forward_project(n_images, rvecs, tvecs, checkerboard_points):
    # TODO: make global, don't run all the time

    rays = np.ndarray((n_images, checkerboard_points.shape[1], 3, 1))
    for i, r, t in zip(range(n_images), rvecs, tvecs):
        rotation_matrix = cv2.Rodrigues(r)[0]
        for j in range(checkerboard_points.shape[1]):
            rays[i, j] = np.dot(rotation_matrix, checkerboard_points[i,j]) + t

            # Normalize each ray
            rays[i, j] = np.divide(rays[i, j], np.linalg.norm(rays[i, j]))

    return rays


def residuals_3d(parameters, p, n_images, points_2D, image_size, cm_shape, checkerboard_points):
    spline_grid = parameters[:np.prod(cm_shape)].reshape(cm_shape)
    rvecs = parameters[np.prod(cm_shape):][:n_images * 3].reshape((94, 3, 1))
    tvecs = parameters[np.prod(cm_shape) + n_images * 3:].reshape((94, 3, 1))

    b_spline_object = CentralModel(
        image_dimensions=image_size,
        grid_dimensions=image_size,
        control_points=spline_grid,
        order=p['cm_order'],
        knot_method=p['cm_knot_method'],
        min_basis_value=p['cm_min_basis_value'],
        end_divergence=p['cm_end_divergence'])

    # Calculate forward projected rays
    forward_projected_rays = forward_project(n_images, rvecs, tvecs, checkerboard_points).ravel()

    # Calculate backward projected rays (b_spline)
    model_rays = np.ndarray((n_images, checkerboard_points.shape[1], 3))
    for i, image_points in zip(range(n_images), points_2D):
        for j, point_2D in zip(range(checkerboard_points.shape[1]), image_points):
            model_rays[i, j] = b_spline_object.sample(point_2D[0, 0], point_2D[0, 1])

    model_rays = model_rays.ravel()

    return forward_projected_rays - model_rays

def residuals_2d(parameters, p, n_images, points_2D, image_size, cm_shape, checkerboard_points):
    spline_grid = parameters[:np.prod(cm_shape)].reshape(cm_shape)
    rvecs = parameters[np.prod(cm_shape):][:n_images * 3].reshape((94, 3, 1))
    tvecs = parameters[np.prod(cm_shape) + n_images * 3:].reshape((94, 3, 1))
    forward_projected_rays = forward_project(n_images, rvecs, tvecs, checkerboard_points)

    b_spline_object = CentralModel(
        image_dimensions=image_size,
        grid_dimensions=image_size,
        control_points=spline_grid,
        order=p['cm_order'],
        knot_method=p['cm_knot_method'],
        min_basis_value=p['cm_min_basis_value'],
        end_divergence=p['cm_end_divergence'])
        
    # Calculate backward projected rays (b_spline)
    model_points = np.ndarray((n_images, checkerboard_points.shape[1], 2))
    for i in tqdm(range(n_images), total=n_images):
        for j in range(checkerboard_points.shape[1]):
            model_points[i, j] = b_spline_object.forward_sample(forward_projected_rays[i, j, :, 0])

    return model_points.ravel() - points_2D.ravel()


def get_sparsity_matrix(n_images, n_points, b_spline_shape, points_2D, image_size):
    m = n_images * n_points * 3  # total number of residuals
    n_spline_params = np.prod(b_spline_shape)  # number of b-spline parameters
    n_rvecs_params = n_images * 3  # number of rotation vector parameters
    n_tvecs_params = n_images * 3  # number of translation vector parameters
    n = n_spline_params + n_rvecs_params + n_tvecs_params  # total number of parameters
    A = lil_matrix((m, n), dtype=int)  # sparse matrix describing the connection between parameters and residuals

    residuals_pr_image = np.arange(n_points * 3)

    # Fill out sparsity matrix for b-spline parameters.
    spline_grid = np.ndarray((b_spline_shape))
    b_spline_object = CentralModel(image_size, image_size, spline_grid, 2)

    points_2D = points_2D.reshape((-1, 2))
    for i in range(len(points_2D)):
        control_points = b_spline_object.active_control_points(points_2D[i, 0], points_2D[i, 1])
        control_points = control_points.reshape((-1, 2))

        temp = np.zeros(b_spline_shape)
        for point in control_points:
            temp[int(point[0]), int(point[1])] = np.array([1, 1, 1])

        A[i * 3:i * 3 + 3, :n_spline_params] = np.vstack([temp.ravel()] * 3)

    # Fill out sparsity matrix for rvecs
    for i in range(n_rvecs_params):
        A[n_points * 3 * int(i / 3) + residuals_pr_image, n_spline_params + i] = 1

    # Fill out sparsity matrix for tvecs
    for i in range(n_tvecs_params):
        A[n_points * 3 * int(i / 3) + residuals_pr_image, n_spline_params + n_rvecs_params + i] = 1

    return A


def old_grid_creation():
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

    k1, k2, p1, p2, k3, k4, k5, k6, *_ = 2.03288270e+00, 4.52548448e+01, -7.73226736e-03, 1.80779861e-02, -1.47886929e+02, 2.09683329e+00, 4.65342697e+01, -1.51419436e+02, -1.83082177e-02, -1.92297600e-03, 9.86103858e-03, 7.83822797e-03, 6.18311558e-03, 3.67951499e-02

    distCoeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([0, 0, 0])

    Rt = np.ndarray((3, 4))
    Rt[:, :3] = R
    Rt[:, 3] = t

    P = np.dot(K, Rt)

    dst = cv2.undistortPoints(
        src=src,
        cameraMatrix=K,
        distCoeffs=distCoeffs,
        P=P)[:, 0, :]

    Kinv = np.linalg.inv(K)

    projected_pts = np.array([np.dot(Kinv, np.array([[pt[0], ], [pt[1], ], [1, ]])) for pt in dst]).reshape((-1, 3))

    return projected_pts


def bundle_adjustment(p):
    start_time = time.time()

    rvecs, tvecs, checkerboard_points, points_2D = dict(np.load(p['ba_initialization_file'])).values()

    checkerboard_points = checkerboard_points.reshape(checkerboard_points.shape[0], checkerboard_points.shape[2], checkerboard_points.shape[3], 1)

    n_images = len(rvecs)

    # Ligger image_shape, cameraMatrix, distCoeffs i ba_initialization_file?
    # start_grid = initialization.grid_creation(image_shape, cameraMatrix, distCoeffs, border=0, step=400)

    # For at koden stadig virker har jeg insat den gamel igen
    start_grid = old_grid_creation()

    start_grid = np.divide(start_grid, np.linalg.norm(start_grid, axis=1).reshape((start_grid.shape[0], 1)))

    if p['cm_fit_control_points']:
        cm, _ = fit_central_model(start_grid.reshape(p['cm_shape']),
                                  image_dimensions=p['image_size'],
                                  grid_dimensions=p['image_size'],
                                  order=p['cm_order'],
                                  knot_method=p['cm_knot_method'],
                                  min_basis_value=p['cm_min_basis_value'],
                                  end_divergence=p['cm_end_divergence'],
                                  verbose=p['ls_verbose'])
        start_grid = cm.a.reshape((-1, 3))

    ba_params = np.hstack((start_grid.ravel(), rvecs.ravel(), tvecs.ravel()))

    residuals_init = residuals_3d(parameters=ba_params,
                                    p=p,
                                    n_images=n_images,
                                    points_2D=points_2D,
                                    image_size=p['image_size'],
                                    cm_shape=p['cm_shape'],
                                    checkerboard_points=checkerboard_points)

    A = None
    if p['ls_sparsity']:
        A = get_sparsity_matrix(n_images, points_2D.shape[1], p['cm_shape'], points_2D, p['image_size'])
        # plt.spy(A, aspect='auto')
        # plt.show()

    res = least_squares(
        fun=residuals_3d,
        x0=ba_params,
        jac_sparsity=A,
        verbose=p['ls_verbose'],
        x_scale='jac',
        ftol=p['ls_ftol'],
        gtol=p['ls_gtol'],
        method=p['ls_method'],
        args=(p,
              n_images,
              points_2D,
              p['image_size'],
              p['cm_shape'],
              checkerboard_points))

    duration = time.time() - start_time

    spline_grid = res.x[:np.prod(p['cm_shape'])].reshape(p['cm_shape'])
    rvecs = res.x[np.prod(p['cm_shape']):][:n_images * 3].reshape((94, 3, 1))
    tvecs = res.x[np.prod(p['cm_shape']) + n_images * 3:].reshape((94, 3, 1))
    res['initial_residuals'] = residuals_init
    res['start_grid'] = start_grid
    res['spline_grid'] = spline_grid
    res['rvecs'] = rvecs
    res['tvecs'] = tvecs

    return res, duration


if __name__ == '__main__':
    parameters = {
        'image_size': (1600, 1200),
        'checkerboard_shape': (12, 12),
        'cm_dimensions': (1600, 1200),
        'cm_shape': (5, 4, 3),
        'cm_order': 3,
        'cm_fit_control_points': True,
        'cm_knot_method': 'open_uniform',
        'cm_min_basis_value': 0.001,
        'cm_end_divergence': 1e-10,
        'cm_threads': 1,
        'ls_sparsity': True,
        'ls_verbose': 2,
        'ls_ftol': 1e-8,
        'ls_gtol': 1e-8,
        'ls_method': 'trf',
        'ba_initialization_file': 'initial_calibration_results.npz',
        'seed': None,
        'os_info': gethostname(),
        'not_default': []
    }

    for arg in argv[1:]:
        key, value = arg.split('=')
        if key in parameters:
            try:
                if parameters[key] != eval(value):
                    parameters['not_default'].append(key)

                parameters[key] = eval(value)
                print("key '{}' set to '{}' of type '{}'".format(key, parameters[key], type(parameters[key])))
            except:
                parameters[key] = value
                print("key '{}' set to '{}' of type '{}'".format(key, parameters[key], type(parameters[key])))
        else:
            print("key '{}' is not in parameters".format(key))

    if parameters['seed'] != None:
        np.random.seed(parameters['seed'])

    res, duration = bundle_adjustment(parameters)

    for key in res:
        if isinstance(res[key], csr_matrix):
            res[key] = res[key].toarray()
        if isinstance(res[key], np.ndarray):
            res[key] = res[key].tolist()

    filename = 'tests/{}_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '{}'
    with open(filename.format('par', '.json'), 'w') as json_out:
        json.dump({'parameters': parameters, 'duration': duration}, json_out, ensure_ascii=False, indent=4)

    with open(filename.format('res', '.pickle'), 'wb') as pickle_out:
        pickle.dump(res, pickle_out)