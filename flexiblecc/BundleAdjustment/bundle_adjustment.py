import numpy as np
import cv2
import json
import pickle
import datetime
from tqdm import tqdm
from copy import deepcopy
from ..BSpline.central_model import CentralModel, fit_central_model
#from flexiblecc.BSpline.central_model import CentralModel, fit_central_model
from . import initialization
#from flexiblecc.BundleAdjustment import initialization
from socket import gethostname
from scipy.sparse import lil_matrix, csr_matrix
from scipy import optimize
from time import time
from math import ceil
from os import makedirs

class BundleAdjustment:
    """
    BundleAdjustment class. This should be used after using parametric camera calibration for an initial guess.
    """
    def __init__(self, obj_points, rvecs, tvecs, all_corners_2D, cameraMatrix, distCoeffs, control_points=None, args=[]):
        """
        Initializes variables and performs an early fitting of b-spline control points, unless p contains 'cm_fit_control_points=False'.

        :param obj_points: Vector of vectors of object points found with calibrate_camera_charuco or calibrate_camera_chessboard from calibrate.py
        :type obj_points: ndarray
        :param rvecs: Rotation vectors estimated for each pattern view found with calibrate_camera_charuco or calibrate_camera_chessboard from calibrate.py
        :type rvecs: ndarray
        :param tvecs: Translation vectors estimated for each pattern view found with calibrate_camera_charuco or calibrate_camera_chessboard from calibrate.py
        :type tvecs: ndarray
        :param all_corners_2D: Vector of vectors of the charuco corners found with calibrate_camera_charuco or calibrate_camera_chessboard from calibrate.py
        :type all_corners_2D: ndarray
        :param cameraMatrix: Intrinsic camera matrix
        :type cameraMatrix: ndarray
        :param distCoeffs: Vector of distortion coefficients of 4, 5, 8, 12 or 14 elements
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
        :type distCoeffs: ndarray
        :param control_points: Optional control points for the CentralModel class.
        :type control_points: ndarray
        :param args: List of strings to set parameters, example: ['cm_stepsize=252'], default parameters are:
            'image_dimensions': (4032, 3024),
            'cm_dimensions': (4032, 3024),
            'cm_stepsize': 252,
            'cm_border': 0,
            'cm_shape': (),
            'cm_order': 2,
            'cm_fit_control_points': True,
            'cm_knot_method': 'open_uniform',
            'cm_min_basis_value': 1e-4,
            'cm_end_divergence': 1e-10,
            'cm_threads': 1,
            'ls_sparsity': True,
            'ls_verbose': 2,
            'ls_ftol': 1e-8,
            'ls_gtol': 1e-8,
            'ls_method': 'trf',
            'os_info': gethostname(),
            'not_default': []
        :type args: list
        """

        self.p = {
            'image_dimensions': (4032, 3024),
            'cm_dimensions': (4032, 3024),
            'cm_stepsize': 252,
            'cm_border': 0,
            'cm_shape': (),
            'cm_order': 2,
            'cm_fit_control_points': True,
            'cm_knot_method': 'open_uniform',
            'cm_min_basis_value': 1e-4,
            'cm_end_divergence': 1e-10,
            'cm_threads': 1,
            'ls_sparsity': True,
            'ls_verbose': 2,
            'ls_ftol': 1e-8,
            'ls_gtol': 1e-8,
            'ls_method': 'trf',
            'os_info': gethostname(),
            'not_default': []
        }

        self.__edit_parameters(args)

        self.p['cm_shape'] = (int(ceil((self.p['cm_dimensions'][0]+self.p['cm_border']*2)/self.p['cm_stepsize'])+1), \
            int(ceil((self.p['cm_dimensions'][1]+self.p['cm_border']*2)/self.p['cm_stepsize'])+1), 3)

        self.obj_points = obj_points

        self.sampled_spline_rays = deepcopy(obj_points)

        #Reshape board for later use in np.dot()
        for i, img in enumerate(self.obj_points):
            self.obj_points[i] = img.reshape((img.shape[0],) + (3,1))  

        self.transformed_corners_3D = deepcopy(obj_points)

        self.all_corners_2D = all_corners_2D

        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)

        if control_points != None:
            self.cm_init_ctrl_ptns = None
            self.cm_control_points = control_points.reshape((-1,3))
        else:
            self.cm_init_ctrl_ptns = initialization.grid_creation(self.p['image_dimensions'], cameraMatrix, distCoeffs, border=self.p['cm_border'], step=self.p['cm_stepsize'])
            self.cm_init_ctrl_ptns = np.divide(self.cm_init_ctrl_ptns, np.linalg.norm(self.cm_init_ctrl_ptns, axis=1).reshape((self.cm_init_ctrl_ptns.shape[0], 1)))

            self.cm_control_points = None
            if self.p['cm_fit_control_points']:
                cm, _ = fit_central_model(self.cm_init_ctrl_ptns.reshape(self.p['cm_shape']),
                                        image_dimensions=self.p['image_dimensions'],
                                        grid_dimensions=self.p['cm_dimensions'],
                                        order=self.p['cm_order'],
                                        knot_method=self.p['cm_knot_method'],
                                        min_basis_value=self.p['cm_min_basis_value'],
                                        end_divergence=self.p['cm_end_divergence'],
                                        verbose=self.p['ls_verbose'])
                self.cm_init_ctrl_ptns = cm.a.reshape((-1, 3))


    def transform_board_to_cam(self):
        """
        Returns the board points in the camera frame

        :return transformed_corners_3D: Vector of vectors of board points in the camera's frame
        :rtype transformed_corners_3D: ndarray
        """

        for i, r, t in zip(range(len(self.obj_points)), self.rvecs, self.tvecs):

            rotation_matrix = cv2.Rodrigues(r)[0]
            for j, corner in enumerate(self.obj_points[i]):

                #Might be able to use tensordot instead
                self.transformed_corners_3D[i][j] = np.dot(rotation_matrix, corner) + t

                #Normalize each corner point
                self.transformed_corners_3D[i][j] = np.divide(self.transformed_corners_3D[i][j], np.linalg.norm(self.transformed_corners_3D[i][j]))

        return self.transformed_corners_3D


    def calc_residuals_3D(self, ls_params, return_points_3D=False):
        """
        Returns 3D residuals, this is used for least_squares optimization.

        :param ls_params: All parameters changed by least squares, flattened array of control points, rvecs and tvecs.
        :type ls_params: ndarray
        :param return_points_3D: Set to true to return residuals, sampled_spline_rays and transformed_corners_3D
        :type return_points_3D: Boolean

        :return residuals: Vector of all residuals
        :rtype residuals: ndarray
        :return sampled_spline_rays: Optional: Vector of vectors of sampled points using the CentralModel class.
        :rtype sampled_spline_rays: ndarray
        :return transformed_corners_3D: Optional: Vector of vectors of board points in the camera's frame
        :rtype transformed_corners_3D: ndarray
        """

        n_images = self.all_corners_2D.shape[0]
        cm_shape = self.p['cm_shape']

        cm_control_points = ls_params[:np.prod(cm_shape)].reshape(cm_shape)
        self.rvecs = ls_params[np.prod(cm_shape):][:n_images * 3].reshape((n_images, 3, 1))
        self.tvecs = ls_params[np.prod(cm_shape) + n_images * 3:].reshape((n_images, 3, 1))

        b_spline_object = CentralModel(
            image_dimensions=self.p['image_dimensions'],
            grid_dimensions=self.p['cm_dimensions'],
            control_points=cm_control_points,
            order=self.p['cm_order'],
            knot_method=self.p['cm_knot_method'],
            min_basis_value=self.p['cm_min_basis_value'],
            end_divergence=self.p['cm_end_divergence'])

        # Transform chessboard corners to camera frame
        transformed_corners_3D = np.concatenate(self.transform_board_to_cam())

        # Calculate backward projected rays (b_spline)
        for i, image_corners_2D in enumerate(self.all_corners_2D):
            for j, corner_2D in enumerate(image_corners_2D):
                self.sampled_spline_rays[i][j] = b_spline_object.sample(corner_2D[0, 0], corner_2D[0, 1])

        sampled_spline_rays = np.concatenate(self.sampled_spline_rays)
        if return_points_3D:
            transformed_corners_3D.ravel() - sampled_spline_rays.ravel(), sampled_spline_rays, transformed_corners_3D
        else:
            return transformed_corners_3D.ravel() - sampled_spline_rays.ravel()


    def calc_residuals_2D(self, ls_params, return_points_2D=False):
        """
        Returns 2D residuals, used to compare with parametric camera calibration.

        :param ls_params: All parameters changed by least squares, flattened array of control points, rvecs and tvecs.
        :type ls_params: ndarray
        :param return_points_2D: Set to true to return residuals, estimated_corners_2D and corners_2D
        :type return_points_2D: Boolean

        :return residuals: Vector of all residuals
        :rtype residuals: ndarray
        :return estimated_corners_2D: Optional: Vector of vectors of estimated corners in 2D.
        :rtype estimated_corners_2D: ndarray
        :return corners_2D: Optional: Vector of vectors of correct corners in 2D.
        :rtype corners_2D: ndarray
        """

        cm_shape = self.p['cm_shape']
        image_size = self.p['image_dimensions']
        n_images = self.obj_points.shape[0]

        control_points = ls_params[:np.prod(cm_shape)].reshape(cm_shape)
        rvecs = ls_params[np.prod(cm_shape):][:n_images * 3].reshape((n_images, 3, 1))
        tvecs = ls_params[np.prod(cm_shape) + n_images * 3:].reshape((n_images, 3, 1))

        self.transform_board_to_cam()

        b_spline_object = CentralModel(
            image_dimensions=self.p['image_dimensions'],
            grid_dimensions=self.p['cm_dimensions'],
            control_points=control_points,
            order=self.p['cm_order'],
            knot_method=self.p['cm_knot_method'],
            min_basis_value=self.p['cm_min_basis_value'],
            end_divergence=self.p['cm_end_divergence'])
            
        # Calculate backward projected rays (b_spline)
        model_points = deepcopy(self.all_corners_2D)
        for i in tqdm(range(n_images), total=n_images):
            for j in range(self.obj_points[i].shape[0]):
                model_points[i][j] = b_spline_object.forward_sample(self.transformed_corners_3D[i][j,:,0])

        residuals_manhattan = np.concatenate(model_points).ravel() - np.concatenate(self.all_corners_2D).ravel()
        residuals_euclidean = np.linalg.norm(residuals_manhattan.reshape((-1,2)), axis=1)
        if return_points_2D:
            return residuals_euclidean, model_points, self.all_corners_2D
        else:
            return residuals_euclidean

    
    def get_sparsity_matrix(self, cm_shape, image_dimensions):
        """
        Returns sparsity matrix, used to speed up least squares optimization.

        :param cm_shape: Shape of control points for CentralModel
        :type ls_params: tuple
        :param image_dimensions: Dimensions of images used
        :type return_points_2D: tuple

        :return A: A sparse list of lists matrix from scipy.sparse.lil_matrix
        :rtype A: List of Lists
        """
        m = 0 # total number of residuals
        for j in range(obj_points.shape[0]):
            m += np.prod(self.obj_points[j].shape)
        
        n_spline_params = np.prod(cm_shape)  # number of b-spline parameters
        n_rvecs_params = obj_points.shape[0] * 3  # number of rotation vector parameters
        n_tvecs_params = obj_points.shape[0] * 3  # number of translation vector parameters
        n = n_spline_params + n_rvecs_params + n_tvecs_params  # total number of parameters
        A = lil_matrix((m, n), dtype=int)  # sparse matrix describing the connection between parameters and residuals

        # Fill out sparsity matrix for the b-spline (central model) parameters.
        spline_grid = np.ndarray((cm_shape))
        b_spline_object = CentralModel(image_dimensions, image_dimensions, spline_grid, 2)

        all_corners_2D = np.concatenate(self.all_corners_2D).reshape((-1, 2))
        for i in range(len(all_corners_2D)):
            control_points = b_spline_object.active_control_points(all_corners_2D[i, 0], all_corners_2D[i, 1])
            control_points = control_points.reshape((-1, 2))

            temp = np.zeros(cm_shape)
            for point in control_points:
                temp[int(point[0]), int(point[1])] = np.array([1, 1, 1])

            A[i * 3:i * 3 + 3, :n_spline_params] = np.vstack([temp.ravel()] * 3)

        #Fill out sparsity matrix for rvecs and tvecs parameters
        for i in range(n_rvecs_params):
            img_index = int(i / 3)

            prev_residuals = 0
            for j in range(img_index):
                prev_residuals += np.prod(self.obj_points[j].shape)

            residuals_this_img = np.arange(np.prod(self.obj_points[img_index].shape))

            # Fill out sparsity matrix for rvecs
            A[prev_residuals + residuals_this_img, n_spline_params + i] = 1

            # Fill out sparsity matrix for tvecs
            A[prev_residuals + residuals_this_img, n_spline_params + n_rvecs_params + i] = 1

        return A


    def least_squares(self, print_to_files=False): 
        """
        Performs least squares optimization and returns a scipy.optimize.OptimizeResult object.

        :return res: scipy.optimize.OptimizeResult object
        :rtype res: scipy.optimize.OptimizeResult

        """
        if self.cm_control_points == None:
            self.cm_control_points = self.cm_init_ctrl_ptns

        ls_params = np.hstack((self.cm_control_points.ravel(), self.rvecs.ravel(), self.tvecs.ravel()))

        print('Calculating initial residuals')
        residuals_init = self.calc_residuals_3D(ls_params, p)

        A = None #Sparsity matrix, describes the relationship between parameters (cm_control_points, rvecs, tvecs) and residuals.
        if self.p['ls_sparsity']:
            print('Generating sparsity matrix')
            A = self.get_sparsity_matrix(cm_shape=self.p['cm_shape'], image_dimensions=self.p['image_dimensions'])
        
        print('Performing least squares optimization on the control points and position of the chessboards')
        res = optimize.least_squares(
            fun=self.calc_residuals_3D,
            x0=ls_params,
            jac_sparsity=A,
            verbose=self.p['ls_verbose'],
            x_scale='jac',
            ftol=self.p['ls_ftol'],
            gtol=self.p['ls_gtol'],
            method=self.p['ls_method'],
            args=(False))

        n_images = len(self.rvecs)

        self.cm_control_points = res.x[:np.prod(self.p['cm_shape'])].reshape(self.p['cm_shape'])
        self.rvecs = res.x[np.prod(self.p['cm_shape']):][:n_images * 3].reshape((n_images, 3, 1))
        self.tvecs = res.x[np.prod(self.p['cm_shape']) + n_images * 3:].reshape((n_images, 3, 1))
        res['initial_residuals'] = residuals_init
        res['initial_control_points'] = self.cm_init_ctrl_ptns
        res['cm_control_points'] = self.cm_control_points
        res['rvecs'] = self.rvecs
        res['tvecs'] = self.tvecs

        self.p = p
        self.res = res

        if print_to_files:
            self._print_to_files(res)

        return res


    def rms(self, residuals):
        """
        Calculates and returns root mean squared value.

        :param residuals: Vector of residuals
        :type residuals: ndarray

        :return rms: root mean squared value of residuals
        :rtype rms: float

        """
        return np.sqrt(residuals.dot(residuals)/residuals.size)


    def __edit_parameters(self, args):
        """
        Changes parameters according to elements in args

        :param args: List of strings of changes to parameters
        :type args: list
        """
        if len(args) == 0:
            print('Running with default parameters')
        for arg in args:
            try:
                key, value = arg.split('=')
            except:
                print("Argument '{}' is invalid and therefore ignored".format(arg))
                continue
            if key in p:
                try:
                    if self.p[key] != eval(value):
                        self.p['not_default'].append(key)

                    self.p[key] = eval(value)
                    print("key '{}' set to '{}' of type '{}'".format(key, self.p[key], type(self.p[key])))
                except:
                    self.p[key] = value
                    print("key '{}' set to '{}' of type '{}'".format(key, self.p[key], type(self.p[key])))
            else:
                print("key '{}' is not in parameters".format(key))


    def _print_to_files(self, res, folder='tests/'):
        """
        Prints results and parameters to files in the folder specified in the arguments.

        :param res: scipy.optimize.OptimizeResult object from least_squares
        :type res: scipy.optimize.OptimizeResult
        :param folder: Path to dump results and paramters to
        :type folder: str
        """
        makedirs(folder, exist_ok=True)

        for key in res:
            if isinstance(res[key], csr_matrix):
                res[key] = res[key].toarray()
            if isinstance(res[key], np.ndarray):
                res[key] = res[key].tolist()

        filename = folder + '{}_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '{}'
        
        print('Saving json file with filename: ', filename.format('par', '.json'))
        with open(filename.format('par', '.json'), 'w') as json_out:
            json.dump({'parameters': self.p}, json_out, ensure_ascii=False, indent=4)
        
        print('Saving pickle file with filename: ', filename.format('res', '.pickle'))
        with open(filename.format('res', '.pickle'), 'wb') as pickle_out:
            pickle.dump([res, self.p], pickle_out)
    

if __name__ == '__main__':

    error, cameraMatrix, distCoeffs, rvecs, tvecs, _, _, _, \
        all_corners_2D, _, _, _, obj_points = np.load('cali.npy', allow_pickle=True)
    
    start_time = time()

    ba = BundleAdjustment(obj_points, rvecs, tvecs, all_corners_2D, cameraMatrix, distCoeffs)

    res = ba.least_squares()

    duration = time() - start_time
    p['duration'] = duration
    p['opencv_calib_error'] = error

    pass