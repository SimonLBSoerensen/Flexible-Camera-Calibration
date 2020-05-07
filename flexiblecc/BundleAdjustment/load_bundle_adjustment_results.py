import numpy as np
import pickle
from bundle_adjustment import BundleAdjustment
from matplotlib import pyplot as plt

def rms(residuals):
    return np.sqrt(residuals.dot(residuals)/residuals.size)

res, p = pickle.load(open('tests/res_2020-05-07_13-44-31.pickle', 'rb'))

error, cameraMatrix, distCoeffs, _, _, _, _, _, \
        all_corners_2D, _, _, _, obj_points = np.load(p['ba_initialization_file'], allow_pickle=True)


n_images = obj_points.shape[0]

res.x = np.array(res.x)

cm_control_points = res.x[:np.prod(p['cm_shape'])].reshape(p['cm_shape'])
rvecs = res.x[np.prod(p['cm_shape']):][:n_images * 3].reshape((n_images, 3, 1))
tvecs = res.x[np.prod(p['cm_shape']) + n_images * 3:].reshape((n_images, 3, 1))

ba = BundleAdjustment(p, obj_points, rvecs, tvecs, all_corners_2D, cameraMatrix, distCoeffs, cm_control_points, use_control_points=True)

ls_params = np.hstack((cm_control_points.ravel(), rvecs.ravel(), tvecs.ravel()))

#residuals_3D = ba.calc_residuals_3D(ls_params, p)

residuals_2D = ba.calc_residuals_2D(ls_params, p)

print('rms: ', rms(residuals_2D))