import numpy as np

p = {
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
    'ls_ftol': 1,
    'ls_gtol': 1e-8,
    'ls_method': 'trf',
    'ba_initialization_file': 'initial_calibration_results.npz',
    'seed': None,
    'not_default': []
}


rvecs, tvecs, checkerboard_points, points_2D = dict(np.load(p['ba_initialization_file'])).values()

count = 0
for checkerboard in checkerboard_points:
    if checkerboard.all() == checkerboard_points[0].all():
        count += 1
print(count)