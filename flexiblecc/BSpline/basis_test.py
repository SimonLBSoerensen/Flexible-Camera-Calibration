from central_model import CentralModel
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import numpy as np

image_dimensions = (10000, 100)
grid_shape =  (10, 10)
control_points = np.zeros(grid_shape + (3,))
knot_method = 'open_uniform'
min_basis_value = 0
end_divergence = 0.01

cmap = get_cmap('rainbow')

for order in range(1, 4):
    figure_name = '{}_{}'.format(knot_method, order)
    plt_title = '{} - Order: {}'.format(knot_method.capitalize(), order)

    plt.figure(figure_name)

    plt.title(plt_title)
    plt.xlabel('x')
    plt.ylabel('Basis function value')

    cm = CentralModel(
        image_dimensions=image_dimensions, 
        grid_dimensions=image_dimensions,
        control_points=control_points,
        order=order,
        knot_method=knot_method,
        min_basis_value=min_basis_value,
        end_divergence=end_divergence
    )

    pts_i = []

    for i in range(grid_shape[0]):
        pts = np.ndarray((image_dimensions[0], 2))
        for u in range(image_dimensions[0]):
            pts[u][0] = u/image_dimensions[0]
            pts[u][1] = cm.__B__(i, order, cm.th, u/image_dimensions[0])
        
        pts_i.append(pts[:,1])

        c = cmap(i / (grid_shape[0] - 1))
        plt.plot(pts[:,0], pts[:,1], '-', c=c)
    
    pts = np.sum(pts_i, axis=0)
    ones = np.where(pts == 1)
    edges = (np.min(ones) / image_dimensions[0], np.max(ones) / image_dimensions[0])

    t1 = plt.plot(np.array(range(image_dimensions[0]))/image_dimensions[0], pts, '--', c='black')[0]
    plt.ylim((0, 1.01))
    
    t2 = plt.vlines(edges, 0, 1.01, colors='r')
    plt.legend((t1, t2), ('Sum of basis functions', 'Spline edges'))

plt.show()
