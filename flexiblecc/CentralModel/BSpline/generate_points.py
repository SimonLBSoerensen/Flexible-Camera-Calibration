import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm

from central_model import CentralModel, fit_central_model

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import glReadBuffer, GL_FRONT

## GENERATION ##
fit = True

image_dimensions = (200, 200)
grid_dimensions = (200, 200)

order = 2
shape = (6,6,3)

knot_method = 'open_uniform' # 'uniform' or 'open_uniform'
 
end_divergence = 1e-10
min_basis_value = 0

## PLOTTING ##
export_option = 'single' # 'rotation', 'single' or 'none'. 
image_name = '{}_{}_{}'.format(order, fit, knot_method)

draw_as_surface = True

draw_pts = True
draw_ctrl = True
draw_tv = True

orbit = False

frame_time = 1000/60 # in ms

np.random.seed(2)
target_values = np.random.normal(0, np.sqrt(np.average(image_dimensions)), np.prod(shape))
target_values = np.reshape(target_values, shape)

if fit:
    cm, results = fit_central_model(
        target_values,
        image_dimensions=image_dimensions, 
        grid_dimensions=grid_dimensions, 
        order=order,
        knot_method=knot_method,
        end_divergence=end_divergence,
        min_basis_value=min_basis_value,
        verbose=2
    )
else:
    cm = CentralModel(
        image_dimensions,
        grid_dimensions,
        target_values,
        order=order,
        knot_method=knot_method,
        end_divergence=end_divergence,
        min_basis_value=min_basis_value
    )

d = np.divide(np.subtract(grid_dimensions, 1), np.subtract(shape[:-1], 1))
c = np.divide(np.subtract(grid_dimensions, image_dimensions), 2)

ctrl = np.round(np.array([[[i * d[0], j * d[1], cm.a[i,j,0]] for j in range(shape[1])] for i in range(shape[0])]))
ctrl = np.reshape(ctrl, (-1, 3))
ctrl[:,:2] -= c

tv = np.round(np.array([[[i * d[0], j * d[1], target_values[i,j,0]] for j in range(shape[1])] for i in range(shape[0])]))
tv = np.reshape(tv, (-1, 3))
tv[:,:2] -= c

pts = np.ndarray(grid_dimensions + (3,))

for u in tqdm(range(image_dimensions[0])):
    for v in range(image_dimensions[1]):
        pts[u, v, :] = cm.sample(u, v)

y = pts[:,:,0]

x = np.array([[[i, j, y[i,j]] for j in range(image_dimensions[1])] for i in range(image_dimensions[0])])
x = np.reshape(x, (-1,3))

# Make pyqtgraph window.
app = pg.mkQApp()
w = gl.GLViewWidget()
w.show()

## Adds point to scatter plot.
if draw_pts:
    ptcolor = np.ndarray((x.shape[0], 4))
    c = np.divide(np.subtract(x[:,2], np.min(x[:,2])), np.max(x[:,2]) - np.min(x[:,2]))
    ptcolor[:,0] = c
    ptcolor[:,1] = 0
    ptcolor[:,2] = np.subtract(1, c)
    ptcolor[:,3] = 1

    # Creates surface/pointcloud for the b-spline samples.
    if draw_as_surface:
        x1 = np.arange(np.max(x[:,0]) + 1)
        x2 = np.arange(np.max(x[:,1]) + 1)
        pts = gl.GLSurfacePlotItem(x1, x2, x[:,2].reshape(len(x1), len(x2)), colors=ptcolor)

    else:
        pts = gl.GLScatterPlotItem(pos=x, color=ptcolor, size=1)

    w.addItem(pts)

scatterPlotItems = {}

# Creates points for the control points of the b-spline.
if draw_ctrl:
    ctrlcolor = np.full((ctrl.shape[0], 4), np.array([0, 1, 0, 1]))
    scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl, color=ctrlcolor)
    w.addItem(scatterPlotItems['ctrl'])

# Creates points for the target points of the b-spline.
if draw_tv:
    tvcolor = np.full((ctrl.shape[0], 4), np.array([1, 1, 1, 1]))
    scatterPlotItems['tv'] = gl.GLScatterPlotItem(pos=tv, color=tvcolor)
    w.addItem(scatterPlotItems['tv'])

# Orients camera in 3d.
w.showFullScreen()
centerpoint = np.average(x, axis=0)
w.pan(centerpoint[0], centerpoint[1], centerpoint[2])
w.setCameraPosition(distance=400)


# Orbiting and image exporting.
da = 0.2
current_angle = 0
i = 0
def update():
    global current_angle, da, i, export_option

    if export_option == 'single':
        w.grabFrameBuffer().save('images/{}.png'.format(image_name))
        export_option = 'none'

    if current_angle < 720 and export_option == 'rotation':
        w.grabFrameBuffer().save('images/{}_{}.png'.format(image_name, i))
        current_angle += da

    if orbit:
        w.orbit(da, 0)
        
    i += 1

timera = pg.QtCore.QTimer()
timera.timeout.connect(update)

if export_option == 'rotation':
    timera.start(1)
else:
    timera.start(frame_time)

glReadBuffer(GL_FRONT) # A little bit of c-code in my lines

# Executes application
app.exec_()