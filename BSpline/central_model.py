import numpy as np

def CubicBezier(points, t):
    # TODO: Optimize. This should be relatively simple with numpy.
    k0 = (1-t)**3
    k1 = ((1-t)**2) * t * 3
    k2 = (1-t) * (t**2) * 3
    k3 = t**3

    return points[0] * k0 + points[1] * k1 + points[2] * k2 + points[3] * k3

def BezierTest():
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    t = np.linspace(0, 1, num=15)

    ctrl0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -0.5], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0]])
    ctrl1 = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0], [3.0, 1.0, -2.0]])
    ctrl2 = np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 1.0], [2.0, 2.0, -1.0], [3.0, 2.0, 1.0]])
    ctrl3 = np.array([[0.0, 3.0, 0.0], [1.0, 3.0, -0.2], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    ctrl = np.array([ctrl0, ctrl1, ctrl2, ctrl3])

    pts = []

    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for ctrln, c in zip(ctrl, ['ro', 'go', 'bo', 'yo']):
        ax.plot3D(ctrln[:,0], ctrln[:,1], ctrln[:,2], c)
    for tm in t:
        for tn in t:
            ctrlm = []
            for ctrln in ctrl:
                ctrlm.append(CubicBezier(ctrln, tn))
            pts.append(CubicBezier(ctrlm, tm))
    pts = np.array(pts)

    ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2])
    plt.show()

class CentralModel():
    def __init__(self, gridshape, imshape):
        assert isinstance(gridshape, tuple) and len(gridshape) == 2, 'gridshape must be a tuple with 2 positive integers.'
        assert isinstance(imshape, tuple) and len(imshape) == 2, 'imshape must be a tuple with 2 positive integers.'

        self.gridshape = gridshape
        self.imshape = imshape
        self.grid = np.full(gridshape + (3,), 0)
        self.dp = np.divide(imshape, np.subtract(gridshape, 3))
    
    def gridpositions(self):
        maxval = np.multiply(np.subtract(self.gridshape, 2), self.dp)

        x = np.linspace(-self.dp[0], maxval[0], self.gridshape[0])
        y = np.linspace(-self.dp[1], maxval[1], self.gridshape[1])

        result = np.transpose(np.meshgrid(x, y))
 
        return result

    def sample(self, point):
        neighbor = np.floor(np.divide(point, self.dp)) + 1 # Remember that the corner of the image starts at grid point (1,1), not (0,0).

        m = np.transpose(np.array([np.linspace(-1, 2, 4)]*2))

        neighbors = np.transpose(m + neighbor)
        neighbors = np.transpose(np.meshgrid(neighbors[0], neighbors[1]))

        neighbors = np.reshape(neighbors, (-1, 2,))

        neighborvals = self.grid[neighbors]

        [tx, ty] = 1/3 * (1 + np.divide(point - np.multiply(neighbor - 1, self.dp), self.dp))

        ctrl = []
        for row in neighbors:
            ctrl.append(CubicBezier(row, ty))
        return CubicBezier(ctrl, tx)

BezierTest()

cm = CentralModel((11, 8), (800,500))
cm.gridpositions()
sample = cm.sample(np.array([270, 99]))