import numpy as np

# The implementation of this B-spline-based camera model is based on the description in the 
# article "Generalized B-spline Camera Model" by Johannes Beck and Christoph Stiller.

# Resources:
# Wikipedia article on b-splines: https://en.wikipedia.org/wiki/B-spline
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html

class CentralModel:
    def __init__(self, image_dimensions, grid_dimensions, grid, order, min_basis_value=0.001, end_divergence = 1e-10):
        """Initializes a CentralModel object. \n

        Keyword arguments: \n
        image_dimensions: tuple containing (width, height) of image in pixels. \n
        grid_dimensions: tuple containing (width, height) of grid in pixels. \n
        grid: numpy 3d array containing unit vectors describing camera intrinsics.\n
        order: the order of the interpolation.\n
        min_basis_value: the minimum basis value 
        """
        assert isinstance(image_dimensions, tuple) and len(image_dimensions) == 2, 'image_dimensions must be a 2-dimensional tuple.'
        assert isinstance(grid_dimensions, tuple) and len(grid_dimensions) == 2, 'grid_dimensions must be a 2-dimensional tuple.'
        assert isinstance(grid, np.ndarray) and len(grid.shape) == 3 and grid.shape[-1] == 3, 'grid must be a 3-dimensional numpy array with a depth of 3.'
        assert grid.shape[0] > order and grid.shape[1] > order, 'order must be smaller than grid size.'

        self.image_width = image_dimensions[0]
        self.image_height = image_dimensions[1]

        self.grid_width = grid_dimensions[0]
        self.grid_height = grid_dimensions[1]

        self.n = grid.shape[0]
        self.m = grid.shape[1]

        self.a = grid

        self.order = order

        self.min_basis_value = min_basis_value

        d = np.divide(grid_dimensions, np.subtract((self.n,  self.m), self.order))

        c = np.divide(np.subtract(grid_dimensions, image_dimensions), 2)

        # The following figure shows a B-spline surface defined by 6 rows and 6 columns of control points.
        # The knot vector and the degree in the u-direction are U = { 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1 } and 2. 
        # The knot vector and the degree in the v-direction are V = { 0, 0, 0, 0, 0.33, 0.66, 1, 1, 1, 1 } and 3.
        
        if end_divergence != 0:
            start_offset = -np.arange(end_divergence * self.order, 0, -end_divergence)
            end_offset = np.arange(end_divergence, end_divergence * (self.order + 1), end_divergence)
        else:
            start_offset = end_offset = 0

        middle = np.arange(0, self.n - self.order + 1) * d[0]
        start = np.full((self.order,), middle.min()) + start_offset
        end = np.full((self.order,), middle.max()) + end_offset
        self.t_vert = np.concatenate((start, middle, end)) - c[0]

        middle = np.arange(0, self.m - self.order + 1) * d[1]
        start = np.full((self.order,), middle.min()) + start_offset
        end = np.full((self.order,), middle.max()) + end_offset
        self.t_hori = np.concatenate((start, middle, end)) - c[1]

    def __B__(self, i, k, t, x):
        """Used to calculate the basis function \n

        Keyword arguments: \n
        i: Index for which to sample grid. \n
        k: Basis function order. \n
        t: Knot vector. \n
        x: Pixel coordinate of sample.   
        """
        #if x == t[-1]:
            #print('a')

        # Equation 2
        if k == 0:
            if t[i] <= x < t[i + 1]:
                return 1
            else:
                return 0
        
        # Equation 3 ish
        term1a = x - t[i]
        term1b = t[i + k] - t[i]
        term1c = self.__B__(i, k - 1, t, x)

        # If term1b is zero, the division will be undefined.
        # 'Solution' suggested in https://en.wikipedia.org/wiki/Talk%3AB-spline#Avoiding_division_by_zero
        if term1b == 0:
            term1a = term1b = 1

        term2a = t[i + k + 1] - x
        term2b = t[i + k + 1] - t[i + 1]
        term2c = self.__B__(i + 1, k - 1, t, x)

        # If term2b is zero, the division will be undefined.
        if term2b == 0:
            term2a = term2b = 1
        
        return term1a/term1b * term1c + term2a/term2b * term2c

    def sample(self, u, v):
        """Used to sample the b-spline surface. \n

        Keyword arguments: \n
        u: Horizontal pixel coordinate of sample. \n
        v: Vertical pixel coordinate of sample. 
        """
        Bh = np.array([self.__B__(i, self.order, self.t_vert, u) for i in range(self.n)])
        Bv = np.array([self.__B__(j, self.order, self.t_hori, v) for j in range(self.m)])

        # Optimization
        vi = np.where(Bh >= self.min_basis_value)[0]
        vj = np.where(Bv >= self.min_basis_value)[0]

        res = np.full((3,), 0.0)
        for i in vi:
            for j in vj:
                aij = self.a[i, j]
                res += np.multiply(aij, Bh[i] * Bv[j])

        return res
    
    def sample_normalized(self, u, v):
        """Used to sample the b-spline surface. Returns a normalized direction. \n

        Keyword arguments: \n
        u: Horizontal pixel coordinate of sample. \n
        v: Vertical pixel coordinate of sample. 
        """
        s = self.sample(u, v)
        return s / np.linalg.norm(s)

    # TODO: Sample with vector of points.

    def sample_grid(self):
        xs = np.floor((self.grid_width - 1) / (self.n - 1) * np.arange(0, self.n))
        ys = np.floor((self.grid_height - 1) / (self.m - 1) * np.arange(0, self.m))

        pts = np.transpose(np.meshgrid(xs, ys))

        samples = np.ndarray((self.n, self.m, 3))
        for i in range(0, self.n):
            for j in range(0, self.m):
                samples[i,j] = self.sample(pts[i,j,0], pts[i,j,1])

        return samples

def grid_creation(grid_size, image_size ):
    A = np.array([(1,0,image_size[0]/2),(0,1,image_size[1]/2),(0,0,1)])
    #K = 
    #distortion =
    
    grid = np.zeros(grid_size+(3,))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = image_size[0]/(grid_size[0]+1)*(i+1)
            y = image_size[1]/(grid_size[1]+1)*(j+1)
            pixel_vector = np.array([x,y,1])
            grid[i][j] = A.dot(pixel_vector)
    return grid

pass

