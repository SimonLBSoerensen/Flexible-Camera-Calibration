import numpy as np
from scipy.optimize import least_squares, minimize
from tqdm import tqdm
import multiprocessing
import pickle

import knot_generators as kg

# The implementation of this B-spline-based camera model is based on the description in the
# article "Generalized B-spline Camera Model" by Johannes Beck and Christoph Stiller.

# Resources:
# Wikipedia article on b-splines: https://en.wikipedia.org/wiki/B-spline
# Notes of how to construct b-splines: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html

class CentralModel:
    def __init__(self, image_dimensions, grid_dimensions, control_points, order, knot_method = 'open_uniform', min_basis_value=0.001, end_divergence = 1e-10, save_basis_values=False):
        """Initializes a CentralModel object. \n

        Keyword arguments: \n
        image_dimensions: tuple containing (width, height) of image in pixels. \n
        grid_dimensions: tuple containing (width, height) of grid in pixels. \n
        control_points: numpy 3d array containing unit vectors describing camera intrinsics.\n
        order: the order of the interpolation.\n
        knot_method: the method used to generate the knot vector.\n
        min_basis_value: the minimum basis value. Used for optimization.\n
        end_divergence: a small term added to the tails of the knot vector when 'open_uniform' is used. Allows for sampling at the endpoints of the spline.\n
        save_basis_values: lets the model save basis calculations. This increases memory use and should mainly be used if sampling with repeated u or v coordinates.
        """
        assert knot_method in ['open_uniform', 'uniform', 'none'], 'knot method should be one of the implemented methods.'
        assert isinstance(image_dimensions, tuple) and len(image_dimensions) == 2, 'image_dimensions must be a 2-dimensional tuple.'
        assert isinstance(grid_dimensions, tuple) and len(grid_dimensions) == 2, 'grid_dimensions must be a 2-dimensional tuple.'
        assert isinstance(control_points, np.ndarray) and len(control_points.shape) == 3 and control_points.shape[-1] == 3, 'grid must be a 3-dimensional numpy array with a depth of 3.'
        assert control_points.shape[0] > order and control_points.shape[1] > order, 'order must be smaller than grid size.'

        self.image_width = image_dimensions[0]
        self.image_height = image_dimensions[1]

        self.grid_width = grid_dimensions[0]
        self.grid_height = grid_dimensions[1]

        self.n = control_points.shape[0]
        self.m = control_points.shape[1]

        self.a = control_points

        self.order = order

        self.B = {}

        if knot_method == 'open_uniform':
            self.th = kg.open_uniform(self.n, order, end_divergence)
            self.tv = kg.open_uniform(self.m, order, end_divergence)

        if knot_method == 'uniform':
            self.th = kg.uniform(self.n, order)
            self.tv = kg.uniform(self.m, order)

        if knot_method == 'none':
            self.th = np.array([])
            self.tv = np.array([])

        self.min_basis_value = min_basis_value

        self.save_basis_values = save_basis_values

    def _B(self, i, k, t, x):
        """Used to calculate the basis function \n

        Keyword arguments: \n
        i: Index for which to sample grid. \n
        k: Basis function order. \n
        t: Knot vector. \n
        x: Pixel coordinate of sample.   
        """
        if self.save_basis_values and (i, k, tuple(t), x) in self.B:
            return self.B[(i, k, tuple(t), x)]

        # Equation 2
        if k == 0:
            if t[i] <= x < t[i + 1]:
                return 1
            else:
                return 0

        # Equation 3
        term1a = x - t[i]
        term1b = t[i + k] - t[i]
        term1c = self._B(i, k - 1, t, x)

        # If term1b is zero, the division will be undefined.
        # 'Solution' suggested in https://en.wikipedia.org/wiki/Talk%3AB-spline#Avoiding_division_by_zero
        if term1b == 0:
            term1a = term1b = 1

        term2a = t[i + k + 1] - x
        term2b = t[i + k + 1] - t[i + 1]
        term2c = self._B(i + 1, k - 1, t, x)

        # If term2b is zero, the division will be undefined.
        if term2b == 0:
            term2a = term2b = 1

        B = term1a / term1b * term1c + term2a / term2b * term2c

        if self.save_basis_values:
            self.B[(i, k, tuple(t), x)] = B

        return B

    def _normalize(self, coord, grid_size, image_size):
        """Returns normalized coordinates. \n

        Keyword arguments: \n
        coord: Pixel coordinate in one axis.\n
        grid_size: the grid size in pixels in the same axis.\n
        image_size: the image size in pixels in the same axis.
        """
        t = coord + (grid_size - image_size) / 2
        t /= grid_size - 1

        return t

    def sample(self, u, v, normalize=False):
        """Used to sample the b-spline surface. \n

        Keyword arguments: \n
        u: Horizontal pixel coordinate of sample. \n
        v: Vertical pixel coordinate of sample. 
        """
        assert not (isinstance(u, (np.ndarray, list)) or isinstance(v, (np.ndarray, list))) and all(np.isreal([u, v])), 'u and v must be numbers.'

        u = self._normalize(u, self.grid_width, self.image_width)
        v = self._normalize(v, self.grid_height, self.image_height)

        Bh = np.array([self._B(i, self.order, self.th, u) for i in range(self.n)])
        Bv = np.array([self._B(j, self.order, self.tv, v) for j in range(self.m)])

        # Optimization
        vi = np.where(Bh >= self.min_basis_value)[0]
        vj = np.where(Bv >= self.min_basis_value)[0]

        res = np.full((3,), 0.0)
        for i in vi:
            for j in vj:
                aij = self.a[i, j]
                res += np.multiply(aij, Bh[i] * Bv[j])

        if normalize and np.linalg.norm(res) > 0:
            res /= np.linalg.norm(res)

        return res

    def sample_grid(self):
        """Used to sample the b-spline surface in the input coordinates corresponding to the control points."""
        xs = np.floor((self.grid_width - 1) / (self.n - 1) * np.arange(0, self.n))
        ys = np.floor((self.grid_height - 1) / (self.m - 1) * np.arange(0, self.m))

        pts = np.transpose(np.meshgrid(xs, ys))

        samples = np.ndarray((self.n, self.m, 3))
        for i in range(0, self.n):
            for j in range(0, self.m):
                samples[i,j] = self.sample(pts[i,j,0], pts[i,j,1])

        return samples

    def _task(self, i, pts, output):
        output.put(np.array([np.hstack([pt[0], self.sample(pt[1], pt[2])]) for pt in pts]))

    def _sample_many(self, pts, threads=1):
        """(DON'T USE THIS METHOD) 
        Used to sample the b-spline surface at multiple points. This method is multi-threaded and will start additional processes even when thread is set to 1. \n

        Keyword arguments: \n
        pts: (n, 2) array of pixel coordinates. \n
        threads: Amount of processes to spawn. 
        """

        assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[1] == 2, '\'pts\' should be a numpy array of shape (n,2) containing the points for which to sample the b-spline.'
        assert isinstance(threads, int) and threads >= 1, '\'threads\' should be a positive integer. Got object with type {}.'.format(type(threads))

        results = multiprocessing.Queue()

        _pts = np.ndarray((len(pts), 3))
        _pts[:,0] = range(len(pts))
        _pts[:,1:] = pts

        split = np.split(_pts, threads)

        processes = [multiprocessing.Process(target=self._task, args=(i, split[threads - i - 1], results)) for i in range(threads)]

        for process in processes:
            process.start()

        results = np.vstack([results.get() for i in range(threads)])
        results = results[results[:,0].argsort()]

        for process in processes:
            process.join()

        return results[:,:-1]

    def active_control_points(self, u, v):
        """Returns the indeces of the control points that are used to calculate the point with the given pixel coordinates. \n

        Keyword arguments: \n
        u: Horizontal pixel coordinate. \n
        v: Vertical pixel coordinate. 
        """
        assert not (isinstance(u, (np.ndarray, list)) or isinstance(v, (np.ndarray, list))) and all(np.isreal([u, v])), 'u and v must be numbers.'

        is_even = lambda x: x % 2 == 0

        dx = self._normalize(u, self.grid_width, self.image_width)
        dy = self._normalize(v, self.grid_height, self.image_height)

        px = (self.a.shape[0] - 1) * dx
        py = (self.a.shape[1] - 1) * dy

        between = lambda a, x, b: max(min(x,b),a)

        px = between(self.order // 2, px, self.a.shape[0] - 1 - self.order // 2)
        py = between(self.order // 2, py, self.a.shape[1] - 1 - self.order // 2)

        if is_even(self.order):
            x = np.arange(np.ceil(-0.5*(self.order + 1)), np.ceil(0.5*self.order) + 1) + np.round(px)
            y = np.arange(np.ceil(-0.5*(self.order + 1)), np.ceil(0.5*self.order) + 1) + np.round(py)

        else:
            x = np.arange(np.ceil(-0.5*self.order), np.ceil(0.5*self.order) + 1) + np.floor(px)
            y = np.arange(np.ceil(-0.5*self.order), np.ceil(0.5*self.order) + 1) + np.floor(py)

        return np.transpose(np.meshgrid(x, y))

    def active_control_points_2(self, u, v):
        """Returns the indeces of the control points that are used to calculate the point with the given pixel coordinates. \n

        Keyword arguments: \n
        u: Horizontal pixel coordinate. \n
        v: Vertical pixel coordinate. 
        """
        assert not (isinstance(u, (np.ndarray, list)) or isinstance(v, (np.ndarray, list))) and all(np.isreal([u, v])), 'u and v must be numbers.'

        w = self.a.shape[0]
        h = self.a.shape[1]

        u = self._normalize(u, self.grid_width, self.image_width)
        v = self._normalize(v, self.grid_height, self.image_height)

        pairs = []

        for x in range(w):
            for y in range(h):
                val = self._B(x, self.order, self.th, u) * self._B(y, self.order, self.tv, v)
                if val > 0:
                    pairs.append([x, y])

        return np.array(pairs)

    def forward_sample(self, ray, x0=None, use_bounds=True, method='Powell', tol=None, options={}, return_results=False, normalize=True):
        """Returns the corresponding pixel coordinates for a given ray. \n

        Keyword arguments: \n
        ray: The projected ray of the point in the image plane. \n
        x0: An initial guess on the position of the point. If set to none the center of the image is used. \n
        use_bounds: Sets the image dimensions as bounds for the optimizer. \n
        method: Optimization method from scipy's \'minimize\'. \n
        options: Options for the optimization. See scipu's \'minimize\'. \n
        return_results: If True the function returns the result of the minimization. Otherwise the function just returns the pixel coordinates of the ray.
        """
        assert isinstance(ray, np.ndarray) and ray.size == 3, '\'ray\' must be an array containing the direction of a 3d ray.'

        if normalize:
            ray = ray.reshape((3,))
            ray /= np.linalg.norm(ray)

        def fun(params, ray, normalize):
            u, v = params[0], params[1]
            s = self.sample(u, v, normalize=normalize)

            return np.linalg.norm(s - ray)

        if x0 is None:
            x0 = np.array([self.image_width/2, self.image_height/2])

        bounds = None
        if use_bounds and not method in ['Powell']:
            bounds = np.array([(0, self.image_width), (0, self.image_height)])

        res = minimize(fun, x0, args=(ray, normalize), tol=tol, bounds=bounds, method=method, options=options)

        if return_results:
            return res
        return res['x']

def fit_central_model(target_values, image_dimensions, grid_dimensions, order = 3, knot_method = 'open_uniform', end_divergence = 0, min_basis_value = 1e-3, verbose = 0, initial_values = None):
    """Used to fit the control points so the spline interpolates between the values in 'target_values'. Returns the fitted CentralModel and the least squares fitting results. \n

    Keyword arguments: \n
    target_values: The values the solver will try to fit the b-spline surface to.\n
    image_dimensions: see CentralModel's initialization.\n
    grid_dimensions: see CentralModel's initialization.\n
    order: see CentralModel's initialization.\n
    knot_method: see CentralModel's initialization.\n
    end_divergence: see CentralModel's initialization.\n
    min_basis_value: see CentralModel's initialization.\n
    verbose: Changes how much the function prints while running.\n
    initial_values: Optional numpy array with the shape of 'target_values'. Provides the initial control points to the least squares solver.
    """
    if initial_values == None:
        initial_values = target_values

    assert target_values.shape == initial_values.shape, 'target_values and initial_values must have the same shape.'

    shape = target_values.shape

    def fun(params, target_values, grid_shape, image_dimensions, grid_dimensions, order, knot_method, end_divergence, min_basis_value):
        grid = np.reshape(params, grid_shape)

        cm = CentralModel(
            image_dimensions=image_dimensions,
            grid_dimensions=grid_dimensions,
            control_points=grid,
            order=order,
            knot_method=knot_method,
            end_divergence=end_divergence,
            min_basis_value=min_basis_value)

        grid_samples = cm.sample_grid().ravel()
        return target_values - grid_samples

    if verbose > 0:
        print('Starting least squares fitting of CentralModel.')

    result = least_squares(fun, initial_values.ravel(), verbose=verbose, args=(target_values.ravel(), shape, image_dimensions, grid_dimensions, order, knot_method, end_divergence, min_basis_value))
    target_values = np.reshape(target_values, (-1,3))

    ctrl = result['x'].reshape(shape)

    cm = CentralModel(
        image_dimensions=image_dimensions,
        grid_dimensions=grid_dimensions,
        control_points=ctrl,
        order=order,
        knot_method=knot_method,
        end_divergence=end_divergence,
        min_basis_value=min_basis_value)

    return cm, result

def cm_load(fp):
    image_width, image_height, grid_width, grid_height, a, order, B, th, tv, min_basis_value, save_basis_values = np.load(fp, allow_pickle=True).values()
    cm = CentralModel((image_width, image_height), (grid_width, grid_height), a, order, 'none', min_basis_value, 0, save_basis_values)
    cm.th = th
    cm.tv = tv
    cm.B = B

    return cm

def cm_save(cm, fp):
    np.savez(fp, cm.image_width, cm.image_height, cm.grid_width, cm.grid_height, cm.a, cm.order, cm.B, cm.th, cm.tv, cm.min_basis_value, cm.save_basis_values)


if __name__ == '__main__':
    import time

    shape = (500, 500)

    ## Multithreading Test
    if False:
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        pts = np.transpose(np.meshgrid(x,y)).reshape(-1, 2)

        print('Test underway. It might take some time. Please wait...')

        start = time.time()
        cm = CentralModel(shape, shape, np.random.normal(0, 1, (6,6,3)), 4)
        _ = cm._sample_many(pts, 16)
        end = time.time()

        print('Threaded version used {:.2f} s. ({} iter/s)'.format(end - start, int(len(pts) / (end - start))))

        start = time.time()
        cm = CentralModel(shape, shape, np.random.normal(0, 1, (6,6,3)), 4)
        _ = cm._sample_many(pts, 1)
        end = time.time()

        print('Non-threaded version used {:.2f} s. ({} iter/s)'.format(end - start, int(len(pts) / (end - start))))

    ## Sparsity Test
    if False:
        dim = (200, 200)
        grid = (5,5,3)

        ctrl = np.random.normal(0, 1, grid)

        cm = CentralModel(dim, dim, ctrl, 2)

        n = 10

        pts = []

        for i in range(n):
            pts.append(cm.active_control_points(i * (dim[0] / (n-1)), 100))

    ## Forward sampling test
    if False:
        dim = (200, 200)
        grid = (5,5,3)
        order = 3

        ctrl = np.full(grid, 0)

        for i, row in enumerate(ctrl):
            for j, cell in enumerate(row):
                ctrl[i, j, 0] = i
                ctrl[i, j, 1] = j
                ctrl[i, j, 2] = i * j

        cm = CentralModel(dim, dim, ctrl, order)

        ray = np.array([[4], [2], [8]])

        options = {'disp': True,'xtol': 1e-8, 'ftol': 1e-8}
        u, v = cm.forward_sample(ray, method='Powell', tol=1e-8, options=options)

        s = cm.sample(u, v)

        print(ray, s, ray-s)


    ## Save/Load Test
    if True:
        dim = (200, 200)
        grid = (5,5,3)
        order = 3

        ctrl = np.full(grid, 0)

        for i, row in enumerate(ctrl):
            for j, cell in enumerate(row):
                ctrl[i, j, 0] = i
                ctrl[i, j, 1] = j
                ctrl[i, j, 2] = i * j

        cm = CentralModel(dim, dim, ctrl, order)

        with open('temp_file.npz', 'wb') as file:
            cm_save(cm, file)

        cm = None

        with open('temp_file.npz', 'rb') as file:
            cm = cm_load(file)

        ray = np.array([[4.0], [2.0], [8.0]])

        options = {'disp': True,'xtol': 1e-8, 'ftol': 1e-8}
        u, v = cm.forward_sample(ray, method='Powell', tol=1e-8, options=options)

        s = cm.sample(u, v, normalize=True).reshape((3,1))

        print('Original ray:\n{}\n\nSampled ray:\n{}\n\nDiff:\n{}\n'.format(ray, s, ray-s))

    ## Real world forward sampling test
    if False:
        filename = r'json\res_2020-04-26_23-18-20.pickle'

        with open(filename, 'rb') as file:
            ctrl = np.array(pickle.load(file)['spline_grid'])

        dim = (1600, 1200)
        cm = CentralModel(dim, dim, ctrl, 2)

        sample_point = (2, 2)
        error_mean = 0
        error_std = 0.0005

        method = 'Powell'
        #options = {'xtol': 1e-15, 'ftol': 1e-15}

        ray = cm.sample(sample_point[0], sample_point[1])
        print('Sampled model in coordinates {}, which resulted in ray {}.'.format(sample_point, ray))

        # add noise
        if False:
            ray += np.random.normal(error_mean, error_std, 3)
            print('Added noise to ray. Now {}.'.format(ray))

        print('using {}'.format(method))
        try:
            u, v = cm.forward_sample(ray, method=method, options=options)
            error = (sample_point[0] - u, sample_point[1] - v)

            print('Found coordinates {} for ray which deviates {} ({} norm).'.format((u, v), error, np.linalg.norm(error)))
        except:
            print('Method caused error.')

        print('')