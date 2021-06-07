from typing import Tuple

import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.sparse.csgraph import laplacian as graph_laplacian


# TODO: Make this an ABC
class ShapeModel:

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        pass

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        pass

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def fit_project(self, X: np.ndarray, y: np.ndarray = None, center: bool = True, dim: int = 2) -> np.ndarray:
        self.fit(X, y)
        return self.project(X, center, dim)


# TODO: rename, redo mode function
class LDA(ShapeModel):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X is the data
        y are the classes corresponding to X
        """
        # TODO: Check for small sample size condition
        self.mean = X.mean(axis=0)
        self.N = X.shape[0]
        X = (X - self.mean).reshape(self.N, -1)
        self.F = X.shape[1]

        # Class means
        self.classes = np.unique(y)
        # print(self.classes)
        self.class_means = {
            c: np.expand_dims(X[y == c].mean(axis=0), axis=1)
            for c in self.classes
        }

        # Between class
        SB = np.zeros((self.F, self.F))
        # Within class
        SW = np.zeros((self.F, self.F))
        for c in self.classes:
            if self.verbose:
                print('Class:', c)
            # print(c)
            # Between class
            class_mean = self.class_means[c]
            SB += np.dot(class_mean, class_mean.T)  # @ class_mean.T
            # print(np.dot(class_mean, class_mean.T).shape)
            # Within class
            centered_class_data = (X[y == c] - class_mean.flatten())
            # print(centered_class_data.shape)
            # TODO: vectorise
            class_SW = np.zeros((self.F, self.F))
            for i in range(centered_class_data.shape[0]):
                _datapoint = centered_class_data[i, :].reshape(self.F, 1)
                class_SW += np.dot(_datapoint, _datapoint.T)
                # print(np.dot(_datapoint, _datapoint.T).shape)
            SW += class_SW

        # TODO: explained variance
        # TODO: inv is not efficient for large matrices
        combined_scatter = np.linalg.inv(SW) @ SB

        # Eigenanalysis
        eigenvalues, eigenvectors = np.linalg.eigh(combined_scatter) 
        self.eigenvalues = eigenvalues[::-1]
        self.eigenvectors = eigenvectors[::-1]

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        # TODO: dim < classes
        assert dim > 0 and dim < len(self.classes)
        if center:
            X = X - self.mean
        X = X.reshape(X.shape[0], -1)
        return X @ self.eigenvectors.T[:, :dim]

    def mode(self, mode_no: int, scale_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Correct this
        # eig = np.sqrt(self.eigenvalues[mode_no])
        # print(self.eigenvalues[mode_no])
        # print(eig)
        # mode = self.V_T.T[:, mode_no].reshape(-1, 3)
        # print(self.eigenvalues.shape)
        # print(self.eigenvalues[:, mode_no].shape)
        mode = self.eigenvectors[:, mode_no].reshape(-1, 3)
        pos = self.mean + stddevs * mode
        neg = self.mean - stddevs * mode 
        return pos, neg


class PCAShapeModel3D(ShapeModel):

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        y is present for API consistency.

        Assuming:
            X.shape[0] = Batch size
            X.shape[1] = Points in the shape
            X.shape[2] = Dimensions of shape == 3 
        """
        assert X.shape[-1] == 3 and len(X.shape) == 3

        self.N = X.shape[0]

        # mean center along each dimension
        self.mean = X.mean(axis=0)
        X = (1 / np.sqrt(self.N - 1)) * (X - self.mean).reshape(self.N, -1)
        assert self.N > X.shape[1], 'Samples must be greater than features'

        # SVD
        self.U, self.S, self.V_T = np.linalg.svd(X, full_matrices=True)

        # max_abs_cols = np.argmax(np.abs(U), axis=0)
        # signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        # U *= signs
        # V_T *= signs[:, np.newaxis]

        self.explained_variance = (self.S ** 2) / (self.N - 1)
        var_sum = self.explained_variance.sum()
        self.explained_variance_ratio = self.explained_variance / var_sum

    def low_rank_approx(self, k: int) -> np.ndarray:
        # TODO: Ensure fit
        # TODO: Ensure k <= f and k <= n
        # mat = np.zeros((self.U.shape[0], self.V_T.shape[0]))
        # mat[:k, :k] = np.diag(self.S[:k])
        # np.dot(np.dot(u,s),vh)
        lr_approx = self.U[:, :k] @ np.diag(self.S[:k]) @ self.V_T[:k, :]
        # lr_approx = np.dot(np.dot(self.U[:, :k], np.diag(self.S[:k])), self.V_T[:k, :])
        return lr_approx.reshape(self.N, -1, 3) + self.mean

    def principal_components(self) -> np.ndarray:
        return self.U @ np.diag(self.S)

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        # TODO: Remove center argument - assume uncentered
        assert dim > 0
        X = X - self.mean
        X = X.reshape(X.shape[0], -1)
        return X @ self.V_T.T[:, :dim]

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        eig = self.S[mode_no]
        mode = self.V_T.T[:, mode_no].reshape(-1, 3)
        pos = self.mean + stddevs * eig * mode 
        neg = self.mean - stddevs * eig * mode 
        return pos, neg


class KPCAShapeModel3D(ShapeModel):

    def __init__(self, kernel_type: str = 'rbf', max_comps: int = 200):
        """
        Kernel can be any of the following: 
            [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’,
            ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’],
        as per sklearn.metrics.pairwise.pairwise_kernels
        """
        self.kernel_type = kernel_type
        self.max_comps = max_comps

    def _calc_kernel(self, X, Y=None):
        return pairwise_kernels(X, Y, metric=self.kernel_type)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        y is present for API consistency.

        Assuming:
            X.shape[0] = Batch size
            X.shape[1] = Points in the shape
            X.shape[2] = Dimensions of shape == 3 
        """
        # assert X.shape[-1] == 3 and len(X.shape) == 3

        self.N = X.shape[0]
        self.X = X.reshape(self.N, -1)

        kernel = self._calc_kernel(self.X)

        self.center = np.mean(kernel, axis=1) - np.mean(kernel)
        self.center = self.center.reshape(-1, 1)
        ones = np.ones_like(kernel) / self.N
        centered_kernel = kernel
        centered_kernel -= ones @ kernel
        centered_kernel -= kernel @ ones
        centered_kernel += ones @ kernel @ ones

        comps = min(kernel.shape[0], self.max_comps)
        e_vals, e_vecs = scipy.linalg.eigh(
            centered_kernel,
            eigvals=(kernel.shape[0] - comps, kernel.shape[0] - 1)
        )

        # TODO: Deterministic sign flip
        self.e_vals = np.sqrt(e_vals[::-1])
        self.e_vecs = e_vecs[:, ::-1] / self.e_vals 

        # self.explained_variance = (self.e_vals ** 2) / (self.N - 1)
        # var_sum = self.explained_variance.sum()
        # self.explained_variance_ratio = self.explained_variance / var_sum

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        # TODO: Remove center argument - assume uncentered
        assert dim > 0
        X = X.reshape(X.shape[0], -1)
        kernel = self._calc_kernel(self.X, X)
        kernel -= np.mean(kernel, axis=1)[:, np.newaxis]
        kernel -= self.center
        return np.dot(kernel.T, self.e_vecs[:, :dim])

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        if self.kernel == '':
            eig = self.S[mode_no]
            mode = self.V_T.T[:, mode_no].reshape(-1, 3)
            pos = self.mean + stddevs * eig * mode 
            neg = self.mean - stddevs * eig * mode 
            return pos, neg
        else:
            raise NotImplementedError()


class GraphSpectralFiltering(ShapeModel):

    def __init__(self, n_verts: int, laplacian_type: str = None):
        self.n_verts = n_verts
        self.laplacian_type = laplacian_type
        self.cache = {}

    def __calc_adj_matrix(self, faces):
        adj = np.zeros((self.n_verts, self.n_verts))
        for a, b, c in faces:
            adj[a, b] = True
            adj[a, b] = True
            adj[b, a] = True
            adj[a, c] = True
            adj[c, a] = True
            adj[b, c] = True
            adj[c, b] = True
        self.adj = adj
        return adj

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        X is the list of faces 
        y is present for API consistency

        Assuming:
            X.shape[0] = No of faces
            X.shape[1] = 3 because we assume triangulated faces 
        """
        self.__calc_adj_matrix(X)
        if self.laplacian_type is None:
            self.lap, self.deg = graph_laplacian(self.adj, normed=False, return_diag=True)
        elif self.laplacian_type == 'sym':
            self.lap, self.deg = graph_laplacian(self.adj, normed=True, return_diag=True)
        elif self.laplacian_type == 'rw':
            lap, self.deg = graph_laplacian(self.adj, normed=False, return_diag=True)
            # self.lap = np.diag(1 / self.deg) @ lap
            self.lap = np.multiply(lap, (1 / self.deg)[:, np.newaxis])
            # self.lap = np.multiply(lap, 1 / self.deg.reshape(-1, 1))
            # np.arange(0, 50).reshape(10, 5),(np.ones(10) * 0.5).reshape(-1, 1)
        elif self.laplacian_type == 'sym_scaled':
            lap, self.deg = graph_laplacian(self.adj, normed=True, return_diag=True)
            max_eig_index = lap.shape[0] - 1
            e_val, _ = scipy.linalg.eigh(lap, eigvals=(max_eig_index, max_eig_index))
            self.lap = ((2 * lap) / e_val[0]) - np.eye(lap.shape[0])
        else:
            assert NotImplementedError
        self.e_vals, self.e_vecs = scipy.linalg.eigh(self.lap)
        # Reset Cache
        self.cache = {}

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        """
        Spectral filtering using spectral values up to mode_no

        X: vertices with the same connectivity as faces provided when fitting ^^^
            i.e. should be from the same domain as the Laplacian modes

        Assuming:
            X.shape[0] = No of faces
        """
        assert X.shape[0] == self.n_verts and X.shape[1] == 3

        if dim in self.cache:
            low_rank_lap = self.cache[dim]
        else:
            modes = self.e_vecs[:, :dim]
            spectrum = np.diag(self.e_vals[:dim])
            low_rank_lap = modes @ spectrum @ modes.T 
            # TODO: Change to using an LRU cache 
            self.cache[dim] = low_rank_lap

        mesh_proj = np.zeros_like(X)
        mesh_proj[:, 0] = low_rank_lap @ X[:, 0]
        mesh_proj[:, 1] = low_rank_lap @ X[:, 1]
        mesh_proj[:, 2] = low_rank_lap @ X[:, 2]
        return mesh_proj

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        pass
