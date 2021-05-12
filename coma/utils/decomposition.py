import numpy as np
from typing import Tuple

# TODO: Make this an ABC
class ShapeModel:

    def fit(self, X: np.ndarray):
        pass

    def project(self, X: np.ndarray, center: bool = True, dim: int = 2) -> np.ndarray:
        pass

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        pass


# TODO: rename, redo mode function
class LDAShapeModel3D(ShapeModel):

    def __init__(self):
        pass

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
        assert dim > 0 and len(X.shape) == 3
        # TODO: dim <= classes - 1
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

    def fit(self, X: np.ndarray):
        """
        Assuming:
            X.shape[0] = Batch size
            X.shape[1] = Points in the shape
            X.shape[2] = Dimensions of shape == 3 
        """
        assert X.shape[-1] == 3 and len(X.shape) == 3
        # mean center along each dimension
        self.mean = X.mean(axis=0)
        self.N = X.shape[0]
        X = (X - self.mean).reshape(self.N, -1)

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
        assert dim > 0
        if center:
            X = X - self.mean
        X = X.reshape(X.shape[0], -1)
        return X @ self.V_T.T[:, :dim]

    def mode(self, mode_no: int, stddevs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        eig = np.sqrt(self.S[mode_no])
        mode = self.V_T.T[:, mode_no].reshape(-1, 3)
        pos = self.mean + stddevs * eig * mode 
        neg = self.mean - stddevs * eig * mode 
        return pos, neg
