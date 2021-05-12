from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree


class RegistrationBase(ABC):

    @abstractmethod
    def align(self, moving_image: np.ndarray, moving_image_faces: np.ndarray, n_iter: int, eps: float) -> np.ndarray:
        pass


class RigidRegistration(RegistrationBase):
    
    def __init__(self, fixed_image: np.ndarray):
        self.fixed_image = fixed_image
        self.fixed_image_kd_tree = KDTree(fixed_image)
        self.fixed_mean_centered, self.fixed_mean = self.mean_centering(fixed_image)
        self.fixed_vertices = fixed_image.shape[0]
        self.fixed_dim = fixed_image.shape[1]

    def get_fixed_mean_centering(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fixed_mean_centered, self.fixed_mean

    def mean_centering(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Accept a density weighting for each pixel in the image
        mean = np.mean(image, axis=0)
        mean_centering = image - mean
        return mean_centering, mean
    
    def calc_rotation_matrix(self, moving_image_mean_centered: np.ndarray, moving_mean: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapted from https://johnwlambert.github.io/icp/

        Procrustes Analysis: https://en.wikipedia.org/wiki/Procrustes_analysis
        Kabsch Algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm

        Args:
            moving_image: Mean centered array of shape (N, D) -- Point Cloud to Align (source)

        Returns:
            R: optimal rotation (D, D)
            t: optimal translation (D, )
        """
        assert moving_image_mean_centered.shape == (self.fixed_vertices, self.fixed_dim)
        
        cross_cov = moving_image_mean_centered.T @ self.fixed_mean_centered
        U, _, V_T = np.linalg.svd(cross_cov)
        
        # TODO: Elaborate on reflection case
        S = np.eye(self.fixed_dim)
        det = np.linalg.det(U) * np.linalg.det(V_T.T)
        if not np.isclose(det, 1.):
            s[self.fixed_dim - 1, self.fixed_dim - 1] = -1
        
        rot = U @ S @ V_T
        trans = self.fixed_mean - moving_mean @ rot
        
        return rot, trans
    
    def manual_rotations(self, moving_image: np.ndarray) -> np.ndarray:
        for theta in range(5, 360, 5):
            radians = np.radians(theta)
            rotate = np.array([
                [math.cos(radians), 0, -math.sin(radians)],
                [0, 1, 0],
                [math.sin(radians), 0, math.cos(radians)],
            ])
            moving_image_rotated = moving_image @ rotate
            knn_dist, l2_dist = self.calc_error(moving_image_rotated)
            print(theta, knn_dist, l2_dist)

        return moving_image_rotated
    
    def apply(self, moving_image_mean_centered: np.ndarray, rotate: np.ndarray, translate: np.ndarray) -> np.ndarray:
        return moving_image_mean_centered @ rotate + translate
    
    def calc_error(self, moving_image: np.ndarray, knn_bi_dir: bool = False):
        # TODO: Could replace knn_dist with chamfer dist
        knn_dist = self.fixed_image_kd_tree.query(moving_image)[0].mean()
        if knn_bi_dir:
            # Make KD tree and find nn in opposite direction
            # Calculate mean knn_dist
            pass
        l2_dist = np.linalg.norm(moving_image - self.fixed_image)
        return knn_dist, l2_dist
    
    def align(self, moving_image: np.ndarray, moving_image_faces: np.ndarray = None, n_iter: int = 1, eps: float = 1e-2) -> np.ndarray:
        knn_dist, l2_dist = self.calc_error(moving_image)
        # print(knn_dist, l2_dist)
        if knn_dist < eps:
            return moving_image
        
        for _ in range(n_iter):
            moving_image_mean_centered, moving_mean = self.mean_centering(moving_image)
            r, t = self.calc_rotation_matrix(moving_image_mean_centered, moving_mean)
            moving_image = self.apply(moving_image, r, t)
            knn_dist, l2_dist = self.calc_error(moving_image)
            # print(knn_dist, l2_dist)
            if knn_dist < eps:
                return moving_image
        
        # moving_image = self.manual_rotations(moving_image)
        
        return moving_image
