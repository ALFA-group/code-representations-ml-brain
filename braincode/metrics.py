import typing
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, mean_squared_error, pairwise_distances


class Metric(ABC):
    def __init__(self) -> None:
        pass

    def __call__(
        self, X: np.ndarray, Y: np.ndarray
    ) -> typing.Union[np.float, np.ndarray]:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if any(y.ndim != 2 for y in [X, Y]):
            raise ValueError("X and Y must be 1D or 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.__class__.__name__ not in ["RepresentationalSimilarity", "LinearCKA"]:
            if X.shape[1] != Y.shape[1]:
                raise ValueError("X and Y must have the same number of dimensions.")
        return self._apply_metric(X, Y)

    @abstractmethod
    def _apply_metric(
        self, X: np.ndarray, Y: np.ndarray
    ) -> typing.Union[np.float, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")


class VectorMetric(Metric):
    def __init__(self, reduction: typing.Callable = np.mean) -> None:
        if reduction:
            if not callable(reduction):
                raise TypeError("Reduction argument must be callable.")
        self._reduction = reduction
        super().__init__()

    def _apply_metric(
        self, X: np.ndarray, Y: np.ndarray
    ) -> typing.Union[np.float, np.ndarray]:
        scores = np.zeros(X.shape[1])
        for i in range(scores.size):
            scores[i] = self._score(X[:, i], Y[:, i])
        if self._reduction:
            return self._reduction(scores)
        return scores

    @staticmethod
    @abstractmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        raise NotImplementedError("Handled by subclass.")


class MatrixMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def _apply_metric(self, X, Y):
        score = self._score(X, Y)
        return score

    @abstractmethod
    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        raise NotImplementedError("Handled by subclass.")


class PearsonR(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        r, _ = pearsonr(x, y)
        return r


class SpearmanRho(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        rho, _ = spearmanr(x, y)
        return rho


class KendallTau(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        tau, _ = kendalltau(x, y)
        return tau


class FisherCorr(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        r, _ = pearsonr(x, y)
        corr = np.arctanh(r)
        return corr


class RMSE(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        loss = mean_squared_error(x, y, squared=False)
        return loss


class ClassificationAccuracy(VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        score = accuracy_score(x, y, normalize=True)
        return score


class RankAccuracy(MatrixMetric):
    def __init__(self, distance: str = "euclidean") -> None:
        self._distance = distance
        super().__init__()

    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        distances = pairwise_distances(X, Y, metric=self._distance)
        scores = (distances.T > np.diag(distances)).sum(axis=0) / (
            distances.shape[1] - 1
        )
        return scores.mean()


class RepresentationalSimilarity(MatrixMetric):
    def __init__(
        self, distance: str = "correlation", comparison: VectorMetric = PearsonR()
    ) -> None:
        self._distance = distance
        self._comparison = comparison
        super().__init__()

    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        X_rdm = pairwise_distances(X, metric=self._distance)
        Y_rdm = pairwise_distances(Y, metric=self._distance)
        if any(m.shape[1] == 1 for m in (X, Y)):  # can't calc 1D corr dists
            X_rdm[np.isnan(X_rdm)] = 0
            Y_rdm[np.isnan(Y_rdm)] = 0
        indices = np.triu_indices(X_rdm.shape[0], k=1)
        score = self._comparison(X_rdm[indices], Y_rdm[indices])
        return score


# inspired by https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
class LinearCKA(MatrixMetric):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _center(K: np.ndarray) -> np.ndarray:
        N = K.shape[0]
        U = np.ones([N, N])
        I = np.eye(N)
        H = I - U / N
        centered = H @ K @ H
        return centered

    def _HSIC(self, A: np.ndarray, B: np.ndarray) -> np.float:
        L_A = A @ A.T
        L_B = B @ B.T
        HSIC = np.sum(self._center(L_A) * self._center(L_B))
        return HSIC

    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        HSIC_XY = self._HSIC(X, Y)
        HSIC_XX = self._HSIC(X, X)
        HSIC_YY = self._HSIC(Y, Y)
        score = HSIC_XY / (np.sqrt(HSIC_XX) * np.sqrt(HSIC_YY))
        return score
