import re
import typing
from abc import abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import braincode.data
import braincode.metrics
from braincode.abstract import Object
from braincode.metrics import Metric
from braincode.plots import Plotter


class Analysis(Object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "code-" not in self.target:
            self._code_model_dim = ""
        self._loader = getattr(braincode.data, f"DataLoader{self._name}")(
            *args, **kwargs
        )

    @property
    def feature(self) -> str:
        return self._feature

    @property
    def target(self) -> str:
        return self._target

    @property
    def score(self) -> np.float:
        if not self._score:
            raise RuntimeError("Score not set. Need to run.")
        return self._score

    @property
    def null(self) -> np.ndarray:
        if not self._null:
            raise RuntimeError("Null not set. Need to run.")
        return self._null

    @property
    def pval(self) -> np.float:
        return (self.score < self.null).sum() / self.null.size

    def _get_fname(self, mode: str) -> Path:
        ids = [
            mode,
            self.feature.split("-")[1],
            self.target.split("-")[1],
            self._code_model_dim,
        ]
        name = re.sub("_+", "_", "_".join(ids).strip("_") + ".npy")
        return self._base_path.joinpath(".cache", "scores", self._name.lower(), name)

    def _set_and_save(
        self, mode: str, val: typing.Union[np.float, np.ndarray], fname: Path
    ) -> None:
        setattr(self, f"_{mode}", val)
        if not self._debug:
            np.save(fname, val)
        tag = f": {val:.3f}" if mode == "score" else ""
        self._logger.info(f"Caching '{fname.name}'{tag}.")

    def _run_pipeline(self, mode: str, iters: int = 1) -> None:
        if mode not in ["score", "null"]:
            raise RuntimeError("Mode set incorrectly. Must be 'score' or 'null'")
        fname = self._get_fname(mode)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        if fname.exists() and not self._debug:
            setattr(self, f"_{mode}", np.load(fname, allow_pickle=True))
            self._logger.info(f"Loading '{fname.name}' from cache.")
            return
        samples = np.zeros((iters))
        for idx in tqdm(range(iters)):
            score = self._run_mapping(mode)
            if mode == "score":
                self._set_and_save(mode, score, fname)
                return
            samples[idx] = score
        self._set_and_save(mode, samples, fname)

    def _check_metric_compatibility(self, Y: np.ndarray) -> np.ndarray:
        if self._metric not in ["", "ClassificationAccuracy"] and Y.ndim == 1:
            Y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))
        return Y

    @abstractmethod
    def _run_mapping(self, mode: str) -> typing.Union[np.float, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")

    def _plot(self) -> None:
        Plotter(self).plot()

    def run(self, iters: int = 1000, plot: bool = False) -> None:
        self._run_pipeline("score")
        if not self._score_only:
            self._run_pipeline("null", iters)
            if plot:
                self._plot()


class BrainAnalysis(Analysis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def subjects(self) -> typing.List[Path]:
        return [
            s
            for s in sorted(self._loader.datadir.joinpath("neural_data").glob("*.mat"))
            if "737" not in str(s)  # remove this subject as in Ivanova et al (2020)
        ]

    @abstractmethod
    def _load_subject(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Handled by subclass.")

    @staticmethod
    @abstractmethod
    def _shuffle(Y: np.ndarray, runs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Handled by subclass.")

    @abstractmethod
    def _calc_score(self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray) -> np.float:
        raise NotImplementedError("Handled by subclass.")

    def _run_mapping(self, mode: str, cache_subject_scores: bool = True) -> np.float:
        scores = np.zeros(len(self.subjects))
        for idx, subject in enumerate(self.subjects):
            X, Y, runs = self._load_subject(subject)
            Y = self._check_metric_compatibility(Y)
            if mode == "null":
                Y = self._shuffle(Y, runs)
            scores[idx] = self._calc_score(X, Y, runs)
        if mode == "score" and cache_subject_scores:
            temp_mode = "subjects"
            self._set_and_save(temp_mode, scores, self._get_fname(temp_mode))
        return scores.mean()


class Mapping(Analysis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_metric(self, Y: np.ndarray) -> Metric:
        if not getattr(self, "_metric"):
            if Y.ndim == 1:
                setattr(self, "_metric", "ClassificationAccuracy")
            elif Y.ndim == 2:
                if Y.shape[1] == 1:
                    setattr(self, "_metric", "PearsonR")
                else:
                    setattr(self, "_metric", "RankAccuracy")
            else:
                raise NotImplementedError("Metrics only defined for 1D and 2D arrays.")
        metric = getattr(braincode.metrics, self._metric)
        if not issubclass(metric, Metric):
            raise ValueError("Invalid metric specified.")
        return metric()

    def _cross_validate_model(
        self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray
    ) -> np.float:
        if any(a.shape[0] != b.shape[0] for a, b in combinations([X, Y, runs], 2)):
            raise ValueError("X Y and runs must all have the same number of samples.")
        model_class = RidgeClassifierCV if Y.ndim == 1 else RidgeCV
        scores = np.zeros(np.unique(runs).size)
        for idx, (train, test) in enumerate(LeaveOneGroupOut().split(X, Y, runs)):
            model = model_class(alphas=np.logspace(-2, 2, 9)).fit(X[train], Y[train])
            metric = self._get_metric(Y)
            scores[idx] = metric(model.predict(X[test]), Y[test])
        return scores.mean()


class BrainMapping(BrainAnalysis, Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_subject(
        self, subject: Path
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, runs = self._loader.get_data(self._name.lower(), subject=subject)
        return X, Y, runs

    @staticmethod
    def _shuffle(Y: np.ndarray, runs: np.ndarray) -> np.ndarray:
        if Y.shape[0] != runs.shape[0]:
            raise ValueError("Y and runs must have the same number of samples.")
        Y_shuffled = np.zeros(Y.shape)
        for run in np.unique(runs):
            Y_shuffled[runs == run] = np.random.permutation(Y[runs == run])
        return Y_shuffled

    def _calc_score(self, X: np.ndarray, Y: np.ndarray, runs: np.ndarray) -> np.float:
        score = self._cross_validate_model(X, Y, runs)
        return score
