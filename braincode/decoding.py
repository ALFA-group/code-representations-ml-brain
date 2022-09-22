import numpy as np

from braincode.analyses import BrainMapping, Mapping


class MVPA(BrainMapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PRDA(Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _run_mapping(self, mode: str) -> np.float:
        X, Y, runs = self._loader.get_data(self._name.lower())
        Y = self._check_metric_compatibility(Y)
        if mode == "null":
            np.random.shuffle(Y)
        return self._cross_validate_model(X, Y, runs)
