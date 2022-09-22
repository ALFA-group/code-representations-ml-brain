from pathlib import Path

import matplotlib.pyplot as plt

from braincode.abstract import Object


class Plotter(Object):
    def __init__(self, analysis) -> None:
        super().__init__()
        self._analysis = analysis
        self._feature = self._analysis.feature.split("-")[1]
        self._target = self._analysis.target.split("-")[1]
        self._type = self._analysis._name

    @property
    def _fname(self) -> Path:
        fname = self._analysis._base_path.joinpath(
            "outputs",
            "plots",
            self._type.lower(),
            f"{self._feature}_{self._target}.png",
        )
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        return fname

    def plot(self, show: bool = False) -> None:
        plt.hist(self._analysis.null, bins=25, color="turquoise", edgecolor="black")
        plt.axvline(self._analysis.score, color="black", linewidth=3)
        plt.xlim([-1, 1])
        plt.savefig(self._fname)
        plt.show() if show else plt.clf()
        self._logger.info(f"Plotting '{self._fname.name}'.")
