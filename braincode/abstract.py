import itertools
import logging
import typing
from abc import ABC


class Object(ABC):
    def __init__(self, *args, **kwargs):
        self._name = self.__class__.__name__
        self._logger = logging.getLogger(self._name)
        if args and kwargs:
            for name, value in itertools.chain(
                zip(["feature", "target"], args), kwargs.items()
            ):
                setattr(self, f"_{name}", value)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> typing.Any:
        return super().__getattribute__(name)
