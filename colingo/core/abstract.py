from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypeVar

from torchtyping import TensorType

T = TypeVar("T")
T_co = TypeVar("T_co")
T_contra = TypeVar("T_contra")
U = TypeVar("U")
U_co = TypeVar("U_co")
U_contra = TypeVar("U_contra")
V = TypeVar("V")
V_co = TypeVar("V_co")
V_contra = TypeVar("V_contra")


class Task(ABC):
    def on_begin(self) -> None:
        pass

    def on_update(self, step: int) -> None:
        pass

    def on_end(self) -> None:
        pass

    @abstractmethod
    def priority(self) -> int:
        ...


class Stoppable(ABC):
    @abstractmethod
    def stop(self, step: int) -> bool:
        ...


class Playable(ABC, Generic[T_co, U_co]):
    @abstractmethod
    def play(self, input: T_co, step: int | None = None) -> U_co:
        ...


class Loggable(ABC, Generic[T_contra]):
    @abstractmethod
    def log(self, input: T_contra, step: int | None = None) -> None:
        ...


class Language(ABC):
    @abstractmethod
    def concept(self) -> TensorType[..., int]:
        ...

    @abstractmethod
    def messages(self) -> Mapping[str, TensorType[..., int]]:
        ...
