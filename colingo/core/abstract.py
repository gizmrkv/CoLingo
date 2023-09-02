from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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


class Stoppable(ABC):
    @abstractmethod
    def stop(self, step: int) -> bool:
        ...


class Playable(ABC, Generic[T_co, U_co]):
    @abstractmethod
    def play(self, input: T_co, step: int | None = None) -> U_co:
        ...


class Computable(ABC, Generic[T_co, U_co, V_co]):
    @abstractmethod
    def compute(self, input: T_co, output: U_co, step: int | None = None) -> V_co:
        ...


class Loggable(ABC, Generic[T_co]):
    @abstractmethod
    def log(self, input: T_co) -> None:
        ...
