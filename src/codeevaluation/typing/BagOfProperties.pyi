from typing import TypeVarTuple, TypeVar, Protocol, Union

Ts = TypeVarTuple("Ts")

T = TypeVar("T", contravariant=True)

class BoPProtocol(Protocol[T]):
    def _func(self, x: T) -> None:
        pass

type BoP[*Ts] = BoPProtocol[Union[*Ts]]
