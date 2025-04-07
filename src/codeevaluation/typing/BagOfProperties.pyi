from typing import TypeVarTuple, TypeVar, Protocol, Union

import polars

T = TypeVar("T", contravariant=True)

class BoPProtocol(Protocol[T]):
    def _func(self, x: T) -> None:
        pass

Ts = TypeVarTuple("Ts")
type BoP[*Ts] = BoPProtocol[Union[*Ts]]

S = TypeVar("S", covariant=True)

class _Polars[S](polars.DataFrame):
    def _func(self) -> S:
        pass

Ls = TypeVarTuple("Ls")
type PolarsBoP[*Ls] = _Polars[BoP[*Ls]]
