from collections.abc import Callable
from typing import TypeVar, Protocol, Union, Any

from typing_extensions import TypeVarTuple

from codeevaluation.typing.BagOfProperties import PolarsBoP


class Test:
    def test_proto(
        self,
        proto_factory: Callable[[], PolarsBoP[str]],
        proto_consumer: Callable[[PolarsBoP[str, int]], None],
        df: PolarsBoP[str],
    ) -> None:
        print(df.columns)  # typechecks as polars.DataFrame
        proto_consumer(proto_factory())  # has desired set properties


if __name__ == "__main__":
    test = Test()
