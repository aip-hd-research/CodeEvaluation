from typing import Type
from codeevaluation.typing.BagOfProperties import BagOfPropertiesFactory


class id:
    datatype: Type = int


class codeJava:
    datatype: Type = str


class Executor:
    def __init__(self):
        self.bop = BagOfPropertiesFactory[id, codeJava].new()


if __name__ == "__main__":
    print("blah")
