from typing import Any

from codeevaluation.typing.BagOfProperties import BoP, BoPColumn


class ID(BoPColumn):
    name = "id"
    datatype = str


class FunctionJava(BoPColumn):
    name = "function_java"
    datatype = str


class TestbedJava(BoPColumn):
    name = "testbed_java"
    datatype = str


class FunctionPython(BoPColumn):
    name = "function_python"
    datatype = str


class TestbedPython(BoPColumn):
    name = "testbed_python"
    datatype = str


class FunctionCpp(BoPColumn):
    name = "function_cpp"
    datatype = str


class TestbedCpp(BoPColumn):
    name = "testbed_cpp"
    datatype = str


def loadGeek4Geeks() -> Any:
    return BoP[
        ID,
        FunctionJava,
        TestbedJava,
        FunctionPython,
        TestbedPython,
        FunctionCpp,
        TestbedCpp,
    ].load_from_huggingface("AIP-Heidelberg/geeks4geeks_fixed")
