from codeevaluation.typing.BagOfProperties import BoPColumn


# Universal usage
class ID(BoPColumn):
    name = "id"
    datatype = str


# For D test data
class DTranslations(BoPColumn):
    name = "d_translations"
    datatype = str


class DWithParams(BoPColumn):
    name = "d_with_params"
    datatype = str


class DExecutable(BoPColumn):
    name = "d_executable"
    datatype = str


# For execution results
class Success(BoPColumn):
    name = "success"
    datatype = bool


class Error(BoPColumn):
    name = "error"
    datatype = str
