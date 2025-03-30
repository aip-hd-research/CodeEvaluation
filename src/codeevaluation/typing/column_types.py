from typing import Type

from codeevaluation.typing.BagOfProperties import BoPColumn

# Universal usage
ID: Type[BoPColumn] = BoPColumn.create("id", str)

# For D test data
DTranslations: Type[BoPColumn] = BoPColumn.create("d_translations", str)
DWithParams: Type[BoPColumn] = BoPColumn.create("d_with_params", str)
DExecutable: Type[BoPColumn] = BoPColumn.create("d_executable", str)

# For execution results
Success: Type[BoPColumn] = BoPColumn.create("success", bool)
Error: Type[BoPColumn] = BoPColumn.create("error", str)
