from codeevaluation.typing.BagOfProperties import BoPColumn

# Universal usage
ID = BoPColumn.create("id", str)

# For D test data
DTranslations = BoPColumn.create("d_translations", str)
DWithParams = BoPColumn.create("d_with_params", str)
DExecutable = BoPColumn.create("d_executable", str)

# For execution results
Success = BoPColumn.create("success", bool)
Error = BoPColumn.create("error", str)
