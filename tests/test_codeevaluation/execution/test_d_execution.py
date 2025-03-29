from codeevaluation.typing.BagOfProperties import BoP
from codeevaluation.typing.column_types import (
    ID,
    DWithParams,
    DTranslations,
    DExecutable,
    Success,
    Error,
)
from codeevaluation.execution.DTestExecutor import execute_d_tests
from codeevaluation.config_variables import REPLACEMENT_MARKER

import polars as pl


def test_d_evaluation():
    dWithParamsData: BoP[ID, DWithParams] = BoP[ID, DWithParams].load_from_huggingface(
        "AIP-Heidelberg/test_code_d_with_params"
    )
    dTranslationsData: BoP[ID, DTranslations] = BoP[
        ID, DTranslations
    ].load_from_huggingface("AIP-Heidelberg/test_code_d_translations")

    assert dWithParamsData.df.shape == (600, 2)
    assert dTranslationsData.df.shape == (600, 2)

    dCodeData = dWithParamsData.join(dTranslationsData)

    assert isinstance(dCodeData, BoP[ID, DWithParams, DTranslations])
    assert dCodeData.df.shape == (600, 3)

    # Check if the REPLACEMENT_MARKER is in any row of `d_with_params`. If not, raise an error.
    missing_to_fill = dCodeData.df.filter(
        ~pl.col("d_with_params").str.contains(REPLACEMENT_MARKER["d"])
    ).height

    if missing_to_fill > 0:
        raise ValueError(
            "Some rows in 'd_with_params' are missing the REPLACEMENT_MARKER."
        )

    dCodeExecutableData = BoP[*dCodeData._type_params, DExecutable]()

    # Replace REPLACEMENT_MARKER with the corresponding value from `d_translations`
    dCodeExecutableData.df = dCodeData.df.with_columns(
        (
            pl.col("d_with_params")
            .str.replace_all(REPLACEMENT_MARKER["d"], pl.col("d_translations"))
            .alias("d_executable")
        )
    ).head(10)

    assert dCodeExecutableData.df.shape == (10, 4)
    assert isinstance(
        dCodeExecutableData, BoP[ID, DWithParams, DTranslations, DExecutable]
    )

    results: BoP[ID, Success, Error] = execute_d_tests(dCodeExecutableData, "workspace")

    assert results.df.shape == (10, 3)
