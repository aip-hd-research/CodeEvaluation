from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    BagOfPropertiesFactory,
    getTypes,
)
from codeevaluation.typing.types import (
    id,
    d_with_params,
    d_translations,
    d_executable,
    success,
    error,
)
from codeevaluation.execution.DExecutor import execute_d_tests

from hydra import initialize, compose
import polars as pl


def test_d_evaluation() -> None:
    with initialize(config_path="../../../conf"):
        cfg = compose(config_name="config")
    dWithParamsData: BagOfProperties[id, d_with_params] = BagOfPropertiesFactory[
        id, d_with_params
    ].load_from_huggingface("AIP-Heidelberg/test_code_d_with_params")
    dTranslationsData: BagOfProperties[id, d_translations] = BagOfPropertiesFactory[
        id, d_translations
    ].load_from_huggingface("AIP-Heidelberg/test_code_d_translations")

    assert dWithParamsData.df.shape == (600, 2)
    assert dTranslationsData.df.shape == (600, 2)

    dCodeData: BagOfProperties[
        id, d_with_params, d_translations
    ] = dWithParamsData.join(dTranslationsData)

    assert getTypes(dCodeData) == set((id, d_with_params, d_translations))

    assert dCodeData.df.shape == (600, 3)

    # Check if the REPLACEMENT_MARKER is in any row of `d_with_params`. If not, raise an error.
    missing_to_fill = dCodeData.df.filter(
        ~pl.col("d_with_params").str.contains(cfg.REPLACEMENT_MARKER.d)
    ).height

    if missing_to_fill > 0:
        raise ValueError(
            "Some rows in 'd_with_params' are missing the REPLACEMENT_MARKER."
        )

    dCodeExecutableData = BagOfPropertiesFactory[
        id, d_with_params, d_translations, d_executable
    ].new()

    # Replace REPLACEMENT_MARKER with the corresponding value from `d_translations`
    dCodeExecutableData.df = dCodeData.df.with_columns(
        (
            pl.col("d_with_params")
            .str.replace_all(cfg.REPLACEMENT_MARKER.d, pl.col("d_translations"))
            .alias("d_executable")
        )
    ).head(10)

    assert dCodeExecutableData.df.shape == (10, 4)
    assert getTypes(dCodeExecutableData) == set(
        (
            id,
            d_with_params,
            d_translations,
            d_executable,
        )
    )

    results: BagOfProperties[id, success, error] = execute_d_tests(
        cfg, dCodeExecutableData
    )

    assert results.df.shape == (10, 3)
