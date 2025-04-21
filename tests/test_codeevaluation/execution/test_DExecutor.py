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
from codeevaluation.execution.DExecutor import (
    execute_d_tests,
    fill_d_functions_into_tests,
)
from omegaconf import OmegaConf


def test_d_evaluation() -> None:
    # Temporary, use hydra once it works
    cfg_dict = {
        "REPLACEMENT_MARKER": {
            "d": "//TOFILL//",
        },
        "COMPILATION_TIMEOUT_D_EXECUTION": 30,
        "RUN_TIMEOUT_D_EXECUTION": 15,
        "workspace_dir": "workspace",
    }

    cfg = OmegaConf.create(cfg_dict)
    ###
    dWithParamsData: BagOfProperties[id, d_with_params] = BagOfPropertiesFactory[
        id, d_with_params
    ].load_from_huggingface("AIP-Heidelberg/test_code_d_with_params")
    dTranslationsData: BagOfProperties[id, d_translations] = BagOfPropertiesFactory[
        id, d_translations
    ].load_from_huggingface("AIP-Heidelberg/test_code_d_translations")

    assert dWithParamsData.df.shape == (600, 2)
    assert dTranslationsData.df.shape == (600, 2)

    dCodeExecutableData = fill_d_functions_into_tests(
        cfg, dWithParamsData, dTranslationsData
    )
    dCodeExecutableData.df = dCodeExecutableData.df.head(10)

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
