from codeevaluation.lang_processors.d_processor import DProcessor
from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    BagOfPropertiesFactory,
)
from codeevaluation.typing.types import id, raw_d_translations, d_translations

import polars as pl

d_proc = DProcessor()


def extract_answer_functions(
    answers: BagOfProperties[id, raw_d_translations],
) -> BagOfProperties[id, d_translations]:
    functions = BagOfPropertiesFactory[id, d_translations].new()
    functions.df = answers.df.select(
        [
            pl.col("id"),
            pl.col("raw_d_translations")
            .map_elements(extract_answer_function)
            .alias("d_translations"),
        ]
    )
    return functions


def extract_answer_function(answer: str) -> str:
    code_start = answer.find("```d") + len("```d")
    code_end = len(answer)  # assume no end of code by default
    if answer.count("```") > 1:
        code_end = answer.find("```", code_start)
    code = answer[code_start:code_end]

    functions, class_functions = d_proc.extract_functions(code, tokenized=False)
    if len(functions) > 0:
        return functions[0]
    if len(class_functions) > 0:
        return class_functions[0]
    return code
