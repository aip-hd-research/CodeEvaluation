import os
import json
import shutil
from tqdm import tqdm
import subprocess
from typing import Tuple
from omegaconf import DictConfig
import tempfile
import polars as pl

from codeevaluation.typing.BagOfProperties import (
    BagOfProperties,
    BagOfPropertiesFactory,
)
from codeevaluation.typing.types import (
    id,
    d_executable,
    success,
    error,
    d_translations,
    d_with_params,
)


class CompilationError(Exception):
    pass


class TestRuntimeError(Exception):
    pass


def fill_d_functions_into_tests(
    cfg: DictConfig,
    dTestCodeWithoutFunctions: BagOfProperties[id, d_with_params],
    dFunctions: BagOfProperties[id, d_translations],
) -> BagOfProperties[id, d_with_params, d_translations, d_executable]:
    dCodeData: BagOfProperties[
        id, d_with_params, d_translations
    ] = dTestCodeWithoutFunctions.join(dFunctions)

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
    )

    return dCodeExecutableData


def execute_d_tests(
    cfg: DictConfig,
    file_contents: BagOfProperties[id, d_executable],
) -> BagOfProperties[id, success, error]:
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_setup(file_contents, tmpdir)
        return execute_d_tests_from_workspace(cfg, tmpdir)


def workspace_setup(
    file_contents: BagOfProperties[id, d_executable],
    workspace_dir: str,
) -> None:
    # clear workspace of old junk
    if os.path.isdir(workspace_dir):
        shutil.rmtree(workspace_dir, ignore_errors=True)
    os.makedirs(workspace_dir)

    # paths to test
    code_under_test_path = os.path.join(workspace_dir, "code_under_test")
    os.makedirs(code_under_test_path)
    for row in file_contents.df.iter_rows(named=True):
        with open(
            os.path.join(code_under_test_path, f"{row['id']}.d"), "w", encoding="utf8"
        ) as f:
            f.write(row["d_executable"])


def execute_d_tests_from_workspace(
    cfg: DictConfig,
    workspace_dir: str,
) -> BagOfProperties[id, success, error]:
    sample_dir = os.path.join(workspace_dir, "code_under_test")
    files_to_test = os.listdir(sample_dir)
    run_dir = os.path.join(workspace_dir, "run_dir")
    results_list = []
    for file_name in tqdm(files_to_test):
        full_path = os.path.join(sample_dir, file_name)
        code_id = file_name.removesuffix(".d")
        os.makedirs(run_dir, exist_ok=True)
        try:
            correct, total = execute_single_test_from_path(cfg, full_path, run_dir)
        except Exception as e:
            results_list.append(
                {
                    "id": code_id,
                    "success": False,
                    "error": repr(e),
                }
            )
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        results_list.append(
            {
                "id": code_id,
                "success": correct == total,
                "error": "Correctness" if correct != total else "No",
            }
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    results_path = os.path.join(workspace_dir, "results.json")
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results_list, f)
    make_accessible(results_path)

    return BagOfPropertiesFactory[id, success, error].from_dicts(results_list)


def execute_single_test_from_path(
    cfg: DictConfig,
    path: str,
    run_dir: str,
) -> Tuple[int, int]:
    file = os.path.split(path)[-1]
    source_code_file = os.path.join(run_dir, file)
    bin_file = os.path.join(run_dir, file.split(".")[-1])
    shutil.copyfile(path, source_code_file)
    build_program(cfg, source_code_file, bin_file)
    return run_program(cfg, bin_file)


def make_accessible(path: str) -> None:
    os.system(f"chmod 0777 {path}")


def build_program(
    cfg: DictConfig,
    path: str,
    bin_path: str,
) -> None:
    result = subprocess.run(
        f"ldc2 -of={bin_path} {path}",
        shell=True,
        text=True,
        capture_output=True,
        timeout=cfg.COMPILATION_TIMEOUT_D_EXECUTION,
    )
    if len(result.stderr) > 0:
        raise CompilationError(result.stderr)


def run_program(
    cfg: DictConfig,
    path: str,
) -> Tuple[int, int]:
    os.system(f"chmod +x {path}")
    result = subprocess.run(
        path, capture_output=True, text=True, timeout=cfg.RUN_TIMEOUT_D_EXECUTION
    )
    if len(result.stderr) > 0:
        raise TestRuntimeError(result.stderr)
    if "#Results:" not in result.stdout:
        raise TestRuntimeError(f"Result does not conform to pattern: '{result.stdout}'")
    correct, total = result.stdout.split("#Results:")[-1].split(", ")
    return int(correct), int(total)
