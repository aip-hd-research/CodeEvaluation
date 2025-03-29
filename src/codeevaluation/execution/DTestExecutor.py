import os
import json
import shutil
from tqdm import tqdm
import subprocess

from codeevaluation.typing.BagOfProperties import BoP
from codeevaluation.typing.column_types import ID, DExecutable, Success, Error
from codeevaluation.config_variables import (
    RUN_TIMEOUT_D_EXECUTION,
    COMPILATION_TIMEOUT_D_EXECUTION,
)


class CompilationError(Exception):
    pass


class TestRuntimeError(Exception):
    pass


def execute_d_tests(
    file_contents: BoP[ID, DExecutable], workspace_dir
) -> BoP[ID, Success, Error]:
    workspace_setup(file_contents, workspace_dir)
    return execute_d_tests_from_workspace(workspace_dir)


def workspace_setup(file_contents: BoP[ID, DExecutable], workspace_dir):
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


def execute_d_tests_from_workspace(workspace_dir) -> BoP[ID, Success, Error]:
    sample_dir = os.path.join(workspace_dir, "code_under_test")
    files_to_test = os.listdir(sample_dir)
    run_dir = os.path.join(workspace_dir, "run_dir")
    results_list = []
    for file_name in tqdm(files_to_test):
        full_path = os.path.join(sample_dir, file_name)
        os.makedirs(run_dir, exist_ok=True)
        try:
            correct, total = execute_single_test_from_path(full_path, run_dir)
        except Exception as e:
            results_list.append(
                {
                    "id": file_name.removesuffix(".d"),
                    "success": False,
                    "error": str(e),
                }
            )
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        results_list.append(
            {
                "id": file_name.removesuffix(".d"),
                "success": correct == total,
                "error": "Correctness" if correct != total else "No",
            }
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    results_path = os.path.join(workspace_dir, "results.json")
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results_list, f)
    make_accessible(results_path)

    return BoP[ID, Success, Error].from_dicts(results_list)


def execute_single_test_from_path(path, run_dir):
    file = os.path.split(path)[-1]
    source_code_file = os.path.join(run_dir, file)
    bin_file = os.path.join(run_dir, file.split(".")[-1])
    shutil.copyfile(path, source_code_file)
    build_program(source_code_file, bin_file)
    return run_program(bin_file)


def make_accessible(path):
    os.system(f"chmod 0777 {path}")


def build_program(path, bin_path):
    result = subprocess.run(
        f"ldc2 -of={bin_path} {path}",
        shell=True,
        text=True,
        capture_output=True,
        timeout=COMPILATION_TIMEOUT_D_EXECUTION,
    )
    if len(result.stderr) > 0:
        raise CompilationError(result.stderr)


def run_program(path):
    os.system(f"chmod +x {path}")
    result = subprocess.run(
        path, capture_output=True, text=True, timeout=RUN_TIMEOUT_D_EXECUTION
    )
    if len(result.stderr) > 0:
        raise TestRuntimeError(result.stderr)
    if "#Results:" not in result.stdout:
        raise TestRuntimeError(f"Result does not conform to pattern: '{result.stdout}'")
    correct, total = result.stdout.split("#Results:")[-1].split(", ")
    correct = int(correct)
    total = int(total)
    return correct, total
