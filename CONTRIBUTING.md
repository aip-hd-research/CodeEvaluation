# Requirements
The only thing you need to start is a working installation of Conda.

# How To Start
Clone this repository and execute the script `setup.sh` while at the root of the repository (e.g. `bash -i ./setup.sh`).

This will create a Conda environment called **CodeEvaluation**, verify that the build process works, and install the git-hooks.

To verify that everything is setup activate the environment and run:

```python -m pytest ../tests```

from inside the `src` directory.

If the tests pass, you are good to go.

# Remote Development
If you want to use a remote interpreter you need to execute the `setup.sh` script on the remote machine to build the environment.

You then point your IDE to that remote environment.

Since the pre-commits run locally you need to install them separately. For this install `pre-commit` via conda or pip.
Then navigate to the root of the repository on your local machine and run:
```pre-commit install```

# Local Testing with VS Code

You need the following two files such that VS Code and pytests recognize the module structure:

- The `settings.json` inside the `.vscode` folder should have the following entries:
```json
{
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

- Create a `.env` file inside the root directory. Set the `PYTHONPATH` environment variable:
```
PYTHONPATH=src
```

Also make sure that the conda environment is selected as the python interpreter of VS Code (`Ctrl/cmd + Shift + P` and write `Python: Select Interpreter`)