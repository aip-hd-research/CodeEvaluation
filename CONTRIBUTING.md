# Requirements
The only thing you need to start is a working installation of Conda.

# How To Start
Clone this repository and run `setup.sh` while at the root of the repository.

This will create a Conda environment called **CodeEvaluation**, verify that the build process works, and install the git-hooks.

To verify that everything is setup activate the environment and run:

```python -m pytest ../tests```

from inside the `src` directory.

If the tests pass, you are good to go.

# Remote Development
If you want to use a remote interpreter you need to run the `setup.sh` on the remote machine to build the environment.

You then point your IDE to that remote environment.

Since the pre-commits run locally you need to install them separately. For this install `pre-commit` via conda or pip.
Then navigate to the root of the repository on your local machine and run:
```pre-commit install```