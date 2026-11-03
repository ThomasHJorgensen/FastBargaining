# Setting up the Python environment

This file explains how to install the Python packages for this repository with uv, pip or conda, how to keep the dependency files up to date, and how to build the replication package. [README.md](README.md) describes the replication itself, including the C++ compiler and the hardware it needs.

## Dependency files

| File | Content | Used by |
| --- | --- | --- |
| `pyproject.toml` | Direct dependencies and the supported Python versions (3.13) | uv |
| `uv.lock` | Exact versions of all packages, including indirect dependencies | uv |
| `.python-version` | The Python version used for the paper (3.13.13) | uv |
| `requirements.txt` | The same exact versions, exported from `uv.lock` | pip, conda |
| `environment.yml` | Python 3.13 and pip from conda-forge, then `requirements.txt` | conda |

All three options install the same package versions. The packages are the model dependencies (EconModel, consav, NumPy, Numba, SciPy, matplotlib, seaborn) and the tools that run the notebooks (papermill and ipykernel).

EconModel is installed from a pinned commit on GitHub, so **git** must be installed for all three options.

The code was last run with Python 3.13.13. The commands below are for the Windows command prompt. On Linux and macOS, activate a virtual environment with `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

## Option A: uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run in the root of the repository:

```bat
uv sync --frozen
.venv\Scripts\activate
```

`uv sync` creates the virtual environment `.venv` with the Python version in `.python-version`, which uv downloads if it is not installed. It then installs the packages in `uv.lock`, including the `dev` group (papermill and ipykernel). `--frozen` installs exactly the versions in `uv.lock` without updating it. Without `--frozen`, `uv sync` first updates `uv.lock` if `pyproject.toml` has changed. The commands in README.md (`uv python install 3.13.13` and `uv sync --frozen --python 3.13.13`) give the same environment.

Instead of activating the environment, you can also prefix commands with `uv run`, for example `uv run python -m papermill run_all.ipynb run_all.ipynb`.

## Option B: pip and venv

Install Python 3.13 from [python.org](https://www.python.org/downloads/) (the paper used 3.13.13) and run in the root of the repository:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`py -3.13` uses the Python launcher for Windows, which comes with the python.org installer, to select Python 3.13 when several Python versions are installed. On Linux and macOS, use `python3.13 -m venv .venv`. The environment is created in `.venv`, the same folder as with uv, so the commands in README.md work unchanged.

## Option C: conda

Install [Miniforge](https://github.com/conda-forge/miniforge) or another conda distribution and run in the root of the repository:

```bat
conda env create -f environment.yml
conda activate fastbargaining
```

conda installs Python 3.13 and pip, and pip then installs the packages in `requirements.txt`. conda-forge provides the newest Python 3.13 release, which can be newer than 3.13.13. The package versions are the same as with uv and pip. The environment is stored in the conda installation, not in `.venv`. To update an existing environment after `requirements.txt` has changed:

```bat
conda env update -n fastbargaining -f environment.yml --prune
```

## Running the replication

Activate the environment (`.venv\Scripts\activate` for options A and B, `conda activate fastbargaining` for option C) and run the master notebook from the root of the repository:

```bat
python -m papermill run_all.ipynb run_all.ipynb
```

All notebooks use the kernel `python3`, which papermill starts with the Python of the active environment. With options A and B, `.venv\Scripts\python -m papermill run_all.ipynb run_all.ipynb` works without activating the environment.

## Using the environment in Jupyter

In VS Code, open a notebook and select the interpreter in `.venv` (options A and B) or the `fastbargaining` conda environment (option C) as the kernel. ipykernel is already installed in the environment.

For JupyterLab or another Jupyter installation outside the environment, register the environment as a kernel once, with the environment activated:

```bat
python -m ipykernel install --user --name fastbargaining
```

## Maintaining the dependency files

`uv.lock` is the reference. Change the dependencies with uv, then export `requirements.txt` so that pip and conda get the same versions:

```bat
uv add <package>                      # add a model dependency
uv add --dev <package>                # add a tool to the dev group
uv lock --upgrade-package <package>   # upgrade one package
uv lock --upgrade                     # upgrade all packages
uv export --no-hashes -o requirements.txt
```

`uv add` updates `pyproject.toml` and `uv.lock` and installs the package in `.venv`. After editing `pyproject.toml` by hand, run `uv lock` to update the lock file.

To use a newer version of EconModel, change the commit hash `rev` under `[tool.uv.sources]` in `pyproject.toml` and run `uv lock`.

To use another Python 3.13 release, run `uv python pin 3.13.<patch>`, which updates `.python-version`. To move to another minor version such as 3.14, also change `requires-python` in `pyproject.toml` and `python=3.13` in `environment.yml`, then run `uv lock` and export `requirements.txt` again. Check first that the locked packages, in particular numba and llvmlite, provide wheels for that version.

`environment.yml` reads `requirements.txt`, so it only needs to be edited when the minor Python version changes.

## Building the replication package

`make_replication_package.py` in the root of the repository builds the zip file that is submitted to the journal. The script is a tool for the authors. It is in the GitHub repository but not in the replication package.

### Requirements

- **Windows with Microsoft Edge.** Edge prints README.pdf. The script looks for it in its default installation folders.
- **pandoc**, which converts README.md to HTML. It is in the `build` dependency group, which a plain `uv sync` does not install. Install it with

  ```bat
  uv sync --frozen --group build
  ```

  Run `uv sync` with `--group build` every time, because a sync without it removes pandoc again. If pandoc is not installed in `.venv`, the script uses a pandoc on the PATH, in pandoc's own installation folder or in Anaconda.
- **git.** The package consists of the files tracked by git.

### Steps

1. Run `run_all.ipynb` (see [Running the replication](#running-the-replication)), update the run times in README.md and check the rest of README.md.
2. Commit all changes, including the outputs. The script refuses to run if tracked files have uncommitted changes. Untracked files are not included in the package, so check `git status` for new output files and add them.
3. Run from the root of the repository:

   ```bat
   .venv\Scripts\python make_replication_package.py
   ```

   or, with uv:

   ```bat
   uv run --frozen --group build python make_replication_package.py
   ```

   `uv run` syncs `.venv` before it runs the script. `--group build` keeps pandoc installed, and `--frozen` keeps `uv.lock` unchanged.

4. Check the printed warnings and open `dist\Replication package.zip` to look through its contents, in particular README.pdf.
5. Submit `dist\Replication package.zip`. Commit the zip file and README.pdf.

To build a test version from a working tree with uncommitted changes, add `--allow-dirty`. The script then prints the uncommitted changes as a warning.

### What the script does

1. **Checks.** It stops if tracked files have uncommitted changes (unless `--allow-dirty` is passed). It warns if any exhibit of the paper listed in `OUTPUT_FILES` is missing.
2. **README.pdf.** pandoc converts README.md to HTML with a small style sheet, and Edge prints the HTML to `README.pdf` in the root of the repository. The PDF is overwritten every time.
3. **Zip.** It zips all files tracked by git, plus the files in `EXTRA_FILES`, minus the files and folders in `EXCLUDE` (`Archive/`, `dist/`, `.gitignore` and the script itself), to `dist\Replication package.zip`, with `Replication package` as the top folder in the zip file. It stops if a listed file does not exist. It prints the number of files and the size, and warns if the zip file is larger than 100 MB, the upload limit of Editorial Express.

Each run replaces `dist\Replication package.zip`. The zip file is tracked by git, so commit it after a build. Other files in `dist\` are ignored. Because `dist/` is excluded, a new package never contains an old one.

### Changing what goes into the package

- **New output file used in the paper:** add it to `OUTPUT_FILES`, so that the script warns if it is missing, and commit it.
- **File that should not be in the package:** add it to `EXCLUDE`. A path ending in `/` excludes a folder.
- **File that must be in the package even if git does not track it:** add it to `EXTRA_FILES`. For example, `README.pdf` is listed there because the script builds it before copying the files.
