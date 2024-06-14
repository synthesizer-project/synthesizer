# Contributing 

Please feel free to submit issues and pull requests to this repository. 
The GitHub workflow will automatically run [Ruff](https://github.com/astral-sh/ruff) on any contributions; builds that fail these tests will not be accepted. Further notes on code style are detailed below.

**Contents:**
- [Setting up your development environment](#setting-up-your-development-environment)
- [Using Ruff](#using-ruff)
- [Setting up pre-commit hooks](#setting-up-pre-commit-hooks)
- [Style guide](#style-guide)
- [Contributing to the Documentation](#contributing-to-the-documentation)
    - [Getting set up](#getting-set-up)
    - [Adding notebooks](#adding-notebooks)
    - [Adding example scripts](#adding-example-scripts)
    
## Settting up your development environment

If you are going to contribute to Synthesizer you might weant to install the extra dependancies we use for development. These can be installed using the `dev` dependancy set.

    pip install -e .[dev]
    
## Using Ruff

We use [Ruff](https://github.com/astral-sh/ruff) for both linting and formatting. Assuming you installed the development dependancies (if not you can install `ruff` with pip: `pip install ruff`), you can run the linting with `ruff check` and the formatting with `ruff format` (each followed by the files to consider).

The `ruff` configuration is defined in our `pyproject.toml` so there's no need to configure it yourself, we've made all the decisions for you (for better or worse). Any merge request will be checked with the `ruff` linter and must pass before being eligable to merge.

## Setting up pre-commit hooks

We also provide a pre-commit hook which will run on any files committed to the repo on any branch. If you plan to commit anything to Synthesizer it is highly recommended you install the pre-commit hook to make your life easier. This pre-commit hook will guard against files containing merge conflict strings, check case conflicts in file names, guard against the commiting of large files, santise jupyter notebooks (using `nb-clean`) and, most importantly, will run `ruff` in both linter and formatter mode.

This requires a small amount of set up on your part. To install the the pre-commit hook navigate to the root of the repo and invoke:
```
pip install ruff pre-commit nb-clean; pre-commit install
```

and you're done. Now every time you commit a file the pre commit hook will be run automatically.

If you would like to test whether it works you can run `pre-commit run --all-files` to run the pre-commit hook on the whole repo. You should see each stage complete without issue in a clean clone.


## Style guide
All new PRs should follow these guidelines. We adhere to the pep8 style guide, and as described above this is verified with `ruff`. We use the [Google docstring format](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).

Some specific examples of common style issues:
- Do not use capatilised single letters for attributes. For example, `T` could be transmission or temperature. Instead, one should write out the full word.
- Operators should be surrounded with whitespace.
- We use `get_` and/or `calculate_` nomenclature for methods that perform a calculation and return the result to the user.
- Variables should adhere to `snake_case` style while class names should be in `PascalCase`.
- Block comments should have their first letter capitalised, i.e.
```
# This is a block comment
x = y
```
- While inline comments should be proceeded by two whitespaces and start with a lowercase letter, i.e.
```
z = x * 2  # this is an inline comment
```
- Inheritance should use `Parent.__init__` instansiation of the parent class over `super()` for clarity.

## Contributing to the Documentation
The synthesizer documentation is written in a combination of restructuredText, Jupyter notebooks and Python scripts. 
Adding content should be relatively simple, if you follow the instructions below.

### Adding notebooks
To add jupyter notebooks to the documentation:

1. Add your jupyter notebook to the `source` directory. Make sure that you 'Restart Kernel and run all cells' to ensure that the notebook is producing up to date, consistent outputs.
2. Add your notebook to the relevant toctree. See below for an example toctree. Each toctree is contained within a sphinx `.rst` file in each documentation source directory. The top level file is `source/index.rst`. If your file is in a subfolder, you need to update the `.rst` file in that directory.

- If you're creating a new sub-directory of documentation, you will need to carry out a couple more steps:
1. Create a new `.rst` file in that directory
2. Update `source/index.rst` with the path to that `.rst` file
3. Add a line to the *pytest* section of `.github/workflows/python-app.yml` to add the notebooks to the testing suite. It should look something like this
  
        name: Test with pytest
          run: |
             pytest
             pytest --nbmake docs/source/*.ipynb
             pytest --nbmake docs/source/cosmo/*.ipynb
             pytest --nbmake docs/source/grids/*.ipynb
             pytest --nbmake docs/source/imaging/*.ipynb
             pytest --nbmake docs/source/parametric/*.ipynb
             pytest --nbmake docs/source/your_new_directory/*.ipynb

Example toctree:

    .. toctree::
       :maxdepth: 2
       :caption: Contents
    
       installation
       grids/grids
       parametric/parametric
       cosmo/cosmo
       imaging/imaging
       filters
       grid_generation

### Adding example scripts

The `examples/` top level directory contains a number of self contained example scripts (python, `.py`) for particular use cases that may not belong in the main documentation, but are still useful for many users. We use the [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) extension to build a gallery of our examples in the documentation.

**Important**: each script (`.py`) should have a top level docstring written in reST, with a header. Examples that do not will fail the automated build process. Further details are provided [here](https://sphinx-gallery.github.io/stable/syntax.html). For example:

    """
    "This" is my example-script
    ===========================

    This example doesn't do much, it just makes a simple plot
    """


Subfolders of examples should contain a `README.rst` with a section heading (please follow the template in other subfolders).

If an example is named `plot_*.py` then `sphinx-gallery` will attempt to run the script and use any images generated in the gallery thumbnail. Images should be generated using `plt.show()` and not saved to disk.

## Debugging C development

If you are writing C extensions for synthesizer you can include debugging checks and optionally activate them using the `WITH_DEBUGGING_CHECKS` preprocessor directive. To use this wrap the debugging code in an ifdef:

```
#ifdef WITH_DEBUGGING_CHECKS
debugging code...
#endif
```

To activate debugging checks simply install with `WITH_DEBUGGING_CHECKS=1 pip install .`.

It is also advisable to turn warnings into errors by including `-Werror` in the CFLAGS, however, the Python source code itself will fail with this turned on for some compilers because it does produce some warnings (observed with gcc).
