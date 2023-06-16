# Contributing to the Documentation
The synthesizer documentation is written in a combination of restructuredText and Jupyter notebooks. 
Adding content should be relatively simple, if you follow the instricutions below.

## Getting set up

First we're going to make some small changes to the git configuration to prevent excessive git diffs in the future when contributing changes to notebooks.

1. First, add the following lines to the end of the `.git/config` file at the root of the synthesizer repository

    [filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"

2. Then (if it does not already exist) create a file called `.gitattributes` in the root of the synthesizer repository, and add the following

    *.ipynb filter=strip-notebook-output


This will reset all instances of `execution_count` with `null`, and replace the `metatdata` tag with an empty dictionary, and prevent spurious git diffs to notebooks when they have been run multiple times.

    ...
    "execution_count": null,
    "metadata": {},
    ...

## Adding notebooks
To add jupyter notebooks to the documentation:

1. Add your jupyter notebook to the `source` directory. Make sure that you 'Restart Kernel and run all cells' to ensure that the notebook is producing up to date, consistent outputs.
2. Add your notebook to the relevant toctree. See below for an example toctree. Each toctree is contained within a sphinx `.rst` file in each documentation source directory. The top level file is `source/index.rst`. If your file is in a subfolder, you need to update the `.rst` file in that directory. 

- If you're creating a new sub-directory of documentation, you will need to create a new `.rst` file in that directory, and update `source/index.rst` with the relevant path.

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


