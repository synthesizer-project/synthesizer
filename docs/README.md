To add content to the documentation:

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


