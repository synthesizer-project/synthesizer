To add content to the documentation:

1. Add your jupyter notebook to the `source` directory
2. Add your notebook to the relevant toctree. This is contained within a sphinx `.rst` file in each documentation source directory. The top level file is `source/index.rst`. If your file is in a subfolder, you need to update the `.rst` file in that directory. 

- If you're creating a new sub-directory of documentation, you will need to create a new `.rst` file in that directory, and update `source/index.rst` with the relevant path.
