Creating Grids
**************

Advanced users can create their own `synthesizer grids <../grids/grids>`_. These can be intrinsic grids of stellar emission, generated from stellar population synthesis models, or grids post-processed through photoionisation codes such as `cloudy <https://trac.nublado.org>`_.

The code for creating custom grids is contained in a separate repository, `grid-generation <https://github.com/synthesizer-project/grid-generation>`_.
You will need a working installation of synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. CLOUDY, python-FSPS). 

Grids should follow the naming convention where possible, see :ref:`grid-naming`.

Please see `Abundances <../abundances.ipynb>`_ for details on how to modify the chemical abundance pattern of gas, stars and dust using the `abundances`` object, and use this when running `cloudy`.


Running your own SPS grids 
--------------------------

Here we will show how to create an incident grid using synthesizer. These incident grids are often used as inputs to photoionisation codes like Cloudy, but are also useful in their own right for understanding the intrinsic properties of stellar populations.

Firstly, choose the grid you want to create, e.g. BC03, maraston05, or FSPS, and find the corresponding python script to install it within the `grid-generation` repository.
To create the grid, you need to specify where you want to place the raw data files from the model (`input_dir`), and where you would like the grid file to be created (`grid_dir`), e.g.

.. code-block:: bash

   python install_bc03.py --input_dir /home/dir/data/synthesizer_data/input_files --grid_dir /home/dir/data/synthesizer_data/grids

Some of the scripts to create grids have special requirements. For example, to create the BC03-2016 grid you need a working fortran compiler to convert the binary files into ascii, and you can check this is available by running `python which gfortran` at the command line.

Many of the scripts have the ability to download the original model data files by adding the command `--download`.
Unfortunately, the data for BPASS needs to be downloaded separately from the `BPASS website <https://bpass.auckland.ac.nz/index.html>`_.
To create the FSPS grid, the `python-fsps` package needs to be installed; details of how to do this can be found `here <https://dfm.io/python-fsps/current/installation/>`_. 

After creating a grid, there is also the option of creating a grid of a reduced size. For example, you can restrict the maximum age of the grid:  

.. code-block:: bash
   
   python create_reduced_grid.py -grid_dir /home/dir/data/synthesizer_data/grids -original_grid maraston13_kroupa -max_age 7
 

where here the maximum age was set to :math:`10^7` years.

Running a grid through Cloudy
-----------------------------

Here we will now show how to create input files for the photoionisation code Cloudy. Details on Cloudy, and how to install it, can be found on the `Cloudy website <https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home>`_.

Within `synthesizer_grids/cloudy/params` are a variety of parameter files that can be used to configure Cloudy, such as the ionisation parameter and hydrogen density. To use our standard approach, where we allow the ionisation parameter to vary with the input ionizing source, normalised to some reference value, the `c23.01-sps.yaml` parameter file is the most appropriate. Alternatively, `c23.01-sps-fixed.yaml` can be used for fixed ionisation parameters.

To create input files with varying parameter values, we can do something like this: 

.. code-block:: bash
   
   python create_cloudy_input_grid.py -grid_dir /home/dir/data/synthesizer_data/grids -cloudy_dir /home/dir/data/synthesizer_data/cloudy -incident_grid maraston11_kroupa -cloudy_params c23.01-sps -cloudy_params_addition test_suite/ionisation_parameter -machine sciama -verbose True 


Then, using the method of your choice, you can run the created input files through Cloudy. Within `grid-generation` are example scripts showing how to run these using different HPC systems.

Once these have been run through Cloudy, we can use the outputs from Cloudy to create a new grid, containing the post-processed spectra and line emission:

.. code-block:: bash
   
   python create_synthesizer_grid.py -grid_dir /home/dir/data/synthesizer_data/grids -cloudy_dir /home/dir/data/synthesizer_data/cloudy -incident_grid maraston11 -cloudy_params c23.01-sps-fixed-hydrogen_density
