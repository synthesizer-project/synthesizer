AGN Grids
=========

.. _grid-naming:

Grid naming
-----------

The naming of AGN grids broadly follows this specification::

    {agn_model}-{agn_version}_{agn_variant}_{photoionisation_code}-{photoionisation_code_version}-{blr/nlr}_{photoionisation_parameters}

For example::

    relagn_fixed_rad_efficiency_0.1_cloudy-c23.01-nlr

specifies that the grid is constructed using using the `RELAGN` model with a fixed radiative efficiency of 0.1, and the photoionisation modelling is done using `cloudy` version c23.01, for the narrow line region (NLR) with the default photoionisation parameters for the NLR.

Photoionisation modelling
-------------------------

All the photoionisation modelling in synthesizer currently uses the `cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionisation code. It can simulate a range of gas and ionisation conditions, with specific geometries. Similar to the SPS grids, our default AGN grids make certain choices to restrict the range of assumptions. For both the broad line region (BLR) and narrow line region (NLR) grids, we specify the geometry to be `plane-parallel` (this was `spherical` for SPS grids). For the NLR, we assume that the cloud is ionisation-bound and hence case-B recombination holds, and we choose the hydrogen densities in the range: :math:`10^{2} \leq n_{\rm H} \leq 10^{6}`. For the BLR, we assume the cloud is density-bound, and we choose the hydrogen column densities in the range :math:`10^{21} \leq N_{\rm H}/{\rm cm}^{2} \leq 10^{26}` with hydrogen densities: :math:`10^{8} \leq n_{\rm H}/{\rm cm}^{3} \leq 10^{12}`. We remind the user that when creating their own grids, some of these parameter space runs can be computationally expensive, especially for the BLR grids, and so it is worth considering carefully the parameter space to run over.
By default we do not include any dust grains in the BLR photoionisation modelling.

The photonisation modelling parameters are included in the grid file attributes. Pre-computed grids are available at `Box <https://sussex.box.com/v/SynthesizerGrids>`_; AGN models without any photoionisation modelling are stored under `incident`, and for our default photoionisation assumptions under `photoionised`.

Common variants
---------------

* `resolution:0.1` outputs the spectra at 10x higher resolution than the `cloudy` default. Useful for looking at various absorption line indices and comparison to high-resolution spectra. Usually suffixed to the grid name with `_0.1res` (e.g. `relagn_fixed_rad_efficiency_0.1_cloudy-c23.01-nlr_0.1res.hdf5`). 
* `var_rad_efficiency` assumes a variable radiative efficiency for the AGN. This can be the case for models that parametrise the black hole spin, which affects the radiative efficiency of the accretion disk, and hence the shape of the incident spectrum. The variable radiative efficiency is calculated self-consistently in the `RELAGN` model. This is not a parameter that is one of the axes values, but can be computed from the spin.
* `reduced` assumes a reduced set of black hole or photoionisation parameters, since the grid can become unwieldy due to the number of parameters.

Grid list
~~~~~~~~~

Currently we only have grids for the `RELAGN`, `AGNSED` and `QSOSED` model, but we plan to add more AGN grids (such as the `cloudy` in-built model, `Feltre et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3354F/>`_) in the future. Below are examples of the pre-computed grids available:

.. collapse:: RELAGN

    - BLR grids
        + relagn_fixed_rad_efficiency_0.1_cloudy-c23.01-blr_0.055res

    - NLR grids
        + relagn_fixed_rad_efficiency_0.1_cloudy-c23.01-nlr
        + relagn_fixed_rad_efficiency_0.1_reduced_cloudy-c23.01-nlr_0.055res
        + relagn_variable_rad_efficiency_reduced_cloudy-c23.01-nlr
