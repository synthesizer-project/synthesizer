"""A suite of tests for the emission model operations."""

import numpy as np


def test_single_star_extraction(
    single_star_particle,
    single_star_parametric,
    test_grid,
    incident_emission_model,
    transmitted_emission_model,
    nebular_emission_model,
    reprocessed_emission_model,
):
    """
    Test extraciton of a single star's emission.

    This will use and compare a single star for a particle Stars object and a
    single SFZH bin for a parametric Stars object. These two descriptions
    should be equivalent.
    """
    # First ensure the sfzh's are equivalent
    single_star_particle.get_sfzh(
        test_grid.log10ages,
        test_grid.log10metallicity,
    )
    assert np.isclose(np.sum(single_star_particle.sfzh.sfzh), 1.0), (
        "The unit particle SFZH does not sum to 1"
        f" (sum={np.sum(single_star_particle.sfzh.sfzh.sum())})"
    )
    assert np.isclose(np.sum(single_star_parametric.sfzh), 1.0), (
        "The unit parametric SFZH does not sum to 1"
        f" (sum={np.sum(single_star_parametric.sfzh)})"
    )
    assert np.allclose(
        single_star_particle.sfzh.sfzh,
        single_star_parametric.sfzh,
    ), (
        f"The SFZH's are not equivalent (non-zero elements: "
        f"particle={np.where(single_star_particle.sfzh.sfzh > 0)}, "
        f"parametric={np.where(single_star_parametric.sfzh >0)})"
    )

    # Ok, we know the SFZH's are equivalent, let's now get the spectra
    # and compare them for a range of emission model complexities.

    # Loop over the emission models
    for model in [
        incident_emission_model,
        transmitted_emission_model,
        nebular_emission_model,
        reprocessed_emission_model,
    ]:
        part_sed = single_star_particle.get_spectra(model)
        param_sed = single_star_parametric.get_spectra(model)
        assert np.allclose(part_sed.shape, param_sed.shape), (
            f"[{model.__class__.__name__}]: "
            f"The SED shapes are not equivalent (particle={part_sed.shape}, "
            f"parametric={param_sed.shape})"
        )
        resi = np.sum(part_sed.lnu) - np.sum(param_sed.lnu)
        assert np.allclose(
            part_sed.lnu,
            param_sed.lnu,
        ), (
            f"[{model.__class__.__name__}]: "
            "The SEDs are not equivalent (part_sed.sum - param_sed.sum = "
            f"{np.sum(part_sed.lnu)} - {np.sum(param_sed.lnu)} = {resi}, "
        )
