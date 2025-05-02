"""A test suite for the transformers module."""

import numpy as np

from synthesizer.emission_models import IntrinsicEmission


def test_fesc_model_level(
    test_grid,
    random_part_stars,
):
    """Test setting fesc on the emission model.

    This test ensures that the fesc set during instantiation of a model
    is set correctly. We only do this for the IntrinsicEmission model here
    since it encompasses all the models that relate to fesc with some parent
    models derived from it.
    """
    # Get the spectra with fesc=0
    model = IntrinsicEmission(test_grid, fesc=0)
    spec_fesc0 = random_part_stars.get_spectra(model)
    random_part_stars.clear_all_emissions()

    # Get the spectra with fesc=1
    model = IntrinsicEmission(test_grid, fesc=1)
    spec_fesc1 = random_part_stars.get_spectra(model)

    # Ensure the two differ
    assert not np.all(spec_fesc0.lnu == spec_fesc1.lnu), (
        "The spectra are the same with different fesc values"
    )


def test_fesc_override(
    test_grid,
    random_part_stars,
    nebular_emission_model,
    reprocessed_emission_model,
    intrinsic_emission_model,
    pacman_emission_model,
):
    """Test the get_spectra fesc override in the emission model.

    This test ensures that the fesc override on get_spectra methods is working
    correctly. Unlike above, we test all the models here to ensure the
    override is correctly passed through different levels of complexity.
    """
    # For this test we need a singular tau_V value (we're only doing integrated
    # spectra)
    random_part_stars.tau_v = 0.1

    # Loop over the models and test
    for model in [
        nebular_emission_model,
        reprocessed_emission_model,
        intrinsic_emission_model,
        pacman_emission_model,
    ]:
        # Get the spectra with fesc=0
        spec_fesc0 = random_part_stars.get_spectra(
            model,
            fesc=0,
        )
        random_part_stars.clear_all_emissions()

        # Get the spectra with fesc=1
        spec_fesc1 = random_part_stars.get_spectra(
            model,
            fesc=1,
        )

        # Ensure the two differ
        assert not np.all(spec_fesc0.lnu == spec_fesc1.lnu), (
            f"[{model.__class__.__name__}]: The spectra "
            "are the same with different fesc values"
        )
