import numpy as np

available_scalings = ['Dopita2006']


class Dopita2006:

    """Scaling functions for Nitrogen."""

    ads = "https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract"
    doi = "10.1086/508261"
    available_elements = ['N', 'C']

    def N(metallicity):
        """

        Args:
            metallicity (float)
                The metallicity (mass fraction in metals)

        Returns:
            abundance (float)
                The logarithmic abundance relative to Hydrogen.

        """

        # the metallicity scaled to the Dopita (2006) value
        dopita_solar_metallicity = 0.016
        scaled_metallicity = metallicity / dopita_solar_metallicity

        abundance = np.log10(
            1.1e-5 * scaled_metallicity
            + 4.9e-5 * (scaled_metallicity) ** 2
        )

        return abundance

    def C(metallicity):
        """
        Scaling functions for Carbon.

        Args:
            metallicity (float)
                The metallicity (mass fraction in metals)

        Returns:
            abundance (float)
                The logarithmic abundance relative to Hydrogen.

        """

        # the metallicity scaled to the Dopita (2006) value
        dopita_solar_metallicity = 0.016
        scaled_metallicity = metallicity / dopita_solar_metallicity

        abundance = np.log10(
            6e-5 * scaled_metallicity + 2e-4 * (scaled_metallicity) ** 2
        )

        return abundance
