"""
"""
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.units import Quantity


class PhotometryCollection:
    """ """

    # Define quantities (there has to be one for rest and observer frame)
    rest_photometry = Quantity()
    obs_photometry = Quantity()

    def __init__(self, filters, rest_frame, **kwargs):
        """
        Instantiate the photometry collection.

        To enable quantities a PhotometryCollection will store the data
        as arrays but enable access via dictionary syntax.

        Args:
            filters (FilterCollection)
                The FilterCollection used to produce the photometry.
            rest_frame (bool)
                A flag for whether the photometry is rest frame luminosity or
                observer frame flux.
            kwargs (dict)
                A dictionary of keyword arguments containing all the photometry
                of the form {"filter_code": photometry}.
        """

        # Store the filter collection
        self.filters = filters

        # Get the filter codes
        self.filter_codes = list(kwargs.keys())

        # Get the photometry
        photometry = np.array(list(kwargs.values()))

        # Put the photometry in the right place (we need to draw a distinction
        # between rest and observer frame for units)
        if rest_frame:
            self.rest_photometry = photometry
            self.obs_photometry = None
        else:
            self.obs_photometry = photometry
            self.rest_photometry = None

        # Construct a dict for the look up, importantly we here store
        # the values in photometry not _photometry meaning they have units.
        self._look_up = {
            f: val
            for f, val in zip(
                self.filter_codes,
                self.rest_photometry if rest_frame else self.obs_photometry,
            )
        }

        # Store the rest frame flag for convinience
        self.rest_frame = rest_frame

    def __getitem__(self, filter_code):
        """
        Enable dictionary key look up syntax to extract specific photometry,
        e.g. Galaxy.broadband_luminosities["JWST/NIRCam.F150W"].

        NOTE: this will always return photometry with units. Unitless
        photometry is accessible in array form via self._rest_photometry
        or self._obs_photometry based on what frame is desired. For
        internal use this should be fine and the UI (where this method
        would be used) should always return with units.

        Args:
            filter_code (str)
                The filter code of the desired photometry.
        """

        # Perform the look up
        return self._look_up[filter_code]

    def plot_photometry(
        self,
        fig=None,
        ax=None,
        show=False,
        ylimits=(),
        xlimits=(),
        marker="+",
        figsize=(3.5, 5),
    ):
        """
        Plot the photometry alongside the filter curves.
        """
        # If we don't already have a figure, make one
        if fig is None:
            # Set up the figure
            fig = plt.figure(figsize=figsize)

            # Define the axes geometry
            left = 0.15
            height = 0.6
            bottom = 0.1
            width = 0.8

            # Create the axes
            ax = fig.add_axes((left, bottom, width, height))

            # Set the scale to log log
            ax.loglog()

        # Add a filter axis
        filter_ax = ax.twinx()
        filter_ax.set_ylim(0, None)

        # PLot each filter curve
        for f in self.filters:
            filter_ax.plot(f.lam, f.t)

        # Plot the photometry
        for f, phot in zip(self.filters, self._photometry):
            pivwv = f.pivwv()
            fwhm = f.fwhm()
            ax.errorbar(pivwv, phot, marker=marker, xerr=fwhm, linestyle=None)

        # Set the x and y lims
        if len(xlimits) > 0:
            ax.set_xlim(*xlimits)
        if len(ylimits) > 0:
            ax.set_ylim(*ylimits)

        return fig, ax
