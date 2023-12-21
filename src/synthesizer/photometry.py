""" A module for working with photometry derived from an Sed.

This module contains a single class definition which acts as a container
for photometry data. It should never be directly instantiated, instead
internal methods that calculate photometry (e.g. Sed.get_broadband_luminosities)
return an instance of this class.
"""
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.units import Quantity


class PhotometryCollection:
    """
    Represents a collection of photometry values and provides unit
    association and plotting functionality.

    This is a utility class returned by functions else where. Although not
    an issue if it is this should never really be directly instantiated.

    Attributes:
        rest_photometry (Quantity):
            Quantity instance representing photometry data in the rest frame.
        obs_photometry (Quantity):
            Quantity instance representing photometry data in the
            observer frame.
        filters (FilterCollection):
            The FilterCollection used to produce the photometry.
        filter_codes (list):
            List of filter codes.
        _look_up (dict):
            A dictionary for easy access to photometry values using
            filter codes.
        rest_frame (bool):
            A flag indicating whether the photometry is in the rest frame
            (True) or observer frame (False).
    """

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
        e.g. Sed.broadband_luminosities["JWST/NIRCam.F150W"].

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

    def __str__(self):
        """
        Allow for a summary to be printed.

        Returns:
            str: A formatted string representation of the PhotometryCollection.
        """

        # Define the headers
        headers = [
            f"{f.filter_code} (\u03BB = {f.pivwv().value:.2e} {str(f.lam.units)})"
            for f in self.filters
        ]

        # Determine the width of each column
        widths = [
            max(
                len(header),
                len(str(format(phot.value, ".2e")) + " " + str(phot.units)),
            )
            + 2
            for header, phot in zip(headers, self._look_up.values())
        ]

        # How many characters make up the table?
        tot_width = sum(widths) + len(widths) + 1

        # Create the table header
        head = "|".join(
            f"{header.center(width)}" for header, width in zip(headers, widths)
        )
        sep = "|".join("-" * width for width in widths)

        # Create the photometry row
        phot = "|".join(
            f"{str(format(self[key].value, '.2e'))} "
            f"{str(self[key].units)}".center(width)
            for key, width in zip(self.filter_codes, widths)
        )

        # Create the centered title with "=" on either side
        if self.rest_frame:
            title = f"{'= REST FRAME PHOTOMETRY ='.center(tot_width, '=')}"
        else:
            title = f"{'= OBSERVED PHOTOMETRY ='.center(tot_width, '=')}"

        # Combine everything into the final table
        table = (
            f"{title}\n-{sep.replace('|', '-')}-\n|{head}|"
            f"\n|{sep}|\n|{phot}|\n-{sep.replace('|', '-')}-"
        )

        return table

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

        Args:
            fig (matplotlib.figure.Figure, optional):
                A pre-existing Matplotlib figure. If None, a new figure will
                be created.
            ax (matplotlib.axes._axes.Axes, optional):
                A pre-existing Matplotlib axes. If None, new axes will be
                created.
            show (bool, optional):
                If True, the plot will be displayed.
            ylimits (tuple, optional):
                Tuple specifying the y-axis limits for the plot.
            xlimits (tuple, optional):
                Tuple specifying the x-axis limits for the plot.
            marker (str, optional):
                Marker style for the photometry data points.
            figsize (tuple, optional):
                Tuple specifying the size of the figure.

        Returns:
            tuple:
                The Matplotlib figure and axes used for the plot.
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
            ax.semilogy()

            # Grid it... as all things should be
            ax.grid(True)

        # Add a filter axis
        filter_ax = ax.twinx()
        filter_ax.set_ylim(0, None)

        # PLot each filter curve
        max_t = 0
        for f in self.filters:
            filter_ax.plot(f.lam, f.t)
            if np.max(f.t) > max_t:
                max_t = np.max(f.t)

        # Get the photometry
        photometry = self.rest_photometry if self.rest_frame else self.obs_photometry

        # Plot the photometry
        for f, phot in zip(self.filters, photometry.value):
            pivwv = f.pivwv()
            fwhm = f.fwhm()
            ax.errorbar(
                pivwv,
                phot,
                marker=marker,
                xerr=fwhm,
                linestyle=None,
                capsize=3,
            )

        # Do we not have y limtis?
        if len(ylimits) == 0:
            max_phot = np.max(photometry)
            ylimits = (
                10 ** (np.log10(max_phot) - 5),
                10 ** (np.log10(max_phot) * 1.1),
            )

        # Do we not have x limits?
        if len(xlimits) == 0:
            # Define initial xlimits
            xlimits = [np.inf, -np.inf]

            # Loop over spectra and get the total required limits
            for f in self.filters:
                # Derive the x limits from data above the ylimits
                trans_mask = f.t > 0
                lams_above = f.lam[trans_mask]

                # Saftey skip if no values are above the limit
                if lams_above.size == 0:
                    continue

                # Derive the x limits
                x_low = 10 ** (np.log10(np.min(lams_above)) * 0.95)
                x_up = 10 ** (np.log10(np.max(lams_above)) * 1.05)

                # Update limits
                if x_low < xlimits[0]:
                    xlimits[0] = x_low
                if x_up > xlimits[1]:
                    xlimits[1] = x_up

        # Set the x and y lims
        ax.set_xlim(*xlimits)
        ax.set_ylim(*ylimits)
        filter_ax.set_ylim(0, 2 * max_t)
        filter_ax.set_xlim(*ax.get_xlim())

        # Parse the units for the labels and make them pretty
        x_units = str(self.filters[self.filter_codes[0]].lam.units)
        y_units = str(photometry.units)
        x_units = x_units.replace("/", r"\ / \ ").replace("*", " ")
        y_units = y_units.replace("/", r"\ / \ ").replace("*", " ")

        # Label the x axis
        if self.rest_frame:
            ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
        else:
            ax.set_xlabel(r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$")

        # Label the y axis handling all possibilities
        if self.rest_frame:
            ax.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
        else:
            ax.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

        # Filter axis label
        filter_ax.set_ylabel("$T$")

        return fig, ax
