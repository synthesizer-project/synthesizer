"""A module for working with photometry derived from an Sed.

This module contains a single class definition which acts as a container
for photometry data. It should never be directly instantiated, instead
internal methods that calculate photometry
(e.g. Sed.get_photo_lnu)
return an instance of this class.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import Quantity, default_units


class PhotometryCollection:
    """A container for photometry data.

    This represents a collection of photometry values and provides unit
    association and plotting functionality.

    This is a utility class returned by functions elsewhere. Although not
    an issue if it is this should never really be directly instantiated.

    Attributes:
        photometry (Quantity):
            Quantity instance representing photometry data.
        photo_lnu (Quantity):
            Quantity instance representing photometry data in the rest frame.
        photo_fnu (Quantity):
            Quantity instance representing photometry data in the
            observer frame.
        filters (FilterCollection):
            The FilterCollection used to produce the photometry.
        filter_codes (list):
            List of filter codes.
        _look_up (dict):
            A dictionary for easy access to photometry values using
            filter codes.
    """

    # Define quantities (there has to be one for rest and observer frame)
    photo_lnu = Quantity("luminosity_density_frequency")
    photo_fnu = Quantity("flux_density_frequency")

    def __init__(self, filters, **kwargs):
        """Instantiate the photometry collection.

        To enable quantities a PhotometryCollection will store the data
        as arrays but enable access via dictionary syntax.

        Whether the photometry is flux or luminosity is determined by the
        units of the photometry passed.

        Args:
            filters (FilterCollection):
                The FilterCollection used to produce the photometry.
            kwargs (dict):
                A dictionary of keyword arguments containing all the photometry
                of the form {"filter_code": photometry}.
        """
        # Store the filter collection
        self.filters = filters

        # Get the filter codes
        self.filter_codes = list(kwargs.keys())

        # Get the photometry
        photometry = list(kwargs.values())

        # Ensure we have units, if not something terrible has happened
        if not isinstance(photometry[0], (unyt_quantity, unyt_array)):
            raise exceptions.InconsistentArguments(
                "Photometry must be passed as a dict of unyt_quantities "
                f"or unyt_arrays. Got {type(photometry[0])} instead."
            )

        # Convert it from a list of unyt_quantities to a unyt_array
        photometry = unyt_array(
            np.array(photometry, dtype=np.float64),
            units=photometry[0].units,
        )

        # Get the dimensions of a flux for testing
        flux_dimensions = default_units[
            "flux_density_frequency"
        ].units.dimensions

        # Check if the photometry is flux or luminosity
        if photometry[0].units.dimensions == flux_dimensions:
            self.photo_fnu = photometry
            self.photo_lnu = None
            self.photometry = self.photo_fnu
        else:
            self.photo_lnu = photometry
            self.photo_fnu = None
            self.photometry = self.photo_lnu

        # Construct a dict for the look up, importantly we here store
        # the values in photometry not _photometry meaning they have units.
        self._look_up = {
            f: val
            for f, val in zip(
                self.filter_codes,
                self.photometry,
            )
        }

        # Attach the units for convenience
        self.units = self.photometry.units

    def __getitem__(self, filter_code):
        """Enable dictionary key look up syntax to extract specific photometry.

        e.g. Sed.photo_lnu["JWST/NIRCam.F150W"].

        NOTE: this will always return photometry with units. Unitless
        photometry is accessible in array form via self._photo_lnu
        or self._photo_fnu based on what frame is desired. For
        internal use this should be fine and the UI (where this method
        would be used) should always return with units.

        Args:
            filter_code (str):
                The filter code of the desired photometry.
        """
        # Perform the look up
        if filter_code in self._look_up:
            return self._look_up[filter_code]

        # We may be being asked for all the photometry for an observatory, e.g.
        # "JWST", in which case we should return all the photometry for that
        # observatory.
        out = {}
        for key in self.filter_codes:
            if filter_code in key:
                out[key.replace(filter_code + "/", "")] = self._look_up[key]

        # If we have found some photometry return it
        if len(out) > 0:
            return out

        # We haven't found any photometry raise an error
        raise KeyError(
            f"Filter code {filter_code} not found in photometry collection."
        )

    def keys(self):
        """Return the filter codes (keys) of the photometry.

        Returns:
            list
                A list of filter codes.
        """
        return self._look_up.keys()

    def values(self):
        """Return the photometry values.

        Returns:
            dict_values
                A dict_values object containing the photometry.
        """
        return self._look_up.values()

    def __len__(self):
        """Return the number of photometry filters in the collection.

        Returns:
            int
                The number of filters in the collection.
        """
        return len(self._look_up)

    def items(self):
        """Return a tuple of both the filter codes and photometry.

        Returns:
            dict_items
                A dict_items object containing the filter codes and
                photometry.
        """
        return self._look_up.items()

    def __len__(self):
        """Enable len() behaviour.

        Returns:
            int
                The number of filter codes.
        """
        return len(self._look_up)

    @property
    def shape(self):
        """Return the shape of the photometry array.

        Returns:
            tuple
                The shape of the photometry array.
        """
        return self.photometry.shape

    @property
    def ndim(self):
        """Return the number of dimensions of the photometry array.

        Returns:
            int
                The number of dimensions of the photometry array.
        """
        return self.photometry.ndim

    def __iter__(self):
        """Enable dict iter behaviour.

        Returns:
            iter
                An iterator over the filter codes and photometry.
        """
        return iter(self._look_up.items())

    def __str__(self):
        """Allow for a summary to be printed.

        Returns:
            str: A formatted string representation of the PhotometryCollection.
        """
        # Define the filter code column
        filters_col = [
            (
                f"{f.filter_code} (\u03bb = {f.pivwv().value:.2e} "
                f"{str(f.lam.units)})"
            )
            for f in self.filters
        ]

        # Define the photometry value column
        value_col = []
        for key in self.filter_codes:
            if self[key].value.ndim > 0:
                value_col.append(
                    f"{str(format(np.sum(self[key].value), '.2e'))} "
                    f"{str(self[key].units)}"
                )
            else:
                value_col.append(
                    f"{str(format(self[key].value, '.2e'))} "
                    f"{str(self[key].units)}"
                )

        # Determine the width of each column
        filter_width = max([len(s) for s in filters_col]) + 2
        phot_width = max([len(s) for s in value_col]) + 2
        widths = [filter_width, phot_width]

        # How many characters across is the table?
        tot_width = filter_width + phot_width + 1

        # Create the separator row
        sep = "|".join("-" * width for width in widths)

        # Initialise the table
        table = f"-{sep.replace('|', '-')}-\n"

        # Create the centered title
        if self.photo_lnu is not None:
            title = f"|{'PHOTOMETRY (LUMINOSITY)'.center(tot_width)}|"
        else:
            title = f"|{'PHOTOMETRY (FLUX)'.center(tot_width)}|"
        table += f"{title}\n|{sep}|\n"

        # Combine everything into the final table
        for filt, phot in zip(filters_col, value_col):
            table += (
                f"|{filt.center(filter_width)}|"
                f"{phot.center(phot_width)}|\n|{sep}|\n"
            )

        # Clean up the final separator
        table = table[: -tot_width - 3]
        table += f"-{sep.replace('|', '-')}-\n"

        return table

    def select(self, *filter_codes):
        """Return a PhotometryCollection with only the specified filters.

        Args:
            filter_codes (list, string):
                The filter codes of the desired photometry.
        """
        # If no filters are specified return the full photometry
        if len(filter_codes) == 0:
            return self

        # Check if the filter codes are valid
        for code in filter_codes:
            if code not in self.filter_codes:
                raise KeyError(
                    f"Filter code {code} not found in photometry collection."
                )

        # Get the photometry for the specified filters
        photometry = {code: self._look_up[code] for code in filter_codes}

        # Also extract a subset of the filters
        filters = self.filters.select(*filter_codes)

        # Return a new PhotometryCollection with the specified photometry
        return PhotometryCollection(filters, **photometry)

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
        """Plot the photometry alongside the filter curves.

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
        photometry = self.photometry

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
                10 ** (np.log10(max_phot) + 0.9),
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
        x_units = self.filters[self.filter_codes[0]].lam.units.latex_repr
        y_units = photometry.units.latex_repr

        # Replace any \frac with a \ division
        pattern = r"\{(.*?)\}\{(.*?)\}"
        replacement = r"\1 \ / \ \2"
        x_units = re.sub(pattern, replacement, x_units).replace(r"\frac", "")
        y_units = re.sub(pattern, replacement, y_units).replace(r"\frac", "")

        # Label the x axis
        if self.photo_lnu is not None:
            ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
        else:
            ax.set_xlabel(
                r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$"
            )

        # Label the y axis handling all possibilities
        if self.photo_lnu is not None:
            ax.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
        else:
            ax.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

        # Filter axis label
        filter_ax.set_ylabel("$T$")

        return fig, ax
