"""A module for working with photometry derived from an Sed.

This module contains a single class definition which acts as a container
for photometry data. It should never be directly instantiated, instead
internal methods that calculate photometry
(e.g. Sed.get_photo_luminosities)
return an instance of this class.
"""

import re
from typing import (
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Optional,
    Tuple,
    Union,
    ValuesView,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from unyt import unyt_array, unyt_quantity

from synthesizer.filters import FilterCollection
from synthesizer.units import Quantity, default_units


class PhotometryCollection:
    """
    A container for photometry data.

    This represents a collection of photometry values and provides unit
    association and plotting functionality.

    This is a utility class returned by functions elsewhere. Although not
    an issue if it is this should never really be directly instantiated.

    Attributes:
        photometry: Quantity instance representing photometry data.
        photo_luminosities: Quantity instance representing photometry data in
                            the rest frame.
        photo_fluxes: Quantity instance representing photometry data in the
                      observer frame.
        filters: The FilterCollection used to produce the photometry.
        filter_codes: List of filter codes.
        _look_up: A dictionary for easy access to photometry values using
                  filter codes.
    """

    filters: FilterCollection
    filter_codes: List[str]
    photometry: unyt_array
    _look_up: Dict[str, unyt_quantity]

    # Define quantities (there has to be one for rest and observer frame)
    _photo_luminosities: NDArray[np.float64]
    photo_luminosities: unyt_array
    photo_luminosities = Quantity()
    _photo_fluxes: NDArray[np.float64]
    photo_fluxes: unyt_array
    photo_fluxes = Quantity()

    def __init__(
        self,
        filters: FilterCollection,
        **kwargs: Dict[str, unyt_quantity],
    ) -> None:
        """
        Instantiate the photometry collection.

        To enable quantities a PhotometryCollection will store the data
        as arrays but enable access via dictionary syntax.

        Whether the photometry is flux or luminosity is determined by the
        units of the photometry passed.

        Args:
            filters: The FilterCollection used to produce the photometry.
            kwargs: A dictionary of keyword arguments containing all the
                    photometry of the form {"filter_code": photometry}.
        """
        # Store the filter collection
        self.filters = filters

        # Get the filter codes
        self.filter_codes = list(kwargs.keys())

        # Get the photometry
        photometry_lst: List[unyt_quantity] = list(kwargs.values())

        # Convert it from a list of unyt_quantities to a unyt_array
        photometry: unyt_array = unyt_array(
            photometry_lst, units=photometry_lst[0].units
        )

        # Get the dimensions of a flux for testing
        flux_dimensions = default_units["photo_fluxes"].units.dimensions

        # Check if the photometry is flux or luminosity
        if photometry[0].units.dimensions == flux_dimensions:
            self.photo_fluxes = photometry
            self.photo_luminosities = None
            self.photometry = self.photo_fluxes
        else:
            self.photo_luminosities = photometry
            self.photo_fluxes = None
            self.photometry = self.photo_luminosities

        # Construct a dict for the look up, importantly we here store
        # the values in photometry not _photometry meaning they have units.
        self._look_up = {
            f: val
            for f, val in zip(
                self.filter_codes,
                self.photometry,
            )
        }

    def __getitem__(self, filter_code: str) -> unyt_quantity:
        """
        Enable dictionary key look up syntax to extract specific photometry.

        e.g. Sed.photo_luminosities["JWST/NIRCam.F150W"].

        NOTE: this will always return photometry with units. Unitless
        photometry is accessible in array form via self._photo_luminosities
        or self._photo_fluxes based on what frame is desired. For
        internal use this should be fine and the UI (where this method
        would be used) should always return with units.

        Args:
            filter_code: The filter code of the desired photometry.
        """
        # Perform the look up
        return self._look_up[filter_code]

    def keys(self) -> KeysView:
        """
        Enable dict.keys behaviour.

        Returns:
             A list of filter codes.
        """
        return self._look_up.keys()

    def values(self) -> ValuesView:
        """
        Enable dict.values behaviour.

        Returns:
            A dict_values object containing the photometry.
        """
        return self._look_up.values()

    def items(self) -> ItemsView:
        """
        Enable dict.items behaviour.

        Returns:
            A dict_items object containing the filter codes and photometry.
        """
        return self._look_up.items()

    def __iter__(self) -> Iterator[Tuple[str, unyt_quantity]]:
        """
        Enable dict iter behaviour.

        Returns:
            An iterator over the filter codes and photometry.
        """
        return iter(self._look_up.items())

    def __str__(self) -> str:
        """
        Allow for a summary to be printed.

        Returns:
            A formatted string representation of the PhotometryCollection.
        """
        # Define the filter code column
        filters_col: List[str] = [
            (
                f"{f.filter_code} (\u03bb = {f.pivwv().value:.2e} "
                f"{str(f.lam.units)})"
            )
            for f in self.filters
        ]

        # Define the photometry value column
        value_col: List[str] = [
            f"{str(format(self[key].value, '.2e'))} {str(self[key].units)}"
            for key in self.filter_codes
        ]

        # Determine the width of each column
        filter_width: int = max([len(s) for s in filters_col]) + 2
        phot_width: int = max([len(s) for s in value_col]) + 2
        widths: List[int] = [filter_width, phot_width]

        # How many characters across is the table?
        tot_width: int = filter_width + phot_width + 1

        # Create the separator row
        sep: str = "|".join("-" * width for width in widths)

        # Initialise the table
        table: str = f"-{sep.replace('|', '-')}-\n"

        # Create the centered title
        title: str
        if self.photo_luminosities is not None:
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

    def plot_photometry(
        self,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        show: bool = False,
        ylimits: Union[Tuple[()], Tuple[float, float]] = (),
        xlimits: Union[Tuple[()], Tuple[float, float]] = (),
        marker: str = "+",
        figsize: Tuple[float, float] = (3.5, 5.0),
    ) -> Tuple[Figure, Axes]:
        """
        Plot the photometry alongside the filter curves.

        Args:
            fig: A pre-existing Matplotlib figure. If None, a new figure will
                 be created.
            ax: A pre-existing Matplotlib axes. If None, new axes will be
                created.
            show: If True, the plot will be displayed.
            ylimits: Tuple specifying the y-axis limits for the plot.
            xlimits: Tuple specifying the x-axis limits for the plot.
            marker: Marker style for the photometry data points.
            figsize: Tuple specifying the size of the figure.

        Returns:
            The Matplotlib figure and axes used for the plot.
        """
        # If we don't already have a figure, make one
        _fig: Figure
        if fig is None:
            # Set up the figure
            _fig = plt.figure(figsize=figsize)
        else:
            _fig = fig

        # If we don't already have an axes, make one
        _ax: Axes
        if ax is None:
            # Define the axes geometry
            left: float = 0.15
            height: float = 0.6
            bottom: float = 0.1
            width: float = 0.8

            # Create the axes
            ax = _fig.add_axes((left, bottom, width, height))

            # Set the scale to log log
            ax.semilogy()

            # Grid it... as all things should be
            ax.grid(True)

        else:
            _ax = ax

        # Add a filter axis
        filter_ax: Axes = cast(Axes, _ax.twinx())
        filter_ax.set_ylim(0, None)

        # PLot each filter curve
        max_t: float = 0.0
        for f in self.filters:
            filter_ax.plot(f.lam, f.t)
            if np.max(f.t) > max_t:
                max_t = max(f.t)

        # Get the photometry
        photometry: unyt_array = self.photometry

        # Plot the photometry
        for f, phot in zip(self.filters, photometry.value):
            pivwv: unyt_quantity = f.pivwv()
            fwhm: unyt_quantity = f.fwhm()
            _ax.errorbar(
                pivwv,
                phot,
                marker=marker,
                xerr=fwhm,
                linestyle=None,
                capsize=3,
            )

        # Do we not have y limtis?
        if len(ylimits) == 0:
            max_phot: np.float64 = np.max(photometry)
            ylimits = (
                10 ** (np.log10(max_phot) - 5),
                10 ** (np.log10(max_phot) * 1.1),
            )

        # Do we not have x limits?
        if len(xlimits) == 0:
            # Define initial xlimits
            xlimits = (np.inf, -np.inf)

            # Loop over spectra and get the total required limits
            for f in self.filters:
                # Derive the x limits from data above the ylimits
                trans_mask: NDArray[np.bool_] = f.t > 0
                lams_above: NDArray[np.float64] = f._lam[trans_mask]

                # Saftey skip if no values are above the limit
                if lams_above.size == 0:
                    continue

                # Derive the x limits
                xlimits = (
                    min(xlimits[0], lams_above.min()),
                    max(xlimits[1], lams_above.max()),
                )

            # Add some padding around the limits of the data
            xlimits = (
                10 ** (np.log10(xlimits[0]) * 0.95),
                10 ** (np.log10(xlimits[1]) * 1.05),
            )

        # Set the x and y lims
        _ax.set_xlim(*xlimits)
        _ax.set_ylim(*ylimits)
        filter_ax.set_ylim(0, 2 * max_t)
        filter_ax.set_xlim(*_ax.get_xlim())

        # Parse the units for the labels and make them pretty
        x_units: str = self.filters[self.filter_codes[0]].lam.units.latex_repr
        y_units: str = photometry.units.latex_repr

        # Replace any \frac with a \ division
        pattern: str = r"\{(.*?)\}\{(.*?)\}"
        replacement: str = r"\1 \ / \ \2"
        x_units = re.sub(pattern, replacement, x_units).replace(r"\frac", "")
        y_units = re.sub(pattern, replacement, y_units).replace(r"\frac", "")

        # Label the x axis
        if self.photo_luminosities is not None:
            _ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
        else:
            _ax.set_xlabel(
                r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$"
            )

        # Label the y axis handling all possibilities
        if self.photo_luminosities is not None:
            _ax.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
        else:
            _ax.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

        # Filter axis label
        filter_ax.set_ylabel("$T$")

        return _fig, _ax
