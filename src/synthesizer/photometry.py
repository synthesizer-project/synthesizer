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

    def __str__(self):
        """
        Allow for a summary to be printed.
        """

        # Determine the width of each column
        column_widths = [
            max(len(str(header)), len(str(phot))) + 2
            for header, phot in self._look_up.items()
        ]

        # How many characters make up the table?
        tot_width = sum(column_widths) + len(column_widths)

        # Create the table header
        header_row = "|".join(
            f"{header.center(width)}"
            for header, width in zip(self.filter_codes, column_widths)
        )
        separator_row = "|".join("-" * width for width in column_widths)

        # Create the photometry row
        data_row = "|".join(
            f"{str(self[key]).center(width)}"
            for key, width in zip(self.filter_codes, column_widths)
        )

        # Combine everything into the final table
        if self.rest_frame:
            table = "REST FRAME PHOTOMETRY".center(tot_width) + "\n"
        else:
            table = "OBSERVED PHOTOMETRY".center(tot_width) + "\n"
        table += f"{header_row}\n{separator_row}\n"
        table += data_row

    def __str__(self):
        """
        Allow for a summary to be printed.
        """

        # Determine the width of each column
        column_widths = [
            max(
                len(str(header)),
                len(str(format(phot.value, ".2e")) + " " + str(phot.units)),
            )
            + 2
            for header, phot in self._look_up.items()
        ]

        # How many characters make up the table?
        tot_width = sum(column_widths) + len(column_widths)

        # Create the table header
        header_row = "|".join(
            f"{header.center(width)}"
            for header, width in zip(self.filter_codes, column_widths)
        )
        separator_row = "|".join("-" * width for width in column_widths)

        # Create the photometry row
        data_row = "|".join(
            f"{(str(format(self[key].value, '.2e')) + ' ' + str(self[key].units)).center(width)}"
            for key, width in zip(self.filter_codes, column_widths)
        )

        # Create the centered title with "=" on either side
        if self.rest_frame:
            title = f"{'= REST FRAME PHOTOMETRY ='.center(tot_width, '=')}"
        else:
            title = f"{'= OBSERVED PHOTOMETRY ='.center(tot_width, '=')}"

        # Combine everything into the final table
        table = f"{title}\n{header_row}\n{separator_row}\n"
        table += data_row

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
