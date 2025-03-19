"""A submodule containing utility functions for working with emission lines."""

import re

import matplotlib.pyplot as plt
import numpy as np
from unyt import angstrom

from synthesizer import exceptions
from synthesizer.emissions.sed import Sed
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts
from synthesizer.utils.util_funcs import wavelength_to_rgba


def get_composite_line_id_from_list(id):
    """
    Convert a list of line ids to a single string describing a composite line.

    A composite line is a line that is made up of multiple lines, e.g.
    a doublet or triplet. This function takes a list of line ids and converts
    them to a single string.

    e.g. ["H 1 4861.32A", "H 1 6562.80A"] -> "H 1 4861.32A, H 1 6562.80A"

    Args
        id (list, tuple)
            a str, list, or tuple containing the id(s) of the lines

    Returns
        id (str)
            string representation of the id
    """
    if isinstance(id, list):
        return ", ".join(id)
    else:
        return id


def get_line_label(line_id):
    """
    Get a line label for a given line_id, ratio, or diagram.

    Where the line_id is one of several predifined lines in line_labels this
    label is used, otherwise the label is constructed from the line_id.

    Argumnents
        line_id (str)
            The line_id either as a list of individual lines or a string. If
            provided as a list this is automatically converted to a single
            string so it can be used as a key.

    Returns
        line_label (str)
            A nicely formatted line label.
    """
    # if the line_id is a list (denoting a doublet or higher)
    if isinstance(line_id, list):
        line_id = ", ".join(line_id)

    if line_id in line_labels.keys():
        line_label = line_labels[line_id]
    else:
        line_id = line_id.split(",")
        _line_labels = []
        for line_id_ in line_id:
            # get the element, ion, and wavelength
            element, ion, wavelength = line_id_.split(" ")

            # extract unit and convert to latex str
            unit = wavelength[-1]

            if unit == "A":
                unit = r"\AA"
            if unit == "m":
                unit = r"\mu m"
            wavelength = wavelength[:-1] + unit

            _line_labels.append(
                f"{element}{get_roman_numeral(int(ion))}{wavelength}"
            )

        line_label = "+".join(_line_labels)

    return line_label


def flatten_linelist(list_to_flatten):
    """
    Flatten a mixed list of lists and strings and remove duplicates.

    Used when converting a line list which may contain single lines
    and doublets.

    Args:
        list_to_flatten (list)
            list containing lists and/or strings and integers

    Returns:
        (list)
            flattened list
    """
    flattened_list = []
    for lst in list_to_flatten:
        if isinstance(lst, list) or isinstance(lst, tuple):
            for ll in lst:
                flattened_list.append(ll)

        elif isinstance(lst, str):
            # If the line is a doublet, resolve it and add each line
            # individually
            if len(lst.split(",")) > 1:
                flattened_list += lst.split(",")
            else:
                flattened_list.append(lst)

        else:
            raise Exception(
                (
                    "Unrecognised type provided. Please provide"
                    "a list of lists and strings"
                )
            )

    return list(set(flattened_list))


def get_roman_numeral(number):
    """
    Convert an integer into a roman numeral str.

    Used for renaming emission lines from the cloudy defaults.

    Args:
        number (int)
            The number to convert into a roman numeral.

    Returns:
        number_representation (str)
            String reprensentation of the roman numeral.
    """
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = [
        "I",
        "IV",
        "V",
        "IX",
        "X",
        "XL",
        "L",
        "XC",
        "C",
        "CD",
        "D",
        "CM",
        "M",
    ]
    i = 12

    roman = ""
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            roman += sym[i]
            div -= 1
        i -= 1
    return roman


# Shorthand for common lines
aliases = {
    "Hb": "H 1 4861.32A",
    "Ha": "H 1 6562.80A",
    "Hg": "H 1 4340.46A",
    "O1": "O 1 6300.30A",
    "O2b": "O 2 3726.03A",
    "O2r": "O 2 3728.81A",
    "O2": "O 2 3726.03A, O 2 3728.81A",
    "O3b": "O 3 4958.91A",
    "O3r": "O 3 5006.84A",
    "O3": "O 3 4958.91A, O 3 5006.84A",
    "Ne3": "Ne 3 3868.76A",
    "N2": "N 2 6583.45A",
    "S2": "S 2 6730.82A, S 2 6716.44A",
}


# Standard names
Ha = aliases["Ha"]
Hb = aliases["Hb"]
O1 = aliases["O1"]
O2b = aliases["O2b"]
O2r = aliases["O2r"]
O2 = aliases["O2"]
O3b = aliases["O3b"]
O3r = aliases["O3r"]
O3 = aliases["O3"]
Ne3 = aliases["Ne3"]
N2 = aliases["N2"]
S2 = aliases["S2"]


# Dictionary of common line labels to use by default
line_labels = {
    "O 2 3726.03A,O 2 3728.81A": "[OII]3726,3729",
    "H 1 4861.32A": r"H\beta",
    "O 3 4958.91A,O 3 5006.84A": "[OIII]4959,5007",
    "H 1 6562.80A": r"H\alpha",
    "O 3 5006.84A": "[OIII]5007",
    "N 2 6583.45A": "[NII]6583",
}


def alias_to_line_id(alias):
    """
    Convert a line alias to a line id.

    Args:
        alias (str)
            The line alias.

    Returns:
        line_id (str)
            The line id.
    """
    if alias in aliases:
        return aliases[alias]
    return alias


def plot_spectra(
    spectra,
    fig=None,
    ax=None,
    show=False,
    ylimits=(),
    xlimits=(),
    figsize=(3.5, 5),
    label=None,
    draw_legend=True,
    x_units=None,
    y_units=None,
    quantity_to_plot="lnu",
):
    """
    Plot a spectra or dictionary of spectra.

    Plots either a specific spectra or all spectra provided in a dictionary.
    The plotted "type" of spectra is defined by the quantity_to_plot keyword
    arrgument which defaults to "lnu".

    This is a generic plotting function to be used either directly or to be
    wrapped by helper methods through Synthesizer.

    Args:
        spectra (dict/Sed)
            The Sed objects from which to plot. This can either be a dictionary
            of Sed objects to plot multiple or a single Sed object to only plot
            one.
        fig (matplotlib.pyplot.figure)
            The figure containing the axis. By default one is created in this
            function.
        ax (matplotlib.axes)
            The axis to plot the data on. By default one is created in this
            function.
        show (bool)
            Flag for whether to show the plot or just return the
            figure and axes.
        ylimits (tuple)
            The limits to apply to the y axis. If not provided the limits
            will be calculated with the lower limit set to 1000 (100) times
            less than the peak of the spectrum for rest_frame (observed)
            spectra.
        xlimits (tuple)
            The limits to apply to the x axis. If not provided the optimal
            limits are found based on the ylimits.
        figsize (tuple)
            Tuple with size 2 defining the figure size.
        label (string)
            The label to give the spectra. Only applicable when Sed is a single
            spectra.
        draw_legend (bool)
            Whether to draw the legend.
        x_units (unyt.unit_object.Unit)
            The units of the x axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        y_units (unyt.unit_object.Unit)
            The units of the y axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        quantity_to_plot (string)
            The sed property to plot. Can be "lnu", "luminosity" or "llam"
            for rest frame spectra or "fnu", "flam" or "flux" for observed
            spectra. Defaults to "lnu".

    Returns:
        fig (matplotlib.pyplot.figure)
            The matplotlib figure object for the plot.
        ax (matplotlib.axes)
            The matplotlib axes object containing the plotted data.
    """
    # Check we have been given a valid quantity_to_plot
    if quantity_to_plot not in (
        "lnu",
        "llam",
        "luminosity",
        "fnu",
        "flam",
        "flux",
    ):
        raise exceptions.InconsistentArguments(
            f"{quantity_to_plot} is not a valid quantity_to_plot"
            "(can be 'fnu' or 'flam')"
        )

    # Are we plotting in the rest_frame?
    rest_frame = quantity_to_plot in ("lnu", "llam", "luminosity")

    # Make a singular Sed a dictionary for ease below
    if isinstance(spectra, Sed):
        spectra = {
            label if label is not None else "spectra": spectra,
        }

        # Don't draw a legend if not label given
        if label is None and draw_legend:
            warn("No label given, we will not draw a legend")
            draw_legend = False

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

    # Loop over the dict we have been handed, we want to do this backwards
    # to ensure the most recent spectra are on top
    keys = list(spectra.keys())[::-1]
    seds = list(spectra.values())[::-1]
    for key, sed in zip(keys, seds):
        # Get the appropriate luminosity/flux and wavelengths
        if rest_frame:
            lam = sed.lam
        else:
            # Ensure we have fluxes
            if sed.fnu is None:
                raise exceptions.MissingSpectraType(
                    f"This Sed has no fluxes ({key})! Have you called "
                    "Sed.get_fnu()?"
                )

            # Ok everything is fine
            lam = sed.obslam

        plt_spectra = getattr(sed, quantity_to_plot)

        # Prettify the label if not latex
        if not any([c in key for c in ("$", "_")]):
            key = key.replace("_", " ").title()

        # Plot this spectra
        ax.plot(lam, plt_spectra, lw=1, alpha=0.8, label=key)

    # Do we not have y limtis?
    if len(ylimits) == 0:
        # Define initial xlimits
        ylimits = [np.inf, -np.inf]

        # Loop over spectra and get the total required limits
        for sed in spectra.values():
            # Get the maximum ignoring infinites
            okinds = np.logical_and(
                getattr(sed, quantity_to_plot) > 0,
                getattr(sed, quantity_to_plot) < np.inf,
            )
            if True not in okinds:
                continue
            max_val = np.nanmax(getattr(sed, quantity_to_plot)[okinds])

            # Derive the x limits
            y_up = 10 ** (np.log10(max_val) * 1.05)
            y_low = 10 ** (np.log10(max_val) - 5)

            # Update limits
            if y_low < ylimits[0]:
                ylimits[0] = y_low
            if y_up > ylimits[1]:
                ylimits[1] = y_up

    # Do we not have x limits?
    if len(xlimits) == 0:
        # Define initial xlimits
        xlimits = [np.inf, -np.inf]

        # Loop over spectra and get the total required limits
        for sed in spectra.values():
            # Derive the x limits from data above the ylimits
            plt_spectra = getattr(sed, quantity_to_plot)
            lam_mask = plt_spectra > ylimits[0]
            if rest_frame:
                lams_above = sed.lam[lam_mask]
            else:
                lams_above = sed.obslam[lam_mask]

            # Saftey skip if no values are above the limit
            if lams_above.size == 0:
                continue

            # Derive the x limits
            x_low = 10 ** (np.log10(np.min(lams_above)) * 0.9)
            x_up = 10 ** (np.log10(np.max(lams_above)) * 1.1)

            # Update limits
            if x_low < xlimits[0]:
                xlimits[0] = x_low
            if x_up > xlimits[1]:
                xlimits[1] = x_up

    # Set the limits
    if not np.isnan(xlimits[0]) and not np.isnan(xlimits[1]):
        ax.set_xlim(*xlimits)
    if not np.isnan(ylimits[0]) and not np.isnan(ylimits[1]):
        ax.set_ylim(*ylimits)

    # Make the legend
    if draw_legend and any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize=8, labelspacing=0.0)

    # Parse the units for the labels and make them pretty
    if x_units is None:
        x_units = lam.units.latex_repr
    else:
        x_units = str(x_units)
    if y_units is None:
        y_units = plt_spectra.units.latex_repr
    else:
        y_units = str(y_units)

    # Replace any \frac with a \ division
    pattern = r"\{(.*?)\}\{(.*?)\}"
    replacement = r"\1 \ / \ \2"
    x_units = re.sub(pattern, replacement, x_units).replace(r"\frac", "")
    y_units = re.sub(pattern, replacement, y_units).replace(r"\frac", "")

    # Label the x axis
    if rest_frame:
        ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
    else:
        ax.set_xlabel(r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$")

    # Label the y axis handling all possibilities
    if quantity_to_plot == "lnu":
        ax.set_ylabel(r"$L_{\nu}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "llam":
        ax.set_ylabel(r"$L_{\lambda}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "luminosity":
        ax.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "fnu":
        ax.set_ylabel(r"$F_{\nu}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "flam":
        ax.set_ylabel(r"$F_{\lambda}/[\mathrm{" + y_units + r"}]$")
    else:
        ax.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

    # Are we showing?
    if show:
        plt.show()

    return fig, ax


def plot_observed_spectra(
    spectra,
    redshift,
    fig=None,
    ax=None,
    show=False,
    ylimits=(),
    xlimits=(),
    figsize=(3.5, 5),
    label=None,
    draw_legend=True,
    x_units=None,
    y_units=None,
    filters=None,
    quantity_to_plot="fnu",
):
    """
    Plot a spectra or dictionary of spectra.

    Plots either a specific observed spectra or all observed spectra
    provided in a dictionary.

    This function is a wrapper around plot_spectra.

    This is a generic plotting function to be used either directly or to be
    wrapped by helper methods through Synthesizer.

    Args:
        spectra (dict/Sed)
            The Sed objects from which to plot. This can either be a dictionary
            of Sed objects to plot multiple or a single Sed object to only plot
            one.
        redshift (float)
            The redshift of the observation.
        fig (matplotlib.pyplot.figure)
            The figure containing the axis. By default one is created in this
            function.
        ax (matplotlib.axes)
            The axis to plot the data on. By default one is created in this
            function.
        show (bool)
            Flag for whether to show the plot or just return the
            figure and axes.
        ylimits (tuple)
            The limits to apply to the y axis. If not provided the limits
            will be calculated with the lower limit set to 1000 (100) times
            less than the peak of the spectrum for rest_frame (observed)
            spectra.
        xlimits (tuple)
            The limits to apply to the x axis. If not provided the optimal
            limits are found based on the ylimits.
        figsize (tuple)
            Tuple with size 2 defining the figure size.
        label (string)
            The label to give the spectra. Only applicable when Sed is a single
            spectra.
        draw_legend (bool)
            Whether to draw the legend.
        x_units (unyt.unit_object.Unit)
            The units of the x axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        y_units (unyt.unit_object.Unit)
            The units of the y axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        filters (FilterCollection)
            If given then the photometry is computed and both the photometry
            and filter curves are plotted
        quantity_to_plot (string)
            The sed property to plot. Can be "fnu", "flam", or "flux".
            Defaults to "fnu".

    Returns:
        fig (matplotlib.pyplot.figure)
            The matplotlib figure object for the plot.
        ax (matplotlib.axes)
            The matplotlib axes object containing the plotted data.
    """
    # Check we have been given a valid quantity_to_plot
    if quantity_to_plot not in ("fnu", "flam"):
        raise exceptions.InconsistentArguments(
            f"{quantity_to_plot} is not a valid quantity_to_plot"
            "(can be 'fnu' or 'flam')"
        )

    # Get the observed spectra plot
    fig, ax = plot_spectra(
        spectra,
        fig=fig,
        ax=ax,
        show=False,
        ylimits=ylimits,
        xlimits=xlimits,
        figsize=figsize,
        label=label,
        draw_legend=draw_legend,
        x_units=x_units,
        y_units=y_units,
        quantity_to_plot=quantity_to_plot,
    )

    # Are we including photometry and filters?
    if filters is not None:
        # Add a filter axis
        filter_ax = ax.twinx()
        filter_ax.set_ylim(0, None)

        # PLot each filter curve
        for f in filters:
            filter_ax.plot(f.lam * (1 + redshift), f.t)

        # Make a singular Sed a dictionary for ease below
        if isinstance(spectra, Sed):
            spectra = {
                label if label is not None else "spectra": spectra,
            }

        # Loop over spectra plotting photometry and filter curves
        for sed in spectra.values():
            # Get the photometry
            sed.get_photo_fnu(filters)

            # Plot the photometry for each filter
            for f in filters:
                piv_lam = f.pivwv()
                ax.scatter(
                    piv_lam * (1 + redshift),
                    sed.photo_fnu[f.filter_code],
                    zorder=4,
                )

    if show:
        plt.show()

    return fig, ax


def plot_spectra_as_rainbow(
    sed,
    figsize=(5, 0.5),
    lam_min=3000,
    lam_max=8000,
    include_xaxis=True,
    logged=False,
    min_log_lnu=-2.0,
    use_fnu=False,
):
    """
    Create a plot of the spectrum as a rainbow.

    Arguments:
        sed (synthesizer.emissions.Sed)
            A synthesizer.emissions object.
        figsize (tuple)
            Fig-size tuple (width, height).
        lam_min (float)
            The min wavelength to plot in Angstroms.
        lam_max (float)
            The max wavelength to plot in Angstroms.
        include_xaxis (bool)
            Flag whther to include x-axis ticks and label.
        logged (bool)
            Flag whether to use logged luminosity.
        min_log_lnu (float)
            Minium luminosity to plot relative to the maximum.
        use_fnu (bool)
            Whether to plot fluxes or luminosities. If True
            fluxes are plotted, otherwise luminosities.

    Returns:
        fig (matplotlib.pyplot.figure)
            The matplotlib figure object for the plot.
        ax (matplotlib.axes)
            The matplotlib axes object containing the plotted data.
    """
    # Take sum of Seds if two dimensional
    sed = sed.sum()

    if use_fnu:
        # Define filter for spectra
        wavelength_indices = np.logical_and(
            sed._obslam < lam_max, sed._obslam > lam_min
        )
        lam = sed.obslam[wavelength_indices].to("nm").value
        spectra = sed._fnu[wavelength_indices]
    else:
        # define filter for spectra
        wavelength_indices = np.logical_and(
            sed._lam < lam_max, sed._lam > lam_min
        )
        lam = sed.lam[wavelength_indices].to("nm").value
        spectra = sed._lnu[wavelength_indices]

    # Normalise spectrum
    spectra /= np.max(spectra)

    # If logged rescale to between 0 and 1 using min_log_lnu
    if logged:
        spectra = (np.log10(spectra) - min_log_lnu) / (-min_log_lnu)
        spectra[spectra < min_log_lnu] = 0

    # initialise figure
    fig = plt.figure(figsize=figsize)

    # Initialise axes
    if include_xaxis:
        ax = fig.add_axes((0, 0.3, 1, 1))
        ax.set_xlabel(r"$\lambda/\AA$")
    else:
        ax = fig.add_axes((0, 0.0, 1, 1))
        ax.set_xticks([])

    # Set background
    ax.set_facecolor("black")

    # Always turn off y-ticks
    ax.set_yticks([])

    # Get an array of colours
    colours = np.array(
        [
            wavelength_to_rgba(lam_, alpha=spectra_)
            for lam_, spectra_ in zip(lam, spectra)
        ]
    )

    # Expand dimensions to get an image array
    im = np.expand_dims(colours, axis=0)

    # Show image
    ax.imshow(im, aspect="auto", extent=(lam_min, lam_max, 0, 1))

    return fig, ax


def get_transmission(intrinsic_sed, attenuated_sed):
    """
    Calculate transmission as a function of wavelength.

    The transmission is defined as the ratio between an attenuated and
    an intrinsic sed.

    Args:
        intrinsic_sed (Sed)
            The intrinsic spectra object.
        attenuated_sed (Sed)
            The attenuated spectra object.

    Returns:
        array-like, float
            The transmission array.
    """
    # Ensure wavelength arrays are equal
    if not np.array_equal(attenuated_sed._lam, intrinsic_sed._lam):
        raise exceptions.InconsistentArguments(
            "Wavelength arrays of input spectra must be the same!"
        )

    return attenuated_sed.lnu / intrinsic_sed.lnu


def get_attenuation(intrinsic_sed, attenuated_sed):
    """
    Calculate attenuation as a function of wavelength.

    Args:
        intrinsic_sed (Sed)
            The intrinsic spectra object.
        attenuated_sed (Sed)
            The attenuated spectra object.

    Returns:
        array-like, float
            The attenuation array in magnitudes.
    """
    # Calculate the transmission array
    transmission = get_transmission(intrinsic_sed, attenuated_sed)

    return -2.5 * np.log10(transmission)


@accepts(lam=angstrom)
def get_attenuation_at_lam(lam, intrinsic_sed, attenuated_sed):
    """
    Calculate attenuation at a given wavelength.

    Args:
        lam (float/array-like, float)
            The wavelength/s at which to evaluate the attenuation in
            the same units as sed.lam (by default angstrom).
        intrinsic_sed (Sed)
            The intrinsic spectra object.
        attenuated_sed (Sed)
            The attenuated spectra object.

    Returns:
        float/array-like, float
            The attenuation at the passed wavelength/s in magnitudes.
    """
    # Ensure lam is in the same units as the sed
    if lam.units != intrinsic_sed.lam.units:
        lam = lam.to(intrinsic_sed.lam.units)

    # Calcilate the transmission array
    attenuation = get_attenuation(intrinsic_sed, attenuated_sed)

    return np.interp(lam.value, intrinsic_sed._lam, attenuation)


def get_attenuation_at_5500(intrinsic_sed, attenuated_sed):
    """
    Calculate rest-frame FUV attenuation at 5500 angstrom.

    Args:
        intrinsic_sed (Sed)
            The intrinsic spectra object.
        attenuated_sed (Sed)
            The attenuated spectra object.

    Returns:
         float
            The attenuation at rest-frame 5500 angstrom in magnitudes.
    """
    return get_attenuation_at_lam(
        5500.0 * angstrom,
        intrinsic_sed,
        attenuated_sed,
    )


def get_attenuation_at_1500(intrinsic_sed, attenuated_sed):
    """
    Calculate rest-frame FUV attenuation at 1500 angstrom.

    Args:
        intrinsic_sed (Sed)
            The intrinsic spectra object.
        attenuated_sed (Sed)
            The attenuated spectra object.

    Returns:
         float
            The attenuation at rest-frame 1500 angstrom in magnitudes.
    """
    return get_attenuation_at_lam(
        1500.0 * angstrom,
        intrinsic_sed,
        attenuated_sed,
    )


def combine_list_of_seds(sed_list):
    """
    Convert a list of Seds into a single Sed object.

    Combines a list of `Sed` objects (length `Ngal`) into a single
    `Sed` object, with dimensions `Ngal x Nlam`. Each `Sed` object
    in the list should have an identical wavelength range.

    Args:
        sed_list (list)
            list of `Sed` objects
    """
    out_sed = sed_list[0]
    for sed in sed_list[1:]:
        out_sed = out_sed.concat(sed)

    return out_sed
