"""A module containing functionality for working with emission lines.

Emission lines in Synthesizer are represented by the LineCollection class. This
holds a collection of emission lines, each with a rest frame wavelength, a
luminosity, and a continuum. The LineCollection class also has the ability to
calculate the flux of the lines given a redshift and cosmology, including the
observed wavelength and flux density. It also provides a number of helpful
conversions and derived properties, such as the equivalent widths, photon
energies, and line ratios.

Note that emission lines held by a LineCollection object exist at a specific
wavelength and thus don't account for any broadening effects at this time. To
account for broadening effects, `Seds` including line contribution should be
used (i.e. spectra extracted from the "nebular" type on a `Grid`).

Example usages::

    # Create a LineCollection objects
    lines = LineCollection(
        line_ids=["O III 5007 A", "H 1 1215.67A"],
        lam=[5007, 1215.67] * angstrom,
        lum=[1e40, 1e39] * erg / s,
        cont=[1e39, 1e38] * erg / s / Hz,
    )

    # Get the flux of the lines at redshift z=0.1
    lines.get_flux(cosmo, z=0.1)

    # Get the flux of the lines at redshift z=0.1 with IGM absorption
    lines.get_flux(cosmo, z=0.1, igm=Inoue14)

    # Get a ratio
    lines.get_ratio("R23")

    # Get a diagram
    lines.get_diagram("OHNO")

"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import (
    Hz,
    angstrom,
    c,
    cm,
    erg,
    eV,
    h,
    pc,
    s,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.conversions import lnu_to_llam, standard_to_vacuum
from synthesizer.emissions import line_ratios
from synthesizer.emissions.utils import alias_to_line_id
from synthesizer.extensions.timers import tic, toc
from synthesizer.synth_warnings import warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter


class LineCollection:
    """A class holding a collection of emission lines.

    The LineCollection class holds a collection of emission lines, each with a
    rest frame wavelength, a luminosity, and a continuum. The LineCollection
    class also has the ability to calculate the flux of the lines given a
    redshift and cosmology, including the observed wavelength and flux density.
    It also provides a number of helpful conversions and derived properties,
    such as the equivalent widths, photon energies, and line ratios.

    Note that emission lines held by a LineCollection object exist at a
    specific wavelength and thus don't account for any broadening effects at
    this time.

    Attributes:
        line_ids (list):
            A list of the line ids.
        lam (unyt_array):
            An array of rest frame line wavelengths.
        luminosity (unyt_array):
            An array of rest frame line luminosities.
        continuum (unyt_array):
            An array of rest frame line continuum.
        description (str):
            An optional description of the line collection.
        obslam (unyt_array):
            An array of observed line wavelengths.
        flux (unyt_array):
            An array of line fluxes.
        continuum_flux (unyt_array):
            An array of line continuum fluxes.
        vacuum_wavelength (unyt_array):
            An array of vacuum wavelengths.
        available_ratios (list):
            A list of the available line ratios based on the lines in the
            collection.
        available_diagrams (list):
            A list of the available line diagrams based on the lines in the
            collection.
        nlines (int):
            The number of lines in the collection.
    """

    # Define quantities, for details see units.py
    lam = Quantity("wavelength")
    luminosity = Quantity("luminosity")
    continuum = Quantity("luminosity_density_frequency")
    flux = Quantity("flux")
    continuum_flux = Quantity("flux_density_frequency")
    obslam = Quantity("wavelength")
    vacuum_wavelength = Quantity("wavelength")

    @accepts(lam=angstrom, lum=erg / s, cont=erg / s / Hz)
    def __init__(self, line_ids, lam, lum, cont, description=None):
        """Initialise the collection of emission lines.

        Args:
            line_ids (list):
                A list or array of line ids.
            lam (unyt_array):
                An array of rest frame line wavelengths.
            lum (unyt_array):
                An array of rest frame line luminosities.
            cont (unyt_array):
                An array of rest frame line continuum.
            description (str):
                A optional description of the line collection.
        """
        start = tic()

        # Set the description
        self.description = description

        # Set the line line IDs (these include the element, the ionisation
        # state, and the vacuum wavelength)
        self.line_ids = np.array(line_ids, dtype=str)

        # How many lines do we have?
        self.nlines = len(self.line_ids)

        # Set the line wavelengths
        self.lam = lam

        # Set the line luminosities
        self.luminosity = lum

        # Set the line continuum
        self.continuum = cont

        # Ensure the luminosity and continuum are the same shape
        if self.luminosity.shape != self.continuum.shape:
            raise exceptions.InconsistentArguments(
                "Luminosity and continuum arrays must have the same shape"
            )

        # We'll populate observer frame properties when get_flux is called
        self.obslam = None
        self.flux = None
        self.continuum_flux = None

        # So we can do easy look ups in the future we need to make an index
        # look up table to map line ids to indices and enable dictionary like
        # look ups
        self._line2index = {
            line_id: i for i, line_id in enumerate(self.line_ids)
        }

        # Atrributes to enable looping
        self._current_ind = 0

        # Get which ratios we have available from the lines we contain
        self.available_ratios = []
        self._which_ratios()

        # Get which diagrams we have available from the lines we contain
        self.available_diagrams = []
        self._which_diagrams()

        toc("Creating LineCollection", start)

    @property
    def id(self):
        """Return the line id.

        This is an alias to be used when theres a single line id.

        Returns:
            line_ids (list):
                A list of the line ids.
        """
        if self.nlines == 1:
            return str(self.line_ids[0])

        raise exceptions.UnimplementedFunctionality(
            "id only applicable for a single line. Use line_ids instead."
        )

    @property
    def lum(self):
        """Return the luminosity of the lines.

        This is an alias for the luminosity property.

        Returns:
            luminosity (unyt_array):
                The luminosity of the lines.
        """
        return self.luminosity

    @property
    def cont(self):
        """Return the continuum of the lines.

        This is an alias for the continuum property.

        Returns:
            continuum (unyt_array):
                The continuum of the lines.
        """
        return self.continuum

    @property
    def elements(self):
        """Return the elements of the lines.

        We relegate this to a property method to reduce the duplication of
        memory. This is rarely used in expensive use cases and is inexpensive
        enough that when it is calculating the elements on the fly is not
        prohibitive.

        Returns:
            elements (list):
                A list of the elements of the lines.
        """
        return list(
            lid.strip().split(" ")[0]
            for lids in self.line_ids
            for lid in lids.split(",")
        )

    @property
    def vacuum_wavelengths(self):
        """Return the vacuum wavelengths of the lines.

        We relegate this to a property method to reduce the duplication of
        memory. This is rarely used in expensive use cases and is inexpensive
        enough that when it is calculating the vacuum wavelengths on the fly
        is not prohibitive.

        Returns:
            vacuum_wavelengths (unyt_array):
                The vacuum wavelengths of the lines.
        """
        return standard_to_vacuum(self.lam)

    @property
    def _individual_line_ids(self):
        """Return the individual line ids.

        We relegate this to a property method to reduce the duplication of
        memory. This is rarely used in expensive use cases and is inexpensive
        enough that when it is calculating the individual line ids on the fly
        is not prohibitive.

        Returns:
            individual_line_ids (list):
                A list of the individual line ids.
        """
        return np.array(
            [lid for lids in self.line_ids for lid in lids.split(",")]
        )

    @property
    def nu(self):
        """Return the frequency of the line.

        Returns:
            nu (unyt_quantity):
                The frequency of the line in Hz.
        """
        return (c / self.lam).to(Hz)

    @property
    def obsnu(self):
        """Return the observed frequency of the line.

        Returns:
            obsnu (unyt_quantity):
                The observed frequency of the line in Hz.
        """
        if self.obslam is None:
            return None
        return (c / self.obslam).to(Hz)

    @property
    def energy(self):
        """Get the wavelengths in terms of photon energies in eV.

        Returns:
            energy (unyt_array):
                The energy coordinate.
        """
        return (h * c / self.lam).to(eV)

    @property
    def continuum_llam(self):
        """Return the continuum in units of Llam (erg/s/AA).

        Returns:
            continuum_llam (unyt_quantity):
                The continuum in units of Llam (erg/s/AA).
        """
        return lnu_to_llam(self.lam, self.continuum)

    @property
    def equivalent_width(self):
        """Return the equivalent width.

        Returns:
            equivalent_width (unyt_quantity):
                The equivalent width of the line.
        """
        return self.luminosity / self.continuum_llam

    @property
    def ndim(self):
        """Get the dimensions of the spectra array.

        Returns:
            Tuple
                The shape of self.lnu
        """
        return np.ndim(self.lum)

    @property
    def shape(self):
        """Get the shape of the spectra array.

        Returns:
            Tuple
                The shape of self.lnu
        """
        return self.luminosity.shape

    @property
    def nlam(self):
        """Get the number of wavelength bins.

        Returns:
            int
                The number of wavelength bins.
        """
        return len(self.line_ids)

    def __str__(self):
        """Return a string representation of the SED object.

        Returns:
            table (str):
                A string representation of the SED object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("LineCollection")

    def __iter__(self):
        """Allow iteration over the LineCollection object.

        Overload iteration to allow simple looping over Line objects,
        combined with __next__ this enables for l in LineCollection syntax

        Returns:
            self (LineCollection)
                The LineCollection object.
        """
        return self

    def __next__(self):
        """Return the next line ID in the LineCollection object.

        Overload iteration to allow simple looping over Line objects,
        combined with __iter__ this enables for l in LineCollection syntax

        Returns:
            line_id (str):
                The line ID.
        """
        # Check we haven't finished
        if self._current_ind >= self.nlines:
            self._current_ind = 0
            raise StopIteration
        else:
            # Increment index
            self._current_ind += 1

            # Return the filter
            return self[self.line_ids[self._current_ind - 1]]

    def __len__(self):
        """Return the number of lines in the collection."""
        return self.nlines

    def __mul__(self, scaling):
        """Scale the line by a given factor.

        Overloads * operator to allow direct scaling of Line objects.

        Returns:
            (Line)
                New instance of Line containing the scaled line.
        """
        return self.scale(scaling)

    def __rmul__(self, scaling):
        """Scale the line by a given factor.

        Overloads * operator to allow direct scaling of Line objects.

        Returns:
            (Line)
                New instance of Line containing the scaled line.
        """
        return self.scale(scaling)

    def __add__(self, other):
        """Add two Line objects together.

        Overloads + operator to allow direct addition of Line objects.

        Note, this will only act on the rest frame quantities. To get the
        flux of the summed lines get_flux must be called on the new Line
        object.

        Returns:
            (Line)
                New instance of Line containing the summed lines.
        """
        # Check if the other object is a LineCollection
        if not isinstance(other, LineCollection):
            raise exceptions.InconsistentArguments(
                "Addition only supported between LineCollection objects"
            )

        # Check the LineCollections are compatible
        if np.any(self.line_ids != other.line_ids):
            raise exceptions.InconsistentArguments(
                "LineCollections must contain the same lines to be added"
            )

        return LineCollection(
            line_ids=self.line_ids,
            lam=self.lam,
            lum=self.luminosity + other.luminosity,
            cont=self.continuum + other.continuum,
        )

    def __getitem__(self, key):
        """Return a subset of lines from the collection.

        This method enables the extraction of various stored and derived line
        properties. The key can be one of the following:

        1. A single line id (e.g. lines["O III 5007 A"]), to return a single.
        2. A list of line ids (e.g. lines[["O III 5007 A", "H 1 1215.67A"]]),
           to return a subset of lines.
        3. A string containing a comma separated list of line ids
           (e.g. lines["O III 5007 A, H 1 1215.67A"]), to return a composite
           line (e.g. a doublet, triplet, etc) where each line is summed.
        4. A string defining a ratio_id from the line_ratios module (e.g.
           lines["R23"]), to return the line ratio (see get_ratio).
        5. A string defining a diagram_id from the line_ratios module (e.g.
           lines["OHNO"]), to return the line diagram (see get_diagram).

        Args:
            key (str/list):
                The id/s of the line/ratio/diagram. If a string contains a
                comma separated list of line_ids a composite line will be
                returned containing the sum of the luminosities and the mean
                of the wavelengths. If a list of line_ids is provided a subset
                of lines will be returned. If a single line_id is passed a
                single line will be returned. If a ratio_id or diagram_id is
                passed the result of the ratio or diagram will be returned.

        Returns:
            line (Line)
                The line object.
        """
        # Is the key a ratio_id?
        if isinstance(key, str) and key in self.available_ratios:
            return self.get_ratio(key)

        # Is the key a diagram_id?
        elif isinstance(key, str) and key in self.available_diagrams:
            return self.get_diagram(key)

        # Otherwise, we have a line_id
        line_id = key

        # Do we have a list of lines?
        if isinstance(line_id, (list, tuple, np.ndarray)):
            # Define lists to hold the indices we'll need to extract
            line_ids = []
            lam = []
            lum = []
            cont = []
            flux = []
            obslam = []
            cont_flux = []

            # Loop over each line_id and extract the relevant properties
            for li in line_id:
                single_line = self[li]

                line_ids.extend(single_line.line_ids)
                lam.extend(single_line.lam)
                lum.extend(single_line.luminosity)
                cont.extend(single_line.continuum)
                if single_line.flux is not None:
                    flux.extend(single_line.flux)
                    obslam.extend(single_line.obslam)
                    cont_flux.extend(single_line.continuum_flux)

            # Convert to arrays
            line_ids = np.array(line_ids)
            lam = np.array(lam)
            lum = np.array(lum).T
            cont = np.array(cont).T
            flux = np.array(flux).T
            obslam = np.array(obslam).T
            cont_flux = np.array(cont_flux).T

            # Create the new line (converting to unyt_arrays)
            new_line = LineCollection(
                line_ids=line_ids,
                lam=unyt_array(lam, self.lam.units),
                lum=unyt_array(lum, self.luminosity.units),
                cont=unyt_array(cont, self.continuum.units),
            )

            # Copy over the flux and observed wavelength if they exist
            if len(flux) > 0:
                new_line.flux = unyt_array(flux, self.flux.units)
                new_line.obslam = unyt_array(obslam, self.obslam.units)
                new_line.continuum_flux = unyt_array(
                    cont_flux, self.continuum_flux.units
                )

            return new_line

        # Do we have a single line?
        elif (
            isinstance(line_id, str)
            and alias_to_line_id(line_id) in self.line_ids
        ):
            # Are we dealing with an alias?
            line_id = alias_to_line_id(line_id)

            # Return the single line
            new_line = LineCollection(
                line_ids=[line_id],
                lam=unyt_array(
                    [self.lam[self._line2index[line_id]]],
                    self.lam.units,
                ),
                lum=unyt_array(
                    [self.luminosity[..., self._line2index[line_id]]],
                    self.luminosity.units,
                ),
                cont=unyt_array(
                    [self.continuum[..., self._line2index[line_id]]],
                    self.continuum.units,
                ),
            )

            # Copy over the flux and observed wavelength if they exist
            if self.flux is not None:
                new_line.flux = unyt_array(
                    [self.flux[..., self._line2index[line_id]]],
                    self.flux.units,
                )
                new_line.continuum_flux = unyt_array(
                    [self.continuum_flux[..., self._line2index[line_id]]],
                    self.continuum_flux.units,
                )
                new_line.obslam = unyt_array(
                    [self.obslam[self._line2index[line_id]]],
                    self.obslam.units,
                )

            return new_line

        # Do we have a blended line?
        elif isinstance(line_id, str) and "," in line_id:
            # Split the line id into a list of individual lines, handling any
            # aliases
            line_ids = [
                alias_to_line_id(li.strip()) for li in line_id.split(",")
            ]

            # Loop over the lines and combine them into a single line
            new_lam = self.lam[self._line2index[line_ids[0]]]
            new_lum = self.luminosity[..., self._line2index[line_ids[0]]]
            new_cont = self.continuum[..., self._line2index[line_ids[0]]]
            new_flux = (
                self.flux[..., self._line2index[line_ids[0]]]
                if self.flux is not None
                else None
            )
            new_obs_cont = (
                self.continuum_flux[..., self._line2index[line_ids[0]]]
                if self.continuum_flux is not None
                else None
            )
            new_obslam = (
                self.obslam[self._line2index[line_ids[0]]]
                if self.obslam is not None
                else None
            )
            for li in line_ids[1:]:
                new_lam += self.lam[self._line2index[li]]
                new_lum += self.luminosity[..., self._line2index[li]]
                new_cont += self.continuum[..., self._line2index[li]]
                if self.flux is not None:
                    new_flux += self.flux[..., self._line2index[li]]
                    new_obs_cont += self.continuum_flux[
                        ..., self._line2index[li]
                    ]
                    new_obslam += self.obslam[self._line2index[li]]

            # Create the new line (converting the wavelength to the mean of the
            # individual lines)
            new_line = LineCollection(
                line_ids=[line_id],
                lam=unyt_array([new_lam / len(line_ids)], self.lam.units),
                lum=unyt_array([new_lum], self.luminosity.units),
                cont=unyt_array([new_cont], self.continuum.units),
            )

            # Copy over the flux and observed wavelength if they exist
            # (converting the wavelength to the mean of the individual lines)
            if self.flux is not None:
                new_line.flux = unyt_array([new_flux], self.flux.units)
                new_line.continuum_flux = unyt_array(
                    [new_obs_cont], self.continuum_flux.units
                )
                new_line.obslam = unyt_array(
                    [new_obslam / len(line_ids)], self.obslam.units
                )

            return new_line

        # Otherwise, raise an exception, we have two options here, either the
        # line_id is not recognised or the type is not recognised
        if isinstance(line_id, str):
            raise exceptions.UnrecognisedOption(
                f"Unrecognised line_id (line_id={line_id})"
            )
        raise exceptions.UnrecognisedOption(
            "Unrecognised line_id type. Please provide a string, list, or "
            f"comma separated string (type={type(line_id)} line_id={line_id})"
        )

    def sum(self, axis=None):
        """Sum the lines in the collection.

        Args:
            axis (int/tuple):
                The axis/axes to sum over. By default this will sum over all
                but the final axis (the axis containing different lines).
        """
        # First lets check if we have a multidimensional line collection, if
        # not we can just return the single line object
        if self.ndim == 1 and self.nlines == 1:
            return self

        # Having a single line is a special case where we still need to sum
        # but ndim > 1
        if self.ndim == 1:
            if axis is not None and axis != 0:
                raise exceptions.InconsistentArguments(
                    f"Axis {axis} not compatible with LineCollection of ndim=1"
                )
            return LineCollection(
                line_ids=self.line_ids,
                lam=self.lam,
                lum=np.nansum(self.luminosity, axis=axis),
                cont=np.nansum(self.continuum, axis=axis),
            )

        # If no axes are passed we will sum over all but the last axis
        if axis is None:
            axis = tuple(range(self.ndim - 1))

        return LineCollection(
            line_ids=self.line_ids,
            lam=self.lam,
            lum=np.nansum(self.luminosity, axis=axis),
            cont=np.nansum(self.continuum, axis=axis),
        )

    def concat(self, *other_lines):
        """Concatenate LineCollections together.

        This will combine the array along the first axis. For example, if we
        have two LineCollections with LineCollection.shape = (10, 100) and
        (100, 100) the resulting LineCollection will have shape (110, 100).

        This cannot combine LineCollections with different line_ids.

        Args:
            other_lines (LineCollection):
                The other LineCollections to concatenate.

        Returns:
            LineCollection
                The concatenated LineCollection.
        """
        # Check the LineCollections are compatible
        for other_line in other_lines:
            if np.any(self.line_ids != other_line.line_ids):
                raise exceptions.InconsistentArguments(
                    "LineCollections must contain the same lines to be added"
                )

        # Concatenate the LineCollections
        return LineCollection(
            line_ids=self.line_ids,
            lam=self.lam,
            lum=np.vstack(
                [self.luminosity, *[ol.luminosity for ol in other_lines]],
            ),
            cont=np.vstack(
                [self.continuum, *[ol.continuum for ol in other_lines]],
            ),
        )

    def extend(self, *other_lines):
        """Extend the LineCollection with additional lines.

        This will add the lines to the end of the LineCollection. The combined
        LineCollections must not contain any duplicate lines.

        Args:
            other_lines (LineCollection):
                The other LineCollections to extend with.

        Returns:
            LineCollection
                The extended LineCollection.
        """
        # Check the LineCollections are compatible
        lst_ids = [*self.line_ids]
        set_ids = set(self.line_ids)
        for other_line in other_lines:
            lst_ids.extend(other_line.line_ids)
            set_ids.update(other_line.line_ids)

        # Check for duplicates
        if len(lst_ids) != len(set_ids):
            raise exceptions.InconsistentArguments(
                "LineCollections must contain different lines to be combined"
            )

        # Extend the LineCollection
        new_line_ids = np.concatenate(
            [self.line_ids, *[ol.line_ids for ol in other_lines]]
        )
        new_lam = np.concatenate(
            [self.lam, *[ol.lam for ol in other_lines]], axis=0
        )
        new_lum = np.concatenate(
            [self.luminosity, *[ol.luminosity for ol in other_lines]],
            axis=-1,
        )
        new_cont = np.concatenate(
            [self.continuum, *[ol.continuum for ol in other_lines]],
            axis=-1,
        )

        return LineCollection(
            line_ids=new_line_ids,
            lam=new_lam,
            lum=new_lum,
            cont=new_cont,
        )

    def _which_ratios(self):
        """Determine the available line ratios for this LineCollection."""
        # Create list of available line ratios
        for ratio_id, ratio in line_ratios.ratios.items():
            # Create a set from the ratio line ids while also unpacking
            # any comma separated lines
            ratio_line_ids = set()
            for lis in ratio:
                ratio_line_ids.update({li.strip() for li in lis.split(",")})

            # If we have all the lines required for this ratio, add it to the
            # list of available ratios
            if ratio_line_ids.issubset(self._individual_line_ids):
                self.available_ratios.append(ratio_id)

    def _which_diagrams(self):
        """Determine the available line diagrams for this LineCollection."""
        # Create list of available line diagnostics
        self.available_diagrams = []
        for diagram_id, diagram in line_ratios.diagrams.items():
            # Create a set from the diagram line ids while also unpacking
            # any comma separated lines
            diagram_line_ids = set()
            for ratio in diagram:
                for lis in ratio:
                    diagram_line_ids.update(
                        {li.strip() for li in lis.split(",")}
                    )

            # If we have all the lines required for this diagram, add it to the
            # list of available diagrams
            if set(diagram_line_ids).issubset(self.line_ids):
                self.available_diagrams.append(diagram_id)

    def get_flux0(self):
        """Calculate the rest frame line flux.

        Uses a standard distance of 10pc to calculate the flux.

        This will also populate the observed_wavelength attribute with the
        wavelength of the line when observed (which in the rest frame is the
        same as the emitted wavelength).

        Returns:
            flux (unyt_quantity):
                Flux of the line in units of erg/s/cm2 by default.
        """
        # Compute flux
        self.flux = self.luminosity / (4 * np.pi * (10 * pc) ** 2)
        self.continuum_flux = self.continuum / (4 * np.pi * (10 * pc) ** 2)

        # Set the observed wavelength (in this case this is the rest frame
        # wavelength)
        self.obslam = self.lam

        return self.flux

    def get_flux(self, cosmo, z, igm=None):
        """Calculate the line flux given a redshift and cosmology.

        This will also populate the observed_wavelength attribute with the
        wavelength of the line when observed.

        NOTE: if a redshift of 0 is passed the flux return will be calculated
        assuming a distance of 10 pc omitting IGM since at this distance
        IGM contribution makes no sense.

        Args:
            cosmo (astropy.cosmology.Cosmology):
                Astropy cosmology object.
            z (float):
                The redshift.
            igm (igm):
                The IGM class. e.g. `synthesizer.igm.Inoue14`.
                Defaults to None.

        Returns:
            flux (unyt_quantity):
                Flux of the line in units of erg/s/cm2 by default.
        """
        # If the redshift is 0 we can assume a distance of 10pc and ignore
        # the IGM
        if z == 0:
            return self.get_flux0()

        # Get the luminosity distance
        luminosity_distance = (
            cosmo.luminosity_distance(z).to("cm").value
        ) * cm

        # Compute flux and observed continuum
        self.flux = self.luminosity / (4 * np.pi * luminosity_distance**2)
        self.continuum_flux = self.continuum / (
            4 * np.pi * luminosity_distance**2
        )

        # Set the observed wavelength
        self.obslam = self.lam * (1 + z)

        # If we are applying an IGM model apply it
        if igm is not None:
            # Support both class references and instantiated objects
            if callable(igm):
                igm_transmission = igm().get_transmission(z, self.obslam)
            else:
                igm_transmission = igm.get_transmission(z, self.obslam)
            self.flux *= igm_transmission
            self.continuum_flux *= igm_transmission

        return self.flux

    def _get_ratio(self, numer_line, denom_line):
        """Measure (and return) a line ratio.

        Args:
            numer_line (str/list):
                The line or lines in the numerator.
            denom_line (str/list):
                The line or lines in the denominator.

        Returns:
            float
                a line ratio
        """
        # Handle the different cases for the numerator
        if isinstance(numer_line, str) and "," in numer_line:
            # Split the line id into a list of individual lines
            numer_line = [lid.strip() for lid in numer_line.split(",")]
        elif isinstance(numer_line, str):
            numer_line = [
                numer_line.strip(),
            ]
        elif isinstance(numer_line, list):
            numer_line = [lid.strip() for lid in numer_line]
        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised line_id type (type={type(numer_line)}). "
                "Please provide a string or list of strings."
            )

        # And the same for the denominator
        if isinstance(denom_line, str) and "," in denom_line:
            # Split the line id into a list of individual lines
            denom_line = [lid.strip() for lid in denom_line.split(",")]
        elif isinstance(denom_line, str):
            denom_line = [
                denom_line.strip(),
            ]
        elif isinstance(denom_line, list):
            denom_line = [lid.strip() for lid in denom_line]
        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised line_id type (type={type(denom_line)}). "
                "Please provide a string or list of strings."
            )

        # Ensure we have the lines required for the ratio
        if not set(numer_line).issubset(self.line_ids):
            raise exceptions.UnrecognisedOption(
                "LineCollection is missing the lines required for "
                f"this ratio (missing={set(numer_line) - set(self.line_ids)})"
            )
        if not set(denom_line).issubset(self.line_ids):
            raise exceptions.UnrecognisedOption(
                "LineCollection is missing the lines required for "
                f"this ratio (missing={set(denom_line) - set(self.line_ids)})"
            )

        # Sum the luminosities of the numerator and denominator
        numer_lum = np.sum(
            [self[lid].luminosity for lid in numer_line],
            axis=0,
        )
        denom_lum = np.sum(
            [self[lid].luminosity for lid in denom_line],
            axis=0,
        )

        # Return a single float if we have a single line
        if numer_lum.ndim == 1:
            return float(numer_lum / denom_lum)

        # Otherwise return the ratio array
        return numer_lum / denom_lum

    def get_ratio(self, ratio_id):
        """Measure (and return) a line ratio.

        Args:
            ratio_id (str, list):
                Either a ratio_id where the ratio lines are defined in
                line_ratios or a list of lines.

        Returns:
            float
                a line ratio
        """
        # If ratio_id is a string we interpret it as a ratio_id for the ratios
        # defined in the line_ratios module
        if isinstance(ratio_id, str):
            # Check if ratio_id exists
            if ratio_id not in line_ratios.available_ratios:
                raise exceptions.UnrecognisedOption(
                    f"ratio_id not recognised ({ratio_id})"
                )

            # Check we have the lines required for this ratio
            elif ratio_id not in self.available_ratios:
                raise exceptions.UnrecognisedOption(
                    "LineCollection is missing the lines required for "
                    f"this ratio ({ratio_id})"
                )

            line1, line2 = line_ratios.ratios[ratio_id]

        # Otherwise interpret it as a list, it must only contain two elements
        # for a ratio though, each element can be a list of lines or a single
        # string
        elif isinstance(ratio_id, list):
            # Ensure the list only contains two lines
            if len(ratio_id) != 2:
                raise exceptions.InconsistentArguments(
                    "A ratio must contain two elements, the first defining"
                    " the numerator and the second the denominator. Either "
                    " can be a single string or a list of strings defining "
                    " the lines."
                )
            line1, line2 = ratio_id

        return self._get_ratio(line1, line2)

    def get_diagram(self, diagram_id):
        """Return a pair of line ratios for a given diagram_id (E.g. BPT).

        Args:
            diagram_id (str, list):
                Either a diagram_id where the pairs of ratio lines are defined
                in line_ratios or a list of lists defining the ratios.

        Returns:
            tuple (float):
                a pair of line ratios
        """
        # If ratio_id is a string interpret as a ratio_id for the ratios
        # defined in the line_ratios module...
        if isinstance(diagram_id, str):
            # check if ratio_id exists
            if diagram_id not in line_ratios.available_diagrams:
                raise exceptions.UnrecognisedOption(
                    f"diagram_id not recognised ({diagram_id})"
                )

            # check if ratio_id exists
            elif diagram_id not in self.available_diagrams:
                raise exceptions.UnrecognisedOption(
                    "LineCollection is missing the lines required for "
                    f"this diagram ({diagram_id})"
                )

            ab, cd = line_ratios.diagrams[diagram_id]

        # Otherwise interpret as a list
        elif isinstance(diagram_id, list):
            ab, cd = diagram_id

        return self._get_ratio(*ab), self._get_ratio(*cd)

    def scale(self, scaling, inplace=False, mask=None, **kwargs):
        """Scale the lines by a given factor.

        Note: this will only scale the rest frame continuum and luminosity.
        To get the scaled flux get_flux must be called on the new Line object.

        Args:
            scaling (float):
                The factor by which to scale the line.
            inplace (bool):
                If True the Line object will be scaled in place, otherwise a
                new Line object will be returned.
            mask (array-like, bool):
                A mask array with an entry for each line. Masked out
                spectra will not be scaled. Only applicable for
                multidimensional lines.
            **kwargs (dict):
                Additional keyword arguments to pass to the scaling function.

        Returns:
            LineCollection
                A new LineCollection object containing the scaled lines, unless
                inplace is True in which case the LineCollection object will be
                scaled in place.
        """
        # If we have units make sure they are ok and then strip them
        if isinstance(scaling, (unyt_array, unyt_quantity)):
            # Check if we have compatible units with the continuum
            if self.continuum.units.dimensions == scaling.units.dimensions:
                scaling_cont = scaling.to(self.continuum.units).value
                scaling_lum = (
                    (scaling * self.nu).to(self.luminosity.units).value
                )
            elif self.luminosity.units.dimensions == scaling.units.dimensions:
                scaling_lum = scaling.to(self.luminosity.units).value
                scaling_cont = (
                    (scaling / self.nu).to(self.continuum.units).value
                )
            else:
                raise exceptions.InconsistentMultiplication(
                    f"{scaling.units} is neither compatible with the "
                    f"continuum ({self.continuum.units}) nor the "
                    f"luminosity ({self.luminosity.units})"
                )
        else:
            # Ok, dimensionless scaling is easier
            scaling_cont = scaling
            scaling_lum = scaling

        # Unpack the arrays we'll need during the scaling
        lum = self._luminosity.copy()
        cont = self._continuum.copy()

        # First we will handle the luminosity scaling (we need to do each
        # individually because the scalings can have different dimensions
        # depending on the conversion done above)

        # Handle a scalar scaling factor
        if np.isscalar(scaling_lum):
            if mask is None:
                lum *= scaling_lum
            else:
                lum[mask] *= scaling_lum

        # Handle an single element array scaling factor
        elif scaling_lum.size == 1:
            scaling_lum = scaling_lum.item()
            if mask is None:
                lum *= scaling_lum
            else:
                lum[mask] *= scaling_lum

        # Handle the case where we have a 1D scaling array that matches the
        # wavelength axis
        elif scaling_lum.ndim == 1 and scaling_lum.size == self.lam.size:
            if mask is None:
                lum *= scaling_lum
            else:
                lum[mask] *= scaling_lum[mask]

        # Handle a multi-element array scaling factor as long as it matches
        # the shape of the lnu array up to the dimensions of the scaling array
        elif isinstance(scaling_lum, np.ndarray) and len(
            scaling_lum.shape
        ) < len(self.shape):
            # We need to expand the scaling array to match the lnu array
            expand_axes = tuple(range(len(scaling_lum.shape), len(self.shape)))
            new_scaling_lum = np.ones(self.shape) * np.expand_dims(
                scaling_lum, axis=expand_axes
            )

            # Now we can multiply the arrays together
            if mask is None:
                lum *= new_scaling_lum
            else:
                lum[mask] *= new_scaling_lum[mask]

        # If the scaling array is the same shape as the lnu array then we can
        # just multiply them together
        elif (
            isinstance(scaling_lum, np.ndarray)
            and scaling_lum.shape == self.shape
        ):
            if mask is None:
                lum *= scaling_lum
            else:
                lum[mask] *= scaling_lum[mask]

        # Otherwise, we've been handed a bad scaling factor
        else:
            out_str = (
                "Incompatible scaling factor for luminsoity "
                f"with type {type(scaling)} "
            )
            if hasattr(scaling, "shape"):
                out_str += f"and shape {scaling.shape}"
            else:
                out_str += f"and value {scaling}"
            raise exceptions.InconsistentMultiplication(out_str)

        # Handle a scalar scaling factor
        if np.isscalar(scaling_cont):
            if mask is None:
                cont *= scaling_cont
            else:
                cont[mask] *= scaling_cont

        # Handle an single element array scaling factor
        elif scaling_cont.size == 1:
            scaling_cont = scaling_cont.item()
            if mask is None:
                cont *= scaling_cont
            else:
                cont[mask] *= scaling_cont

        # Handle the case where we have a 1D scaling array that matches the
        # wavelength axis
        elif scaling_cont.ndim == 1 and scaling_cont.size == self.lam.size:
            if mask is None:
                cont *= scaling_cont
            else:
                cont[mask] *= scaling_cont[mask]

        # Handle a multi-element array scaling factor as long as it matches
        elif isinstance(scaling_cont, np.ndarray) and len(
            scaling_cont.shape
        ) < len(self.shape):
            # We need to expand the scaling array to match the lnu array
            expand_axes = tuple(
                range(len(scaling_cont.shape), len(self.shape))
            )
            new_scaling_cont = np.ones(self.shape) * np.expand_dims(
                scaling_cont, axis=expand_axes
            )

            # Now we can multiply the arrays together
            if mask is None:
                cont *= new_scaling_cont
            else:
                cont[mask] *= new_scaling_cont[mask]

        # If the scaling array is the same shape as the lnu array then we can
        # just multiply them together
        elif (
            isinstance(scaling_cont, np.ndarray)
            and scaling_cont.shape == self.shape
        ):
            if mask is None:
                cont *= scaling_cont
            else:
                cont[mask] *= scaling_cont[mask]

        # Otherwise, we've been handed a bad scaling factor
        else:
            out_str = (
                "Incompatible scaling factor for continuum "
                f"with type {type(scaling)} "
            )
            if hasattr(scaling, "shape"):
                out_str += f"and shape {scaling.shape}"
            else:
                out_str += f"and value {scaling}"
            raise exceptions.InconsistentMultiplication(out_str)

        # If we aren't doing this inplace then return a new Line object
        if not inplace:
            return LineCollection(
                line_ids=self.line_ids,
                lam=self.lam,
                lum=lum * self.luminosity.units,
                cont=cont * self.continuum.units,
            )

        # Otherwise, we need to update the Line inplace
        self._luminosity = lum
        self._continuum = cont

        return self

    def apply_attenuation(
        self,
        tau_v,
        dust_curve,
        mask=None,
    ):
        """Apply attenuation to this LineCollection.

        Args:
            tau_v (float/np.ndarray of float):
                The V-band optical depth for every star particle.
            dust_curve (synthesizer.emission_models.attenuation.*):
                An instance of one of the dust attenuation models. (defined in
                synthesizer/emission_models.attenuation.py)
            mask (array-like, bool):
                A mask array with an entry for each line. Masked out
                spectra will be ignored when applying the attenuation. Only
                applicable for multidimensional lines.

        Returns:
                LineCollection
                    A new LineCollection object containing the attenuated
                    lines.
        """
        # Ensure the mask is compatible with the spectra
        if mask is not None:
            if self._luminosity.ndim < 1:
                raise exceptions.InconsistentArguments(
                    "Masks are only applicable for Lines containing "
                    "multiple elements"
                )
            if self._luminosity.shape[0] != mask.size:
                raise exceptions.InconsistentArguments(
                    "Mask and lines are incompatible shapes "
                    f"({mask.shape}, {self.lum.shape})"
                )

        # If tau_v is an array it needs to match the spectra shape
        if isinstance(tau_v, np.ndarray):
            if self._luminosity.ndim < 1:
                raise exceptions.InconsistentArguments(
                    "Arrays of tau_v values are only applicable for Lines"
                    " containing multiple elements"
                )
            if self._luminosity.shape[0] != tau_v.size:
                raise exceptions.InconsistentArguments(
                    "tau_v and lines are incompatible shapes "
                    f"({tau_v.shape}, {self.lum.shape})"
                )

        # Compute the transmission
        transmission = dust_curve.get_transmission(tau_v, self.lam)

        # Apply the transmision
        att_lum = self.luminosity
        att_cont = self.continuum
        if mask is None:
            att_lum *= transmission
            att_cont *= transmission
        else:
            att_lum[mask] *= transmission[mask]
            att_cont[mask] *= transmission[mask]

        return LineCollection(
            line_ids=self.line_ids,
            lam=self.lam,
            lum=att_lum,
            cont=att_cont,
        )

    def plot_lines(
        self,
        subset=None,
        figsize=(8, 6),
        show=False,
        xlimits=(),
        ylimits=(),
    ):
        """Plot the lines in the LineCollection.

        Args:
            subset (str/list):
                The line_id or list of line_ids to plot. If None, all lines
                will be plotted.
            figsize (tuple):
                The size of the figure (width, height) in inches. Defaults
                to (8, 6).
            show (bool):
                Whether to show the plot.
            xlimits (tuple):
                The x-axis limits. Must be a length 2 tuple.
                Defaults to (), in which case the default limits are used.
            ylimits (tuple):
                The y-axis limits. Must be a length 2 tuple.
                Defaults to (), in which case the default limits are used.

        Returns:
            fig (matplotlib.figure.Figure)
                The figure object.
            ax (matplotlib.axes.Axes)
                The axis object.
        """
        # Are we doing all lines?
        if subset is None:
            subset = self.line_ids
            plot_lines = self
        else:
            plot_lines = self[subset]

        # Collect luminosities and wavelengths
        line_ids = plot_lines.line_ids
        luminosities = plot_lines.luminosity
        wavelengths = plot_lines.lam.to("angstrom").value

        # Remove 0s and nans
        mask = np.logical_and(luminosities > 0, ~np.isnan(luminosities))
        luminosities = luminosities[mask]
        wavelengths = wavelengths[mask]

        # Warn the user if we removed anything
        if np.sum(~mask) > 0:
            warn(
                f"Removed {np.sum(~mask)} lines with zero or NaN luminosities"
            )

        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy()

        # Plot vertical lines
        ax.vlines(
            x=wavelengths,
            ymin=min(luminosities) / 10,
            ymax=luminosities,
            color="C0",
        )

        # If we haven't been given a lower lim, set it to the minimum
        if len(xlimits) == 0:
            xlimits = (
                np.min(wavelengths),
                np.max(wavelengths),
            )
        if len(ylimits) == 0:
            ylimits = (
                min(luminosities) / 10 / (erg / s),
                max(luminosities) * 10 / (erg / s),
            )
        elif ylimits[0] is None:
            ylimits = list(ylimits)
            ylimits[0] = min(luminosities) / 10

        # Optionally label each line at the tip
        # (assuming self.line_ids is in the same order)
        for x, y, label in zip(wavelengths, luminosities, line_ids):
            # On a log scale, you might want a small offset (e.g., y*1.05)
            ax.text(x, y, label, rotation=45, ha="left", va="bottom")

        # Set the x-axis to be in Angstroms
        ax.set_xlabel(r"$ \lambda / \AA$")
        ax.set_ylabel("$L / $erg s$^{-1}$")

        # Apply limits if requested
        if len(xlimits) > 0:
            ax.set_xlim(xlimits)
        if len(ylimits) > 0:
            ax.set_ylim(ylimits)

        # Show the plot if requested
        if show:
            plt.show()

        return fig, ax

    @accepts(wavelength_bins=angstrom)
    def get_blended_lines(self, wavelength_bins):
        """Blend lines based on a set of wavelength bins.

        We use a set of wavelength bins to enable the user to control exactly
        which lines are blended together. This also enables an array to be
        used emulating an instrument's resolution.

        A simple resolution would lead to ambiguity in situations where A and
        B are blended, and B and C are blended, but A and C are not.

        Args:
            wavelength_bins (unyt_array):
                The wavelength bin edges into which the lines will be blended.
                Any lines outside the range of the bins will be ignored.

        Returns:
            LineCollection
                A new LineCollection object containing the blended lines.
        """
        # Ensure the bins are sorted and actually have a length
        wavelength_bins = np.sort(wavelength_bins)
        if len(wavelength_bins) < 2:
            raise exceptions.InconsistentArguments(
                "Wavelength bins must have a length of at least 2"
            )

        # Sort wavelengths into the bins getting the indices in each bin
        bin_inds = np.digitize(self.lam, wavelength_bins)

        # Define the blended lines new shape
        new_shape = list(self.shape)
        new_shape[-1] = len(wavelength_bins) - 1

        # Create the arrays we'll need to store the blended lines
        blended_lines_counts = np.zeros(len(wavelength_bins) - 1, dtype=int)
        blended_line_ids = [[] for _ in range(len(wavelength_bins) - 1)]
        blended_line_lams = np.zeros(len(wavelength_bins) - 1, dtype=float)
        blended_line_lums = np.zeros(new_shape, dtype=float)
        blended_line_conts = np.zeros(new_shape, dtype=float)

        # Loop bin indices and combine the lines into the blended_lines array
        for i, bin_ind in enumerate(bin_inds):
            # If the bin index is 0 or the length of the bins then it lay
            # outside the range of the bins
            if bin_ind == 0 or bin_ind == len(wavelength_bins):
                continue

            # Ok, now we can handle the off by 1 error that digitize gives us
            bin_ind -= 1

            # Added this lines contribution to the arrays
            blended_lines_counts[bin_ind] += 1
            blended_line_ids[bin_ind].append(self.line_ids[i])
            blended_line_lams[bin_ind] += self._lam[i]
            blended_line_lums[..., bin_ind] += self._luminosity[..., i]
            blended_line_conts[..., bin_ind] += self._continuum[..., i]

        # Now we need to clean up and concatenate the ids of the blended lines
        # into comma separated strings
        for i in range(len(wavelength_bins) - 1):
            if blended_lines_counts[i] == 0:
                blended_line_ids[i] = None
                continue

            # Combine the line ids into a single string
            blended_line_ids[i] = ", ".join(blended_line_ids[i])

        # Remove any bins with no lines
        mask = blended_lines_counts > 0
        blended_lines_counts = blended_lines_counts[mask]
        blended_line_ids = np.array(blended_line_ids)[mask]
        blended_line_lams = blended_line_lams[mask]
        blended_line_lums = blended_line_lums[..., mask]
        blended_line_conts = blended_line_conts[..., mask]

        # Convert the wavelengths into the means
        blended_line_lams /= blended_lines_counts

        return LineCollection(
            line_ids=blended_line_ids,
            lam=blended_line_lams * self.lam.units,
            lum=blended_line_lums * self.luminosity.units,
            cont=blended_line_conts * self.continuum.units,
        )
