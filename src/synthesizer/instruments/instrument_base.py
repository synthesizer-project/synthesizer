"""Shared abstract base class for observational instruments.

This module defines :class:`InstrumentBase`, the common interface shared by
all specialised instrument implementations in Synthesizer. It provides the
small set of behaviour that is genuinely common across the hierarchy:
identity, comparison, collection composition, and capability flags.

Specialised subclasses are responsible for owning their own observing-mode
state, validation, serialisation, and observation-specific behaviour.
"""

from abc import ABC, abstractmethod

from synthesizer import exceptions
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.utils.ascii_table import TableFormatter
from synthesizer.utils.util_funcs import obj_to_hashable


class InstrumentBase(ABC):
    """Shared abstract base class for all instrument implementations.

    Capability flags are declared explicitly here. When a new instrument
    capability is introduced, it should be added to this interface and then
    overridden by the relevant specialised instrument classes.

    This base class deliberately does not attempt to represent every possible
    instrument configuration itself. Instead, it defines the common contract
    that specialised child classes must satisfy.

    In practice this means the base class owns only the genuinely shared
    behaviour of the hierarchy, while child classes own the details of their
    own observing modes.
    """

    def __init__(self, label):
        """Initialise the shared base state for an instrument.

        Args:
            label (str): A human-readable label for the instrument. This is
                used throughout Synthesizer to identify the instrument in
                stored outputs, collections, and serialised data.
        """
        self.label = label

    @property
    @abstractmethod
    def instrument_type(self):
        """Return the serialised instrument type tag.

        Returns:
            str: Short string used when serialising and deserialising the
                instrument.
        """

    @property
    def can_do_photometry(self):
        """Return whether this instrument supports photometry.

        Returns:
            bool: ``True`` if the instrument supports integrated photometry,
                otherwise ``False``.
        """
        return False

    @property
    def can_do_imaging(self):
        """Return whether this instrument supports imaging.

        Returns:
            bool: ``True`` if the instrument supports image generation,
                otherwise ``False``.
        """
        return False

    @property
    def can_do_psf_imaging(self):
        """Return whether this instrument supports PSF imaging.

        Returns:
            bool: ``True`` if the instrument has the information required to
                apply a PSF to images, otherwise ``False``.
        """
        return False

    @property
    def can_do_noisy_imaging(self):
        """Return whether this instrument supports noisy imaging.

        Returns:
            bool: ``True`` if the instrument has the information required to
                apply noise to images, otherwise ``False``.
        """
        return False

    @property
    def can_do_spectroscopy(self):
        """Return whether this instrument supports spectroscopy.

        Returns:
            bool: ``True`` if the instrument supports one-dimensional
                spectroscopy, otherwise ``False``.
        """
        return False

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether this instrument supports noisy spectroscopy.

        Returns:
            bool: ``True`` if the instrument has the information required to
                apply noise to one-dimensional spectra, otherwise ``False``.
        """
        return False

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether this instrument supports resolved spectroscopy.

        Returns:
            bool: ``True`` if the instrument supports resolved spectroscopy,
                otherwise ``False``.
        """
        return False

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this instrument supports PSF spectroscopy.

        Returns:
            bool: ``True`` if the instrument has the information required to
                apply a PSF in the resolved-spectroscopy case, otherwise
                ``False``.
        """
        return False

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this instrument supports noisy IFU work.

        Returns:
            bool: ``True`` if the instrument has the information required to
                apply noise in the resolved-spectroscopy case, otherwise
                ``False``.
        """
        return False

    @abstractmethod
    def to_hdf5(self, group):
        """Write the instrument to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised. Specialised child classes are responsible for
                writing their own observing-mode specific state into this
                group.
        """

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        """Load an instrument instance according to the class contract.

        Returns:
            InstrumentBase: A new instance of ``cls`` loaded from a persisted
                representation. The exact accepted arguments depend on the
                specialised child class.
        """

    @abstractmethod
    def _comparison_state(self):
        """Return a hashable tuple representing the instrument state.

        Returns:
            tuple: Normalised state used for hashing and equality. Child
                classes should include every attribute that materially affects
                the behaviour of the instrument.
        """

    def __str__(self):
        """Return a table representation of the instrument.

        Returns:
            str: Tabulated representation of the instrument attributes.
        """
        return TableFormatter(self).get_table("Instrument")

    def __add__(self, other):
        """Combine instruments into an instrument collection.

        Args:
            other (InstrumentBase or InstrumentCollection): Instrument or
                collection to combine with this one. If another instrument is
                passed, a new :class:`InstrumentCollection` containing both
                instruments is returned. If an existing collection is passed,
                this instrument is added to that collection.

        Returns:
            InstrumentCollection: Collection containing both instruments.
        """
        # Only instruments and instrument collections can be combined via the
        # addition protocol
        if not isinstance(other, (InstrumentBase, InstrumentCollection)):
            raise exceptions.InconsistentAddition(
                f"Cannot combine Instrument with {type(other)}."
            )

        # If we are adding into an existing collection, defer to the collection
        # implementation so the addition semantics remain centralised there
        if isinstance(other, InstrumentCollection):
            return other + self

        # Two instruments with the same label cannot be combined safely because
        # the resulting collection would have an ambiguous key
        if self.label == other.label:
            raise exceptions.InconsistentAddition(
                "Adding two instruments with the same label is ill-defined. "
                "If you want to add extra filters to an instrument, use the "
                "add_filters method."
            )

        # Otherwise create a new collection containing both instruments
        collection = InstrumentCollection()
        collection.add_instruments(self, other)
        return collection

    def __hash__(self):
        """Hash instruments by type, label, and state.

        Returns:
            int: Stable hash derived from the comparison state.
        """
        return hash((type(self), self.label, self._comparison_state()))

    def __eq__(self, other):
        """Compare instruments by type, label, and state.

        Args:
            other (object): Object to compare against.

        Returns:
            bool or NotImplemented: Equality result for compatible objects.
        """
        if not isinstance(other, InstrumentBase):
            return NotImplemented

        return (
            type(self) is type(other)
            and self.label == other.label
            and self._comparison_state() == other._comparison_state()
        )


def _hashable_state(value):
    """Normalise instrument state for comparison and hashing.

    Args:
        value (object): Arbitrary instrument attribute value. This may be a
            scalar, array-like object, dictionary, or nested structure.

    Returns:
        object: Hashable representation of ``value``.
    """
    # Delegate the actual normalisation to the shared utility so every
    # instrument class follows the same comparison rules
    return obj_to_hashable(value)
