"""Shared abstract base class for observational instruments."""

from abc import ABC, abstractmethod

from synthesizer import exceptions
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.utils.ascii_table import TableFormatter
from synthesizer.utils.util_funcs import obj_to_hashable


class InstrumentBase(ABC):
    """Shared abstract base class for all instrument implementations.

    Capability flags are declared explicitly here. When a new instrument
    capability is introduced, it should be added to this interface and then
    overridden by the relevant concrete instrument classes.

    Concrete subclasses are responsible for owning the state and behaviour
    specific to their observing mode. `InstrumentBase` only provides the
    common identity, comparison, collection-composition, and capability
    interface shared across the hierarchy.
    """

    def __init__(self, label):
        """Initialise the shared base state for an instrument."""
        self.label = label

    @property
    @abstractmethod
    def instrument_type(self):
        """Return the serialised instrument type tag."""

    @property
    def can_do_photometry(self):
        """Return whether this instrument supports photometry."""
        return False

    @property
    def can_do_imaging(self):
        """Return whether this instrument supports imaging."""
        return False

    @property
    def can_do_psf_imaging(self):
        """Return whether this instrument supports PSF imaging."""
        return False

    @property
    def can_do_noisy_imaging(self):
        """Return whether this instrument supports noisy imaging."""
        return False

    @property
    def can_do_spectroscopy(self):
        """Return whether this instrument supports spectroscopy."""
        return False

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether this instrument supports noisy spectroscopy."""
        return False

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether this instrument supports resolved spectroscopy."""
        return False

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this instrument supports PSF spectroscopy."""
        return False

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this instrument supports noisy IFU work."""
        return False

    @abstractmethod
    def to_hdf5(self, group):
        """Write the instrument to an HDF5 group."""

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        """Load an instrument instance according to the class contract."""

    @abstractmethod
    def _comparison_state(self):
        """Return a hashable tuple representing the instrument state."""

    def __str__(self):
        """Return a table representation of the instrument."""
        return TableFormatter(self).get_table("Instrument")

    def __add__(self, other):
        """Combine instruments into an instrument collection."""
        if not isinstance(other, (InstrumentBase, InstrumentCollection)):
            raise exceptions.InconsistentAddition(
                f"Cannot combine Instrument with {type(other)}."
            )

        if isinstance(other, InstrumentCollection):
            return other + self

        if self.label == other.label:
            raise exceptions.InconsistentAddition(
                "Adding two instruments with the same label is ill-defined. "
                "If you want to add extra filters to an instrument, use the "
                "add_filters method."
            )

        collection = InstrumentCollection()
        collection.add_instruments(self, other)
        return collection

    def __hash__(self):
        """Hash instruments by concrete type, label, and state."""
        return hash((type(self), self.label, self._comparison_state()))

    def __eq__(self, other):
        """Compare instruments by concrete type, label, and state."""
        if not isinstance(other, InstrumentBase):
            return NotImplemented

        return (
            type(self) is type(other)
            and self.label == other.label
            and self._comparison_state() == other._comparison_state()
        )


def _hashable_state(value):
    """Normalise instrument state for comparison and hashing."""
    return obj_to_hashable(value)
