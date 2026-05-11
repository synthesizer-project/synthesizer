"""Container for one or more instrument objects.

An :class:`InstrumentCollection` behaves like a lightweight labelled registry
for instrument instances. It supports dictionary-style lookup by instrument
label, iteration over stored instruments, and HDF5
serialisation/deserialisation.
"""

from copy import deepcopy

import h5py

from synthesizer import exceptions
from synthesizer._version import __version__
from synthesizer.synth_warnings import warn
from synthesizer.utils.ascii_table import TableFormatter


class InstrumentCollection:
    """Container for a set of labelled instruments.

    The collection is primarily used when workflows need to combine multiple
    instruments while preserving their labels and serialised form.

    Attributes:
        instruments (dict): Mapping from instrument label to instrument
            instance.
        instrument_labels (list): Labels stored in insertion order.
        ninstruments (int): Number of instruments currently stored.
        all_filters (FilterCollection or None): Combined filters from all
            photometric instruments in the collection.
    """

    def __init__(self, filepath=None):
        """Initialise the collection.

        Args:
            filepath (str, optional): Path to a file containing instruments to
                load. If omitted, an empty collection is created and
                instruments can be added manually afterwards.
        """
        # Create the attributes to later be populated with instruments.
        self.instruments = {}
        self.instrument_labels = []

        # Create a helper attribute for getting all filters in the collection
        # without having to iterate over the collection
        self.all_filters = None

        # Variables to keep track of the current instrument when iterating
        # over the collection
        self._current_ind = 0
        self.ninstruments = 0

        # Load instruments from a file if a path is provided
        if filepath:
            self.load_instruments(filepath)

    def load_instruments(self, filepath):
        """Load instruments from a file.

        Args:
            filepath (str): Path to the file containing serialised
                instruments. Each top-level group in the file, apart from the
                header group, is interpreted as one serialised instrument.
        """
        # Have to import here to avoid circular imports
        from synthesizer.instruments import Instrument

        # Open the file
        with h5py.File(filepath, "r") as hdf:
            # Warn if the synthesizer versions don't match
            if hdf["Header"].attrs["synthesizer_version"] != __version__:
                warn(
                    "Synthesizer versions differ between the code and "
                    "FilterCollection file! This is probably fine but there "
                    "is no gaurantee it won't cause errors."
                )

            # Iterate over the groups in the file
            for group in hdf:
                # Skip the header group
                if group == "Header":
                    continue

                # Create an instrument from the group
                instrument = Instrument._from_hdf5(hdf[group])

                # Add the instrument to the collection
                self.add_instruments(instrument)

    def add_instruments(self, *instruments):
        """Add instruments to the collection.

        Args:
            *instruments (InstrumentBase): Instruments to add to the
                collection. Each instrument must have a unique label so it can
                be stored unambiguously in the collection mapping.

        Raises:
            InconsistentArguments: If an object is not an instrument or an
                instrument label is duplicated.
        """
        # Have to import here to avoid circular imports
        from synthesizer.instruments import Instrument

        # Iterate over the instruments to add
        for instrument in instruments:
            # Ensure the object is an Instrument
            if not isinstance(instrument, Instrument):
                raise exceptions.InconsistentArguments(
                    f"Object {type(instrument)} is not an Instrument."
                )

            # Ensure the label doesn't already exist in the Collection
            if instrument.label in self.instruments:
                raise exceptions.InconsistentArguments(
                    f"Instrument {instrument.label} already exists."
                )

            # Add the instrument to the collection
            self.instruments[instrument.label] = instrument
            self.instrument_labels.append(instrument.label)
            self.ninstruments += 1

            # Keep a combined filter collection for convenience when the member
            # instruments support photometry
            if instrument.can_do_photometry:
                if self.all_filters is None:
                    self.all_filters = deepcopy(instrument.filters)
                else:
                    self.all_filters += deepcopy(instrument.filters)

    def write_instruments(self, filepath):
        """Save the instruments in the collection to a file.

        Args:
            filepath (str): Path to the file in which to save the instruments.
                The output file will contain a header group followed by one
                group per stored instrument.
        """
        # Open the file
        with h5py.File(filepath, "w") as hdf:
            # Create header group
            head = hdf.create_group("Header")

            # Include the Synthesizer version
            head.attrs["synthesizer_version"] = __version__

            # Include the number of instruments
            head.attrs["ninstruments"] = self.ninstruments

            # Iterate over the instruments in the collection
            for label, instrument in self.instruments.items():
                # Save the instrument to the file
                instrument.to_hdf5(hdf.create_group(label))

    def __len__(self):
        """Return the number of instruments in the collection.

        Returns:
            int: Number of stored instruments.
        """
        return len(self.instruments)

    def __iter__(self):
        """Return the collection iterator.

        Returns:
            InstrumentCollection: Iterator over the stored instruments.
        """
        return self

    def __next__(self):
        """Get the next instrument in the collection.

        Returns:
            InstrumentBase: The next instrument in the collection.
        """
        # Check we haven't finished
        if self._current_ind >= self.ninstruments:
            self._current_ind = 0
            raise StopIteration
        else:
            # Increment index
            self._current_ind += 1

            # Return the instrument
            return self.instruments[
                self.instrument_labels[self._current_ind - 1]
            ]

    def __getitem__(self, key):
        """Get an instrument by its label.

        Args:
            key (str): Label of the desired instrument, matching the
                ``label`` attribute of the stored instrument.

        Returns:
            InstrumentBase: Instrument stored under ``key``.

        Raises:
            KeyError: If the instrument label is not present.
        """
        return self.instruments[key]

    def __str__(self):
        """Return a string representation of the collection.

        Returns:
            str: Tabulated representation of the collection contents.
        """
        # Intialise the table formatter for a compact summary of the stored
        # instruments
        formatter = TableFormatter(self)

        return formatter.get_table("Instrument Collection")

    def __add__(self, other):
        """Add an instrument or another collection to this one.

        Args:
            other (InstrumentCollection or InstrumentBase): Object to combine
                with this collection. Passing another collection appends all of
                its instruments, while passing a single instrument adds just
                that one instrument.

        Returns:
            InstrumentCollection: The updated collection.

        Raises:
            InconsistentAddition: If ``other`` is not an instrument or another
                collection.
        """
        # Have to import here to avoid circular imports
        from synthesizer.instruments import Instrument

        # Ensure other is an InstrumentCollection or Instrument
        if not isinstance(other, (InstrumentCollection, Instrument)):
            raise exceptions.InconsistentAddition(
                f"Cannot combine InstrumentCollection with {type(other)}."
            )

        # Handle addition of InstrumentCollections
        if isinstance(other, InstrumentCollection):
            self.add_instruments(*other.instruments.values())
            return self

        # Otherwise we are adding a single Instrument into the collection
        self.add_instruments(other)

        return self

    def __contains__(self, key):
        """Check if an instrument is in the collection.

        Args:
            key (str): Label of the instrument to check for.

        Returns:
            bool: ``True`` if the instrument is present, otherwise ``False``.
        """
        return key in self.instruments

    def items(self):
        """Get the items in the InstrumentCollection.

        Returns:
            dict_items: Label-instrument pairs stored in the collection. This
                mirrors the behaviour of ``dict.items()`` on the underlying
                mapping.
        """
        return self.instruments.items()

    def to_set(self):
        """Return a set containing the stored instruments.

        Returns:
            set: Set of instrument instances in the collection. This is a
                convenience helper for APIs that expect an unordered collection
                of instruments.
        """
        return {inst for inst in self}
