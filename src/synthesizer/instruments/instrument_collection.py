"""A module defining a container for instruments.

A InstrumentCollection is a container for Instrument objects. It can be
treated as a dictionary of instruments, with the label of the instrument
as the key. It can also be treated as an iterable, allowing for simple
iteration over the instruments in the collection.

InstrumentCollections can either be created from an existing file or
initialised as an empty collection to which instruments can be added.

Example usage:

    # Create an empty InstrumentCollection
    collection = InstrumentCollection()

    # Add an instrument to the collection
    instrument = Instrument(label="my_instrument", ...)
    collection.add_instruments(instrument)

    # Add multiple instruments to the collection
    instrument1 = Instrument(label="instrument1", ...)
    instrument2 = Instrument(label="instrument2", ...)
    collection.add_instruments(instrument1, instrument2)

    # Save the collection to a file
    collection.save_instruments("path/to/file.hdf5")

    # Load the collection from a file
    collection = InstrumentCollection("path/to/file.hdf5")

    # Iterate over the instruments in the collection
    for instrument in collection:
        print(instrument.label)

    # Get an instrument by its label
    instrument = collection["my_instrument"]

"""

import h5py
from syntesizer import exceptions
from syntheiszer.instruments.instrument import Instrument

from synthesizer.utils.ascii_table import TableFormatter


class InstrumentCollection:
    """
    A container for instruments.

    The InstrumentCollection class is a container for Instrument objects.
    It can be treated as a dictionary of instruments, with the label of the
    instrument as the key. It can also be treated as an iterable, allowing
    for simple iteration over the instruments in the collection.

    InstrumentCollections can either be created from an existing file or
    initialised as an empty collection to which instruments can be added.

    Attributes:
        instruments (dict):
            A dictionary of instruments, with the label of the instrument as
            the key.
        instrument_labels (list):
            A list of the labels of the instruments in the collection.
        ninstruments (int):
            The number of instruments in the collection.
    """

    def __init__(self, filepath=None):
        """
        Initialise the collection ready to collect together instruments.

        Args:
            filepath (str):
                A path to a file containing instruments to load, if desired.
                Otherwise, an empty collection will be created and
                instruments can be added manually.
        """
        # Create the attributes to later be populated with instruments.
        self.instruments = {}
        self.instrument_labels = []

        # Variables to keep track of the current instrument when iterating
        # over the collection
        self._current_ind = 0
        self.ninstruments = 0

        # Load instruments from a file if a path is provided
        if filepath:
            self.load_instruments(filepath)

    def load_instruments(self, filepath):
        """
        Load instruments from a file.

        Args:
            filepath (str):
                The path to the file containing the instruments to load.
        """
        # Open the file
        with h5py.File(filepath, "r") as hdf:
            # Iterate over the groups in the file
            for group in hdf:
                # Create an instrument from the group
                instrument = Instrument.from_hdf5(hdf[group])

                # Add the instrument to the collection
                self.add_instruments(instrument)

    def add_instruments(self, *instruments):
        """
        Add instruments to the collection.

        Args:
            *instruments (Instrument):
                The instruments to add to the collection.
        """
        # Iterate over the instruments to add
        for instrument in instruments:
            # Ensure the object is an Instrument
            if not isinstance(instrument, Instrument):
                raise exceptions.InconsistentArgument(
                    f"Object {instrument} is not an Instrument."
                )

            # Ensure the label doesn't already exist in the Collection
            if instrument.label in self.instruments:
                raise exceptions.DuplicateInstrument(
                    f"Instrument {instrument.label} already exists."
                )

            # Add the instrument to the collection
            self.instruments[instrument.label] = instrument
            self.instrument_labels.append(instrument.label)
            self.ninstruments += 1

    def save_instruments(self, filepath):
        """
        Save the instruments in the collection to a file.

        Args:
            filepath (str):
                The path to the file in which to save the instruments.
        """
        # Open the file
        with h5py.File(filepath, "w") as hdf:
            # Iterate over the instruments in the collection
            for label, instrument in self.instruments.items():
                # Save the instrument to the file
                instrument.save_hdf(hdf, label)

    def __len__(self):
        """Return the number of instruments in the collection."""
        return len(self.instruments)

    def __iter__(self):
        """
        Iterate over the instrument colleciton.

        Overload iteration to allow simple looping over instrument objects,
        combined with __next__ this enables for f in InstrumentCollection
        syntax.
        """
        return self

    def __next__(self):
        """
        Get the next instrument in the collection.

        Overload iteration to allow simple looping over filter objects,
        combined with __iter__ this enables for f in InstrumentCollection
        syntax.

        Returns:
            Instrument
                The next instrument in the collection.
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
        """
        Get an Instrument by its label.

        Enables the extraction of instrument objects from the
        InstrumentCollection by getitem syntax (InstrumentCollection[key]
        rather than InstrumentCollection.instruments[key]).

        Args:
            key (string)
                The label of the desired instrument.

        Returns:
            Instrument
                The Instrument object stored at self.instruments[key].

        Raises:
            KeyError
                When the filter does not exist in self.filters an error is
                raised.
        """
        return self.filters[key]

    def __str__(self):
        """
        Return a string representation of the InstrumentCollection.

        Returns:
            str
                A string representation of the InstrumentCollection.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Instrument Collection")

    def __add__(self, other):
        """
        Combine InstrumentCollections or add an Instrument to this collection.

        Args:
            other (InstrumentCollection/Instrument):
                The InstrumentCollection/Instrument to combine with this one.

        Returns:
            InstrumentCollection:
                The combined InstrumentCollection.
        """
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
