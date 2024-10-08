"""A module defining a container for instruments."""

import h5py
from syntheiszer.instruments.instrument import Instrument


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
        for instrument in instruments:
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
