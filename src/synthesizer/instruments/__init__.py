from synthesizer.instruments.filters import UVJ, Filter, FilterCollection

# We have to import InstrumentCollection here before Instrument to avoid
# circular imports
from synthesizer.instruments.instrument_collection import InstrumentCollection

# Now we can import Instrument
from synthesizer.instruments.instrument import Instrument



__all__ = [
    "Instrument",
    "InstrumentCollection",
    "UVJ",
    "Filter",
    "FilterCollection",
]
