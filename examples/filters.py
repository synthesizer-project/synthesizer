

from synthesizer.filters import Filter, FilterCollection



# --- define individual filters

filter = Filter('JWST/NIRCam.F200W') # use filter code
filter = Filter(('JWST','NIRCam','F200W')) # use Tuple

# --- define a filter collection

filter_codes = [f'JWST/NIRCam.{f}' for f in ['F200W', 'F277W']] # define a list of filter codes

fc = FilterCollection(filter_codes)

fc.plot_transmission_curves()
