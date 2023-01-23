from synthesizer.filters import FilterFromSVO, FilterCollection,\
    TopHatFilterCollection, UVJ

# define individual filters
filt = FilterFromSVO('JWST/NIRCam.F200W')  # use filter code
filt = FilterFromSVO(('JWST', 'NIRCam', 'F200W'))  # use Tuple

# define a filter collection
fs = [f'JWST/NIRCam.{f}' for f in ['F200W', 'F277W']]  # a list of filter codes
fc = FilterCollection(fs)
fc.plot_transmission_curves()

# define a filter collection
fs = [('U', {'lam_eff': 3650, 'lam_fwhm': 660})]
fc = TopHatFilterCollection(fs)
fc.plot_transmission_curves()

# pre-set UVJ filter collection
fc = UVJ()
fc.plot_transmission_curves()
