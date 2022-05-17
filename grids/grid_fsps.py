import numpy as np
import fsps
from utils import write_data_h5py, write_attribute


def grid(Nage=80, NZ=20, zsolar=0.0142):
    """
    Generate grid of spectra with FSPS

    Returns:
        spec (array, float) spectra, dimensions NZ*Nage
        metallicities (array, float) metallicity array, units log10(Z)
        ages (array, flota) age array, units years
        wl (array, float) wavelength array, units Angstroms
    """

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=2, sf_start=0.)

    wl = sp.wavelengths  # units: Angstroms
    ages = sp.log_age  # units: log10(years)
    metallicities = np.log10(sp.zlegend)  # units: log10(Z)

    spec = np.zeros((len(metallicities), len(ages), len(wl)))

    for i, Z in enumerate(metallicities):
        print(i, Z)
        for j, a in enumerate(10**ages / 1e9):
            sp.params['logzsol'] = Z
            spec[i, j] = sp.get_spectrum(tage=a, peraa=True)[1]   # Lsol / AA

    # convert spec units
    spec *= (3.826e33 / 1.1964952e40)  # erg / s / cm^2 / AA

    return spec, ages, metallicities, wl


if __name__ == "__main__":

    Nage = 81
    NZ = 41

    spec, age, Z, wl = grid(Nage=Nage, NZ=NZ)

    fname = 'output/fsps.h5'
    write_data_h5py(fname, 'spectra', data=spec, overwrite=True)
    write_attribute(fname, 'spectra', 'Description',
                    'Three-dimensional spectra grid, [Z,Age,wavelength]')
    write_attribute(fname, 'spectra', 'Units', 'erg s^-1 cm^2 AA^-1')

    write_data_h5py(fname, 'ages', data=age, overwrite=True)
    write_attribute(fname, 'ages', 'Description',
                    'Stellar population ages in log10 years')
    write_attribute(fname, 'ages', 'Units', 'log10(yr)')

    write_data_h5py(fname, 'metallicities', data=Z, overwrite=True)
    write_attribute(fname, 'metallicities', 'Description',
                    'raw abundances in log10')
    write_attribute(fname, 'metallicities', 'Units',
                    'dimensionless [log10(Z)]')

    write_data_h5py(fname, 'wavelength', data=wl, overwrite=True)
    write_attribute(fname, 'wavelength', 'Description',
                    'Wavelength of the spectra grid')
    write_attribute(fname, 'wavelength', 'Units', 'AA')
