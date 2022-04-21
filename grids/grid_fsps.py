import numpy as np
import os
import fsps

from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from _utils import write_data_h5py


def grid(Nage=80, NZ=20, nebular=True, dust=False):
    """
    Generate grid of spectra with FSPS

    Returns:
        spec (array, float) spectra, dimensions NZ*Nage
        metallicities (array, float) metallicity array, units Z / Zsol
        scale_factors (array, flota) age array in units of the scale factor
        wl (array, float) wavelength array in Angstroms
    """

    if dust:
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, 
                                    logzsol=0.0, add_neb_emission=nebular,
                                    dust_type=2, dust2=0.2, cloudy_dust=True,  
                                    dust1=0.0)
    else:
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, cloudy_dust=True,
                                    logzsol=0.0, add_neb_emission=nebular)


    wl = np.array(sp.get_spectrum(tage=13, peraa=True)).T[:,0]

    ages = np.logspace(-3.5, np.log10(cosmo.age(0).value-0.4), num=Nage, base=10)

    scale_factors = cosmo.scale_factor([z_at_value(cosmo.lookback_time, age * u.Gyr) for age in ages])
    metallicities = np.linspace(-3, 1, num=NZ)# log(Z / Zsol)

    spec = np.zeros((len(metallicities), len(ages), len(wl)))

    for i, Z in enumerate(metallicities):
        for j, a in enumerate(ages):

            sp.params['logzsol'] = Z
            if nebular: sp.params['gas_logz'] = Z

            spec[i,j] = sp.get_spectrum(tage=a, peraa=True)[1]   # Lsol / AA


    return spec, scale_factors, metallicities, wl 



if __name__ == "__main__":

    Nage = 81 
    NZ = 41 

    spec, age, Z, wl = grid(nebular=False, dust=False, Nage=Nage, NZ=NZ)
    
    fname = 'output/fsps.h5'
    write_data_h5py(fname,'spec',data=spec, overwrite=True)
    write_data_h5py(fname,'ages',data=age, overwrite=True)
    write_data_h5py(fname,'metallicities',data=Z, overwrite=True)
    write_data_h5py(fname,'wavelength',data=wl, overwrite=True)

    spec, age, Z, wl = grid(nebular=True, dust=False, Nage=Nage, NZ=NZ)
    fname = 'output/fsps_neb.h5'
    write_data_h5py(fname,'spec',data=spec, overwrite=True)
    write_data_h5py(fname,'ages',data=age, overwrite=True)
    write_data_h5py(fname,'metallicities',data=Z, overwrite=True)
    write_data_h5py(fname,'wavelength',data=wl, overwrite=True)

