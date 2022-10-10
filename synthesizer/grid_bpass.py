import sys
import numpy as np
from glob import glob

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from hoki import load

from utils import write_data_h5py, write_attribute


def main(outfile):
    model_dir = 'BPASSv2.2.1_sin-imf_chab300'
    models = glob(model_dir+'/*')
    
    output_temp = load.model_output(models[0])
    
    print(cosmo.age(0).value)
    ages = np.array([float(a) for a in output_temp.columns[1:]])
    age_mask = (10**ages / 1e9) < cosmo.age(0).value - 0.4 # Gyr
    ages = ages[age_mask]
    # ages_Gyr = ages_Gyr[age_mask]
    
    # scale_factors = cosmo.scale_factor([z_at_value(cosmo.lookback_time, age * u.Gyr) for age in ages_Gyr])
    
    wl = output_temp['WL'].values
    metallicities = np.array([None] * len(models))
    
    spec = np.zeros((len(metallicities),len(ages),len(wl)))
    
    for i,mod in enumerate(models):   
        try:
            metallicities[i] = float('0.'+mod[-7:-4])
        except: # ...catch em# format 
            metallicities[i] = 10**-float(mod[-5])
        
    
    # sort by increasing metallicity
    Z_idx = np.argsort(metallicities)
    metallicities = metallicities[Z_idx].astype(float)
    
    for i,(Z,mod) in enumerate(zip(metallicities,np.array(models)[Z_idx])):
        #print(i,mod,metallicities[i])
    
        output = load.model_output(mod)
    
        for j,a in enumerate(ages):
            #print(j,str(a))
            spec[i,j] = output[str(a)].values 
    
    # convert units
    # metallicities = np.log10(metallicities / 0.0127)  # log(Z / Zsol)
    metallicities = np.log10(metallicities)  # log(Z)
    spec /= 1e6 # Msol
    
    # write_data_h5py(fname,'spec',data=spec,overwrite=True)
    # write_data_h5py(fname,'ages',data=scale_factors,overwrite=True)
    # write_data_h5py(fname,'metallicities',data=metallicities,overwrite=True)
    # write_data_h5py(fname,'wavelength',data=wl,overwrite=True)

    write_data_h5py(outfile, 'spectra', data=spec, overwrite=True)
    write_attribute(outfile, 'spectra', 'Description',
                    'Three-dimensional spectra grid, [Z,Age,wavelength]')
    write_attribute(outfile, 'spectra', 'Units', 'erg s^-1 cm^2 AA^-1')

    write_data_h5py(outfile, 'ages', data=ages, overwrite=True)
    write_attribute(outfile, 'ages', 'Description',
                    'Stellar population ages in log10 years')
    write_attribute(outfile, 'ages', 'Units', 'log10(yr)')

    write_data_h5py(outfile, 'metallicities', data=metals, overwrite=True)
    write_attribute(outfile, 'metallicities', 'Description',
                    'raw abundances in log10')
    write_attribute(outfile, 'metallicities', 'Units',
                    'dimensionless [log10(Z)]')

    write_data_h5py(outfile, 'wavelength', data=wl, overwrite=True)
    write_attribute(outfile, 'wavelength', 'Description',
                    'Wavelength of the spectra grid')
    write_attribute(outfile, 'wavelength', 'Units', 'AA')



# Lets include a way to call this script not via an entry point
if __name__ == "__main__":
    outfile = sys.argv[1]
    main(outfile)

