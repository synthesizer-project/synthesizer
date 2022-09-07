"""
Download BPASS v2.2.1 and convert to HDF5 synthesizer grid.
"""

from hoki import load
import numpy as np
from utils import write_data_h5py, write_attribute
import gdown
import tarfile
import h5py
from scipy import integrate

from unyt import h, c
from synthesizer.sed import calculate_Q


synthesizer_data = "/Users/stephenwilkins/Dropbox/Research/data/synthesizer"


model_url = {}
model_url['bpass-v2.2.1_chab100-bin'] = "https://drive.google.com/file/d/1az7_hP3RDovr-BN9sXgDuaYqOZHHUeXD/view?usp=sharing"
model_url['bpass-v2.2.1_chab300-bin'] = "https://drive.google.com/file/d/1JcUM-qyOQD16RdfWjhGKSTwdNfRUW4Xu/view?usp=sharing"
# model_url[''] = ""
# model_url[''] = ""
# model_url[''] = ""
# model_url[''] = ""
# model_url[''] = ""



def download_data(model):

    if model in model_url.keys():
        filename = gdown.download(model_url[model], quiet=False, fuzzy=True)
        return filename
    else:
        print('ERROR: no url for that model')

def untar_data(model):

    model_dir = f'{synthesizer_data}/input_files/bpass/{model}'
    tar = tarfile.open(f'{model}.tar.gz')
    tar.extractall(path = model_dir)
    tar.close()
    os.remove(f'{model}.tar.gz')


def make_grid(model):

    imf = model.split('_')[1]

    model_dir = f'{synthesizer_data}/input_files/bpass/{model}'
    output_name = f'{synthesizer_data}/grids/{model}'

    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {'zem5': 0.00001, 'zem4': 0.0001, 'z001': 0.001, 'z002': 0.002, 'z003': 0.003, 'z004': 0.004, 'z006': 0.006, 'z008': 0.008, 'z010': 0.01, 'z014': 0.014, 'z020': 0.020, 'z030': 0.030, 'z040': 0.040}
    Z_to_Zk = {k:v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    log10Zs = np.log10(Zs)
    print('metallicities', Zs)


    # --- get ages
    fn_ = f'starmass-sin-imf_{imf}.{Z_to_Zk[Zs[0]]}.dat.gz'
    starmass = load.model_output(f'{model_dir}/{fn_}')
    log10ages = starmass['log_age'].values

    # --- get wavelength grid
    fn_ = f'spectra-sin-imf_{imf}.{Z_to_Zk[Zs[0]]}.dat.gz'
    spec = load.model_output(f'{model_dir}/{fn_}')
    wavelengths = spec['WL'].values # \AA
    nu = 3E8/(wavelengths*1E-10)


    nZ = len(log10Zs)
    na = len(log10ages)


    for bs in ['bin','sin']:

        stellar_mass = np.zeros((na,nZ))
        remnant_mass = np.zeros((na,nZ))
        spectra = np.zeros((na, nZ, len(wavelengths)))

        out_name = output_name+'-'+bs+'.h5'

        for iZ, Z in enumerate(Zs):

            # --- get remaining and remnant fraction
            fn_ = f'starmass-{bs}-imf_{imf}.{Z_to_Zk[Z]}.dat.gz'
            starmass = load.model_output(f'{model_dir}/{fn_}')
            stellar_mass[:, iZ] = starmass['stellar_mass'].values/1E6
            remnant_mass[:, iZ] = starmass['remnant_mass'].values/1E6

            # --- get spectra
            fn_ = f'spectra-{bs}-imf_{imf}.{Z_to_Zk[Z]}.dat.gz'
            spec = load.model_output(f'{model_dir}/{fn_}')

            for ia, log10age in enumerate(log10ages):
                spectra[ia, iZ, :] = spec[str(log10age)].values # Lsol AA^-1 10^6 Msol^-1


        spectra /= 1E6 # Lsol AA^-1 Msol^-1
        spectra *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
        spectra *= wavelengths/nu # erg s^-1 Hz^-1 Msol^-1


        write_data_h5py(out_name, 'star_fraction', data=stellar_mass, overwrite=True)
        write_attribute(out_name, 'star_fraction', 'Description',
                        'Two-dimensional remaining stellar fraction grid, [age,Z]')

        write_data_h5py(out_name, 'remnant_fraction', data=remnant_mass, overwrite=True)
        write_attribute(out_name, 'remnant_fraction', 'Description',
                        'Two-dimensional remaining remnant fraction grid, [age,Z]')

        write_data_h5py(out_name, 'spectra/stellar', data=spectra, overwrite=True)
        write_attribute(out_name, 'spectra/stellar', 'Description',
                        'Three-dimensional spectra grid, [Z,Age,wavelength]')
        write_attribute(out_name, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')

        write_data_h5py(out_name, 'log10ages', data=log10ages, overwrite=True)
        write_attribute(out_name, 'log10ages', 'Description',
                'Stellar population ages in log10 years')
        write_attribute(out_name, 'log10ages', 'Units', 'log10(yr)')

        write_data_h5py(out_name, 'metallicities', data=Zs, overwrite=True)
        write_attribute(out_name, 'metallicities', 'Description',
                'raw abundances')
        write_attribute(out_name, 'metallicities', 'Units', 'dimensionless [log10(Z)]')

        write_data_h5py(out_name, 'log10metallicities', data=log10Zs, overwrite=True)
        write_attribute(out_name, 'log10metallicities', 'Description',
                'raw abundances in log10')
        write_attribute(out_name, 'log10metallicities', 'Units', 'dimensionless [log10(Z)]')

        write_data_h5py(out_name, 'spectra/wavelength', data=wavelengths, overwrite=True)
        write_attribute(out_name, 'spectra/wavelength', 'Description',
                'Wavelength of the spectra grid')
        write_attribute(out_name, 'spectra/wavelength', 'Units', 'AA')


def add_log10Q(model):

    for bs in ['sin', 'bin']:

        with h5py.File(f'{synthesizer_data}/grids/{model}-{bs}.h5', 'a') as hf:

            log10metallicities = hf['log10metallicities'][()]
            log10ages = hf['log10ages'][()]

            nZ = len(log10metallicities)
            na = len(log10ages)

            lam = hf['spectra/wavelength'][()]
            if 'log10Q' in hf.keys(): del hf['log10Q'] # delete log10Q if it already exists
            hf['log10Q'] = np.zeros((na, nZ))

            # ---- determine stellar log10Q

            print(f'Calculating log10Q for {model}-{bs}')

            for iZ, log10Z  in enumerate(log10metallicities):
                for ia, log10age in enumerate(log10ages):


                    # Lnu = 1E-7 * hf['spectra/stellar'][ia, iZ, :] # W s^-1 Hz^-1
                    # Llam = Lnu * c / (lam**2*1E-10) # W s^-1 \AA^-1
                    # nlam = (Llam*lam*1E-10)/(h*c) # s^-1 \AA^-1
                    # f = lambda l: np.interp(l, lam, nlam)
                    # n_LyC = integrate.quad(f, 10.0, 912.0)[0]
                    # hf['log10Q'][ia, iZ] = np.log10(n_LyC)

                    hf['log10Q'][ia, iZ] = np.log10(calculate_Q(lam, hf['spectra/stellar'][ia, iZ, :]))

                    # print(iZ, ia, np.log10(n_LyC), np.log10(calculate_Q(lam, hf['spectra/stellar'][ia, iZ, :])))









if __name__ == "__main__":

    models = ['bpass-v2.2.1_chab100']

    for model in models:

        # download_data(model)
        # untar_data(model)
        # make_grid(model)
        add_log10Q(model)
