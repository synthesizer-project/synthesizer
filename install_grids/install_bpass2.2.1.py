
from hoki import load
import numpy as np
from utils import write_data_h5py, write_attribute
import gdown
import tarfile


synthesizer_data = "/Users/stephenwilkins/Dropbox/Research/data/synthesizer"


model_url = {}
model_url['bpass_v2.2.1_imf_chab100'] = "https://drive.google.com/file/d/1az7_hP3RDovr-BN9sXgDuaYqOZHHUeXD/view?usp=sharing"



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

    model_dir = f'{synthesizer_data}/input_files/bpass/{model}'
    output_name = f'{synthesizer_data}/grids/{model}'

    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {'zem5': 0.00001, 'zem4': 0.0001, 'z001': 0.001, 'z002': 0.002, 'z003': 0.003, 'z004': 0.004, 'z006': 0.006, 'z008': 0.008, 'z010': 0.01, 'z014': 0.014, 'z020': 0.020, 'z030': 0.030, 'z040': 0.040}
    Z_to_Zk = {k:v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    log10Zs = np.log10(Zs)
    print('metallicities', Zs)


    # --- calculate ages
    fn_ = 'starmass-sin-'+'_'.join(model.split('_')[2:])+f'.{Z_to_Zk[Zs[0]]}.dat.gz'
    starmass = load.model_output(f'{model_dir}/{fn_}')
    print(starmass)
    log10ages = starmass['log_age'].values
    print('log10(age/yr):', log10ages)

    # --- get wavelength grid
    fn_ = 'spectra-sin-'+'_'.join(model.split('_')[2:])+f'.{Z_to_Zk[Zs[0]]}.dat.gz'
    spec = load.model_output(f'{model_dir}/{fn_}')
    wavelengths = spec['WL'].values # \AA
    nu = 3E8/(wavelengths*1E-10)


    for bs in ['bin','sin']:

        stellar_mass = np.zeros((len(log10Zs),len(log10ages)))
        remnant_mass = np.zeros((len(log10Zs),len(log10ages)))
        spectra = np.zeros((len(log10Zs),len(log10ages), len(wavelengths)))

        out_name = output_name+'-'+bs+'.h5'

        for iZ, Z in enumerate(Zs):

            # --- get remaining and remnant fraction
            fn_ = 'starmass-'+bs+'-'+'_'.join(model.split('_')[2:])+f'.{Z_to_Zk[Z]}.dat.gz'
            starmass = load.model_output(f'{model_dir}/{fn_}')
            stellar_mass[iZ, :] = starmass['stellar_mass'].values/1E6
            remnant_mass[iZ, :] = starmass['remnant_mass'].values/1E6

            # --- get spectra
            fn_ = 'spectra-'+bs+'-'+'_'.join(model.split('_')[2:])+f'.{Z_to_Zk[Z]}.dat.gz'
            spec = load.model_output(f'{model_dir}/{fn_}')

            for ia, log10age in enumerate(log10ages):
                spectra[iZ, ia, :] = spec[str(log10age)].values # Lsol AA^-1 10^6 Msol^-1


        spectra /= 1E6 # Lsol AA^-1 Msol^-1
        spectra *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
        spectra *= wavelengths/nu # erg s^-1 Hz^-1 Msol^-1


        write_data_h5py(out_name, 'spectra/stellar', data=spectra, overwrite=True)
        write_attribute(out_name, 'spectra/stellar', 'Description',
                        'Three-dimensional spectra grid, [Z,Age,wavelength]')
        write_attribute(out_name, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')

        write_data_h5py(out_name, 'log10ages', data=log10ages, overwrite=True)
        write_attribute(out_name, 'log10ages', 'Description',
                'Stellar population ages in log10 years')
        write_attribute(out_name, 'log10ages', 'Units', 'log10(yr)')

        write_data_h5py(out_name, 'log10Zs', data=log10Zs, overwrite=True)
        write_attribute(out_name, 'log10Zs', 'Description',
                'raw abundances in log10')
        write_attribute(out_name, 'log10Zs', 'Units', 'dimensionless [log10(Z)]')

        write_data_h5py(out_name, 'wavelength', data=wavelengths, overwrite=True)
        write_attribute(out_name, 'wavelength', 'Description',
                'Wavelength of the spectra grid')
        write_attribute(out_name, 'wavelength', 'Units', 'AA')



if __name__ == "__main__":

    models = ['bpass_v2.2.1_imf_chab100']

    for model in models:

        download_data(model)
        untar_data(model)
        make_grid(model)
