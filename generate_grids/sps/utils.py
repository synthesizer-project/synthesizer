import h5py
import numpy as np
from synthesizer.sed import calculate_Q
from synthesizer.cloudy import Ions
from decimal import Decimal


# def add_log10Q(filename):
#     """ add ionising photon luminosity """
#
#     with h5py.File(filename, 'a') as hf:
#
#         metallicities = hf['metallicities'][()]
#         log10ages = hf['log10ages'][()]
#
#         nZ = len(metallicities)
#         na = len(log10ages)
#
#         lam = hf['spectra/wavelength'][()]
#         if 'log10Q' in hf.keys():
#             del hf['log10Q']  # delete log10Q if it already exists
#         hf['log10Q'] = np.zeros((na, nZ))
#
#         # ---- determine stellar log10Q
#
#         for iZ, Z in enumerate(metallicities):
#             for ia, log10age in enumerate(log10ages):
#                 hf['log10Q'][ia, iZ] = np.log10(calculate_Q(lam, hf['spectra/stellar'][ia, iZ, :]))


def add_log10Q(grid_filename, ions=['HI', 'HeII'], limit=100):
    """
    A function to calculate the ionising photon luminosity for different ions.

    Parameters
    ---------
    grid_filename : str
        the filename of the HDF5 grid
    ions : list
        a list of ions to calculate Q for
    limit: float or int, optional
        An upper bound on the number of subintervals 
        used in the integration adaptive algorithm.

    """

    with h5py.File(grid_filename, 'a') as hf:

        metallicities = hf['metallicities'][()]
        log10ages = hf['log10ages'][()]

        nZ = len(metallicities)
        na = len(log10ages)

        lam = hf['spectra/wavelength'][()]

        if 'log10Q' in hf.keys():
            del hf['log10Q']  # delete log10Q if it already exists

        for ion in ions:

            ionisation_energy = Ions.energy[ion]

            hf[f'log10Q/{ion}'] = np.zeros((na, nZ))

            # ---- determine stellar log10Q

            for iZ, Z in enumerate(metallicities):
                for ia, log10age in enumerate(log10ages):
                    # print(ia, iZ)

                    lnu = hf['spectra/stellar'][ia, iZ, :]

                    Q = calculate_Q(lam, lnu, ionisation_energy=ionisation_energy, limit=limit)

                    hf[f'log10Q/{ion}'][ia, iZ] = np.log10(Q)


def get_model_filename(model):

    synthesizer_model_name = f'{model["sps_name"]}'

    if model["sps_version"] != '':
        synthesizer_model_name += f'-{model["sps_version"]}'

    if model["sps_variant"] != '':
        synthesizer_model_name += f'-{model["sps_variant"]}'

    mass_limits_label = ','.join(map(lambda x: str(np.round(x, 2)), model["imf_masses"]))

    synthesizer_model_name += f'_{model["imf_type"]}-{mass_limits_label}'

    if model["imf_type"] == 'bpl':
        imf_slopes_label = ','.join(map(lambda x: str(np.round(x, 2)), model["imf_slopes"]))
        synthesizer_model_name += '-'+imf_slopes_label
    if model["alpha"]:
        synthesizer_model_name += f'_alpha{model["alpha"]}'

    return synthesizer_model_name


def write_data_h5py(filename, name, data, overwrite=False):
    check = check_h5py(filename, name)

    with h5py.File(filename, 'a') as h5file:
        if check:
            if overwrite:
                print('Overwriting data in %s' % name)
                del h5file[name]
                h5file[name] = data
            else:
                raise ValueError('Dataset already exists, ' +
                                 'and `overwrite` not set')
        else:
            h5file.create_dataset(name, data=data)


def check_h5py(filename, obj_str):
    with h5py.File(filename, 'a') as h5file:
        if obj_str not in h5file:
            return False
        else:
            return True


def load_h5py(filename, obj_str):
    with h5py.File(filename, 'a') as h5file:
        dat = np.array(h5file.get(obj_str))
    return dat


def write_attribute(filename, obj, key, value):
    """
    Write attribute to an HDF5 file

    Args
    obj (str) group  or dataset to attach attribute to
    key (str) attribute key string
    value (str) content of the attribute
    """
    with h5py.File(filename, 'a') as h5file:
        dset = h5file[obj]
        dset.attrs[key] = value


def get_names_h5py(filename, group):
    """
    Return list of the names of objects inside a group
    """
    with h5py.File(filename, 'r') as h5file:
        keys = list(h5file[group].keys())

    return keys


def load_arr(name, filename):
    """
    Load Dataset array from file
    """
    with h5py.File(filename, 'r') as f:
        if name not in f:
            raise ValueError("'%s' Dataset doesn't exist..." % name)

        arr = np.array(f.get(name))

    return arr


def read_params(param_file):
    """
    Args:
    param_file (str) location of parameter file

    Returns:
    parameters (object)
    """
    return __import__(param_file)
