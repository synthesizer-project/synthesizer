import h5py
import numpy as np
from synthesizer.sed import calculate_Q



def add_log10Q(filename):


    with h5py.File(filename, 'a') as hf:

        log10metallicities = hf['log10metallicities'][()]
        log10ages = hf['log10ages'][()]

        nZ = len(log10metallicities)
        na = len(log10ages)

        lam = hf['spectra/wavelength'][()]
        if 'log10Q' in hf.keys(): del hf['log10Q'] # delete log10Q if it already exists
        hf['log10Q'] = np.zeros((na, nZ))

        # ---- determine stellar log10Q

        for iZ, log10Z  in enumerate(log10metallicities):
            for ia, log10age in enumerate(log10ages):
                hf['log10Q'][ia, iZ] = np.log10(calculate_Q(lam, hf['spectra/stellar'][ia, iZ, :]))






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
