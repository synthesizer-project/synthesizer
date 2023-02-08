import h5py
import numpy as np


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


def explore_hdf5_grid(name, item):
    """
    A simple function for exploring HDF5 grid files.

    NOTE: this should be moved to some kind of utilities.
    TODO: modify to not loop over every line.
    """

    split = name.split('/')
    name_ = '    '*(len(split)-1)+split[-1]
    print(name_, item)

    for k, v in item.attrs.items():
        print('    '*(len(split)-1), k, ':', v)


class Singleton(type):
    """ A metaclass used to ensure singleton behaviour, i.e. there can only
        ever be a single instance of a class in a namespace.

    Adapted from:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    # Define private dictionary to store instances
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """ When a new instance is made (calling class), the original instance
            is returned giving it a new reference to the single insance"""

        # If we don't already have an instance the dictionary will be empty
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]
