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
                raise ValueError(
                    'Dataset already exists, and `overwrite` not set')
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


def load_arr(name, filename):
    """
    Load Dataset array from file
    """

    with h5py.File(filename, 'r') as f:

        if name not in f:
            raise ValueError("'%s' Dataset doesn't exist..." % name)

        arr = np.array(f.get(name))

    return arr


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
