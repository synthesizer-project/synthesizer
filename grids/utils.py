import h5py
import numpy as np

def write_data_h5py(filename, name, data, overwrite=False):
    check = check_h5py(filename, name)

    with h5py.File(filename, 'a') as h5file:
        if check:
            if overwrite:
                print('Overwriting data in %s'%name)
                del h5file[name]
                h5file[name] = data
            else:
                raise ValueError('Dataset already exists, and `overwrite` not set')
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

    with h5py.File(filename, 'a') as h5file:
        dset = h5file[obj]
        dset.attrs[key] = value


