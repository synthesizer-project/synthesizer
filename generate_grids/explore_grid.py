

import h5py


def explore(name, item):
    print(name, item)
    for k, v in item.attrs.items():
        print('    -', k, v)


if __name__ == "__main__":

    grid_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'
    grid_name = 'bpass-2.2.1-sin_chabrier03-0.1,100.0'

    with h5py.File(f'{grid_dir}/{grid_name}.hdf5', 'r') as hf:

        for k, v in hf.attrs.items():
            print('    -', k, v)

        hf.visititems(explore)
