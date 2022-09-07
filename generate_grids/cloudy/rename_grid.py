

import h5py
import os
import shutil


""" this is code to rename the cloudy output files from my legacy runs """

path_to_sps_grid = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'

sps_model = 'bpass-v2.2.1_chab100-bin'
cloudy_model = 'cloudy-v17.0_logUref-2'

hf = h5py.File(f'{path_to_sps_grid}/{sps_model}.h5','r')



print(hf['metallicities'][()])



for iZ, Z in enumerate(hf['metallicities']):
    for ia, log10age in enumerate(hf['log10ages']):

        for ext in ['cont', 'ovr', 'lines']:

            infile = f'data/{sps_model}_{cloudy_model}/{Z}_{log10age}.{ext}'
            outfile = f'data/{sps_model}_{cloudy_model}/{ia}_{iZ}.{ext}'

            if os.path.isfile(infile):
                exists = 'exists'
                os.rename(infile, outfile)
            else:
                exists = 'does not exist'

            print(ia, iZ, log10age, Z, infile, exists)
