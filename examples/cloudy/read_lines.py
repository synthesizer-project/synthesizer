

import numpy as np
from synthesizer.cloudy import get_roman_numeral, read_lines, read_all_lines


synthesizer_data_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/cloudy/bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2.0'
filename = '0_7'

# wavelengths, cloudy_line_ids, intrinsic, emergent = np.loadtxt(
#     f'{synthesizer_data_dir}/{filename}.lines', dtype=str, delimiter='\t', usecols=(0, 1, 2, 3)).T
#
# print(len(cloudy_line_ids))


ids, blend, emergent = read_all_lines(f'{synthesizer_data_dir}/{filename}')


print(len(ids))
print(len(emergent))
print(np.sum(blend))

hb = 'HI4861'

print(ids)
print(emergent[ids == hb])

s = (emergent > emergent[ids == hb]-2.5) & (blend == False)

print(np.sum(s))

print(ids[s])
