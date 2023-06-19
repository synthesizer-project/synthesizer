"""
Read in a full emission line grid from cloudy and produce a linelist of strong lines relative to another line.
"""

import numpy as np
from synthesizer.cloudy import read_lines


output_dir = 'output'
model_name = 'default'

threshold_line = 'H 1 4862.69A' # Hbeta
relative_threshold = 2.5 # log

cloudy_ids, blends, wavelengths, intrinsic, emergent = read_lines(
    f'{output_dir}/{model_name}')

threshold = emergent[cloudy_ids == threshold_line] - relative_threshold

# select line meeting various conditions
s = (emergent > threshold) & (blends == False) & (wavelengths < 50000)


print(f'number of lines: {len(cloudy_ids[s])}')

# save a list of all lines
with open('alllines.dat', 'w') as f:
    f.writelines('\n'.join(cloudy_ids) + '\n')

# save a list of only lines meeting the conditions above
with open('defaultlines.dat', 'w') as f:
    f.writelines('\n'.join(cloudy_ids[s]) + '\n')