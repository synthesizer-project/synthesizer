

import numpy as np
from synthesizer.cloudy import read_lines


synthesizer_data_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/cloudy/'
model = 'bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy'
filename = '0_7'
threshold_line = 'H 1 4862.69A'
relative_threshold = 4.0

cloudy_ids, blends, wavelengths, intrinsic, emergent = read_lines(
    f'{synthesizer_data_dir}/{model}/{filename}')

threshold = emergent[cloudy_ids == threshold_line] - relative_threshold

s = (emergent > threshold) & (blends == False) & (wavelengths < 50000)

print(cloudy_ids[s])
print(len(cloudy_ids[s]))
print(len(set(cloudy_ids[s])))

ion = np.array([''.join(id.split(' ')[:2]) for id in cloudy_ids])
print(ion[s])

# with open('all_lines.dat', 'w') as f:
#     f.writelines('\n'.join(cloudy_ids) + '\n')
#
# with open('default_lines.dat', 'w') as f:
#     f.writelines('\n'.join(cloudy_ids[s]) + '\n')
