"""
Read in a full emission line grid from cloudy and produce a linelist of strong lines relative to another line.
"""

import numpy as np
from synthesizer.cloudy import read_linelist

# output directory
output_dir = 'output'

# model name
model_name = 'default'

line_ids, wavelengths, luminosities = read_linelist(f'{output_dir}/{model_name}')

print(line_ids)
print(wavelengths)
print(luminosities)