#!/bin/bash
# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/"
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
machine="apollo"
sps="bpass-2.2.1-bin_chabrier03-0.1,300.0" # FLARE default
p="vary_alpha.yaml"
c=$CLOUDY17

cd ..
python3 make_cloudy_input_grid_array.py -dir $synthesizer_dir -m $machine -sps $sps  -p $p  -c $c
