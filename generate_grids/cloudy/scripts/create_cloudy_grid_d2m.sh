#!/bin/bash
# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/"
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
machine="apollo"

sps="bpass-2.2.1-bin_chabrier03-0.1,300.0 "

c=$CLOUDY17

cd ..
# python3 make_cloudy_input_grid.py -dir $synthesizer_dir -m $machine -sps $sps  -p "default.yaml"  -c $c
python3 make_cloudy_input_grid.py -dir $synthesizer_dir -m $machine -sps $sps  -p "no_depletion.yaml"  -c $c
python3 make_cloudy_input_grid.py -dir $synthesizer_dir -m $machine -sps $sps  -p "no_grains.yaml"  -c $c
