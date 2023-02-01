#!/bin/bash
# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/"
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
machine="apollo"

# sps="bpass-2.2.1-bin_chabrier03-0.1,100.0 "
# sps="bpass-2.2.1-bin_chabrier03-0.1,100.0 "
# sps="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0 bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35 bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.7 bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0 bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35 bpass-2.2.1-bin_chabrier03-0.1,100.0 bpass-2.2.1-bin_chabrier03-0.1,300.0 bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.0 bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.35 bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.7 bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.0 bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.35 bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7 bpass-2.2.1-sin_chabrier03-0.1,100.0 bpass-2.2.1-sin_chabrier03-0.1,300.0"


# FSPS high-mass slope variants
sps="fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.5 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.6 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.7 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.8 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.9 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.0 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.2 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.3 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.4 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.5 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.6 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.7 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.8 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.9 fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,3.0"

# FSPS high-mass cut off variants
# sps=$sps" fsps-3.2_chabrier03-0.08,120 fsps-3.2_chabrier03-0.08,100 fsps-3.2_chabrier03-0.08,50 fsps-3.2_chabrier03-0.08,20 fsps-3.2_chabrier03-0.08,10 fsps-3.2_chabrier03-0.08,5 fsps-3.2_chabrier03-0.08,2 fsps-3.2_chabrier03-0.08,1"


p="default_param.yaml"
c=$CLOUDY17

cd ..
python3 make_cloudy_input_grid.py -dir $synthesizer_dir -m $machine -sps $sps  -p $p  -c $c
