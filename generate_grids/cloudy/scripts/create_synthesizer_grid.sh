#!/bin/bash
# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/" # SW's machine
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
# grid="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy"
# grid="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy"
# grid="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy" # done

# bpass grids
# grid="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.7_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.7_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy bpass-2.2.1-sin_chabrier03-0.1,100.0 bpass-2.2.1-sin_chabrier03-0.1,300.0_cloudy"

grid=""

grid=$sps" bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy bpass-2.2.1-sin_chabrier03-0.1,100.0_cloudy bpass-2.2.1-sin_chabrier03-0.1,300.0_cloudy"

# bpass alpha variation
grid=$sps" bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpham0.2 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.0 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.2 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.4 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.6 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.8 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha1.0"

# bpass U variation
grid=$sps" bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-log10U_refm1 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-log10U_refm2 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-log10U_refm3 bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-log10U_refm4"

grid=$sps" bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy bpass-2.2.1-sin_chabrier03-0.1,100.0_cloudy bpass-2.2.1-sin_chabrier03-0.1,300.0_cloudy"

# FSPS slope variants
sps=$sps" fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.5_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.6_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.7_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.8_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,1.9_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.0_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.2_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.3_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.4_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.5_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.6_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.7_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.8_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.9_cloudy fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,3.0_cloudy"

# FSPS high-mass cut off
sps=$sps" fsps-3.2_chabrier03-0.08,1_cloudy fsps-3.2_chabrier03-0.08,2_cloudy fsps-3.2_chabrier03-0.08,5_cloudy fsps-3.2_chabrier03-0.08,10_cloudy fsps-3.2_chabrier03-0.08,20_cloudy fsps-3.2_chabrier03-0.08,50_cloudy fsps-3.2_chabrier03-0.08,100_cloudy"

# FSPS high-mass slope variants

cd ..
python3 convert_cloudy_grid_to_hdf5.py -grid $grid -dir $synthesizer_dir
