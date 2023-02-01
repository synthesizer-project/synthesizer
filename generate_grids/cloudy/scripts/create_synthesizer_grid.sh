#!/bin/bash
# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/" # SW's machine
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
# grid="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy"
# grid="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy"
grid="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.7_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.7_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy bpass-2.2.1-sin_chabrier03-0.1,100.0 bpass-2.2.1-sin_chabrier03-0.1,300.0_cloudy"
# grid="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy" # failed
grid="bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy" # done
cd ..
python3 convert_cloudy_grid_to_hdf5.py -grid $grid -dir $synthesizer_dir
