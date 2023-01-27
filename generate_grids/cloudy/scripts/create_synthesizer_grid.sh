#!/bin/bash
synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/"
machine="apollo"
grid="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy"

cd ..
python3 convert_cloudy_grid_to_hdf5.py -grid $grid -dir $synthesizer_dir
