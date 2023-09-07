
INPUT_DIR="/Users/sw376/Dropbox/Research/data/synthesizer/grids"
INPUT_GRID="bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
OUTPUT_DIR="../tests/test_grid/"
OUTPUT_GRID="test_grid"

python rebin_grid.py -input_dir $INPUT_DIR -input_grid $INPUT_GRID -output_dir $OUTPUT_DIR -output_grid $OUTPUT_GRID