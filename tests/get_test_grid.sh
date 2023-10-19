# This script will download the test grids from Google Drive.
# The test grid is needed to run the scripts in examples and tests.
#
# The download uses gdown which interfaces with Google Drive to download the
# files. Once downloaded the files can be found in tests/test_grid/ as defined by
# the --output flag passed to gdown.
pip install gdown
gdown https://drive.google.com/uc?id=1txY9PY-ejtRaMSZorpZSxGPsm6muvh9X --output tests/test_grid/test_grid.hdf5
gdown https://drive.google.com/uc?id=1WkQQ1s3uh9lmx1Ye6RlM9qhrxcWp5N87 --output tests/test_grid/test_grid_blackholes.hdf5
