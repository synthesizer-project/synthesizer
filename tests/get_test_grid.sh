# This script will download the test grid from Google Drive.
# The test grid is needed to run the scripts in examples and tests.
#
# The download uses gdown which interfaces with Google Drive to download the
# file. Once downloaded the file can be found in tests/test_grid/ as defined by
# the --output flag passed to gdown.
pip install gdown
gdown https://drive.google.com/uc?id=1txY9PY-ejtRaMSZorpZSxGPsm6muvh9X --output tests/test_grid/test_grid.hdf5
