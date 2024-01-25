# This script will download the test grids from Google Drive.
# The test grid is needed to run the scripts in examples and tests.
#
# The download uses gdown which interfaces with Google Drive to download the
# files. Once downloaded the files can be found in tests/test_grid/ as defined by
# the --output flag passed to gdown.
pip install gdown
gdown https://drive.google.com/uc?id=1FOs5S7rNPJP6NCe_p_BzFQ4Cckx4IKXL --output test_grid/MW3.1.hdf5
