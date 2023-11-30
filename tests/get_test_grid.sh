# This script will download the test grids from Google Drive.
# The test grid is needed to run the scripts in examples and tests.
#
# The download uses gdown which interfaces with Google Drive to download the
# files. Once downloaded the files can be found in tests/test_grid/ as defined by
# the --output flag passed to gdown.
pip install gdown
gdown https://drive.google.com/uc?id=1XBUG04VUcuzRtQBR40J20RzDmFLu8EeX --output test_grid/test_grid.hdf5
gdown https://drive.google.com/uc?id=1k332yxFwyydlFn4XK-3J3lUTdimboYoN --output test_grid/test_grid_agn-blr.hdf5
gdown https://drive.google.com/uc?id=1XQtJ4lEQUN-8Q_rfJCvcVBB4_3l1tCjU --output test_grid/test_grid_agn-nlr.hdf5
