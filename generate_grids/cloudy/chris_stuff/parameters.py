"""
Example parameter file
"""

synthesizer_dir = '/cosma7/data/dp004/dc-love2/codes/synthesizer/'

# ---- grid parameters
sps = 'bc03_chabrier03'
grid_dir = '/cosma7/data/dp004/dc-love2/codes/synthesizer_data/'
sps_grid = f'{grid_dir}/grids/{sps}.h5'


# ---- CLOUDY parameters
cloudy_dir = "/cosma7/data/dp004/dc-love2/codes/c17.02/"
cloudy_output_dir = f'{grid_dir}/cloudy_output/'

max_age = 7  # log(yr), maximum age of HII regions

fixed_U = False  # use fixed ionisation parameter
U_target = -2  # if fixed_U=True, log ionisation parameter
index_age = 20
index_Z = 3 

# ---- COSMA parameters
cosma_project = "cosma7"
cosma_account = "dp004"


