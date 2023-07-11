import numpy as np


def calculate_ew(lam_arr, lnu_arr, index):
    # Define the wavelength and flux arrays
    wavelength = np.array(lam_arr)
    flux = np.array(lnu_arr)
    flux = flux*(wavelength**2)

    # Define the wavelength range of the absorption feature
    absorption_start = index[0]
    absorption_end = index[1]

    # Define the wavelength ranges of the two sets of continuum
    blue_start = index[2]
    blue_end = index[3]

    red_start = index[4]
    red_end = index[5]

    # Compute the average continuum level
    continuum_indices = np.where((wavelength >= absorption_start) & (wavelength <= absorption_end))[0]

    blue_indices = np.where((wavelength >= blue_start) & (wavelength <= blue_end))[0]
    red_indices = np.where((wavelength >= red_start) & (wavelength <= red_end))[0]

    blue_mean = np.mean(flux[blue_indices])
    red_mean = np.mean(flux[red_indices])

    avg_blue = 0.5*(blue_start + blue_end)
    avg_red = 0.5*(red_start + red_end)

    line = np.polyfit([avg_blue, avg_red], [blue_mean, red_mean], 1)

    continuum = (line[0]*wavelength)+line[1]

    # Calculate the equivalent width
    ew = np.trapz((continuum[continuum_indices] - flux[continuum_indices]) / continuum[continuum_indices],
                  wavelength[continuum_indices])

    print("The EW of the absorption feature is:", ew, index)

    return ew











