"""
Generate parametric SFZH
========================

Example for generating a parametric star formation and metal enrichment history
- shows how to generate star formation histories assuming diffferent parameterisations
- shows how to combine star formation histories
"""

import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_sfzh_from_array
# from synthesizer.plt import single, single_histxy, mlabel

# TODO: SFH currently reliant on sfzh to get binned history
# TODO: need new binning method for SFH class


if __name__ == '__main__':

    # --- define a age and metallicity grid. In practice these are pulled from the SPS model.
    log10ages = np.arange(6., 10.5, 0.1)
    log10metallicities = np.arange(-5., -1.5, 0.25)
    metallicities = 10**log10metallicities

    # --- define the SFH as an array and the metallicity as a number

    sfh = np.ones(len(log10ages))
    Z = 0.01
    sfzh = generate_sfzh_from_array(log10ages, metallicities, sfh, Z)
    print(sfzh)
    sfzh.plot()

    # --- define the parameters of the star formation and metal enrichment histories

    Z_p = {'log10Z': -2.5}  # can also use linear metallicity e.g. {'Z': 0.01}
    Zh = ZH.deltaConstant(Z_p)
    print(Zh)  # print summary

    # --- make a constant SFH

    sfh_p = {'duration': 100 * Myr}
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print summary

    constant = generate_sfzh(log10ages, metallicities, sfh, Zh)
    print(constant)  # print summary of the star formation history
    constant.plot()

    # --- make an exponential SFH

    sfh_p = {'tau': 100 * Myr, 'max_age': 200 * Myr}
    sfh = SFH.TruncatedExponential(sfh_p)  # constant star formation
    print(sfh)  # print summary of the star formation history

    exponential = generate_sfzh(log10ages, metallicities, sfh, Zh)
    print(exponential)  # print summary of the star formation history
    exponential.plot()

    # --- add them together

    combined = constant + exponential
    print(combined)  # print summary of the star formation history
    combined.plot()
