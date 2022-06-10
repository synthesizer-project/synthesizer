# This script is a modified version of
# https://github.com/stephenmwilkins/SPS_tools/blob/master/SPS_tools/cloudy/abundances.py

import numpy as np

# Define elements and their abundances
name = {}
name['H']   = 'Hydrogen'
name['He']  = 'Helium'
name['Li']  = 'Lithium'
name['Be']  = 'Beryllium'
name['B']   = 'Boron'
name['C']   = 'Carbon'
name['N']   = 'Nitrogen'
name['O']   = 'Oxygen'
name['F']   = 'Fluorine'
name['Ne']  = 'Neon'
name['Na']  = 'Argon'
name['Mg']  = 'Magnesium'
name['Al']  = 'Aluminium'
name['Si']  = 'Silicon'
name['P']   = 'Phosphorus'
name['S']   = 'Sulphur'
name['Cl']  = 'Chlorine'
name['Ar']  = 'Argon'
name['K']   = 'Potassium'
name['Ca']  = 'Calcium'
name['Sc']  = 'Scandium'
name['Ti']  = 'Titanium'
name['V']   = 'Vanadium'
name['Cr']  = 'Chromium'
name['Mn']  = 'Manganese'
name['Fe']  = 'Iron'
name['Co']  = 'Cobalt'
name['Ni']  = 'Nickel'
name['Cu']  = 'Copper'
name['Zn']  = 'Zinc'

A = {}
A['H']  = 1.008
A['He'] = 4.003
A['Li'] = 6.940
A['Be'] = 9.012
A['B']  = 10.81
A['C']  = 12.011
A['N']  = 14.007
A['O']  = 15.999
A['F']  = 18.998
A['Ne'] = 20.180
A['Na'] = 22.990
A['Mg'] = 24.305
A['Al'] = 26.982
A['Si'] = 28.085
A['P']  = 30.973
A['S']  = 32.06
A['Cl'] = 35.45
A['Ar'] = 39.948
A['K']  = 39.0983
A['Ca'] = 40.078
A['Sc'] = 44.955
A['Ti'] = 47.867
A['V']  = 50.9415
A['Cr'] = 51.9961
A['Mn'] = 54.938
A['Fe'] = 55.845
A['Co'] = 58.933
A['Ni'] = 58.693
A['Cu'] = 63.546
A['Zn'] = 65.38


# Define some of the required elements that should be passed to cloudy,
# the most abundant species and the ones we care more about
all     = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
           'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
           'Cu', 'Zn']
metals  = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
            'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn']

# Elemental abundances
# Asplund (2009) Solar, same as GASS (Grevesse et al. (2010)) in cloudy
# https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract

Z_sol = 0.01316

sol = {}
#These are log10(element/H) ratios
sol['H']    = 0.0
sol['He']   = -1.07
sol['Li']   = -10.95
sol['Be']   = -10.62
sol['B']    = -9.3
sol['C']    = -3.57
sol['N']    = -4.17
sol['O']    = -3.31
sol['F']    = -7.44
sol['Ne']   = -4.07
sol['Na']   = -5.07
sol['Mg']   = -4.40
sol['Al']   = -5.55
sol['Si']   = -4.49
sol['P']    = -6.59
sol['S']    = -4.88
sol['Cl']   = -6.5
sol['Ar']   = -5.60
sol['K']    = -6.97
sol['Ca']   = -5.66
sol['Sc']   = -8.85
sol['Ti']   = -7.05
sol['V']    = -8.07
sol['Cr']   = -6.36
sol['Mn']   = -6.57
sol['Fe']   = -4.50
sol['Co']   = -7.01
sol['Ni']   = -5.78
sol['Cu']   = -7.81
sol['Zn']   = -7.44


# ---------------- Depletion
# --- ADOPTED VALUES
# Gutkin+2016: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract
# Dopita+2013: https://ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstract
# Dopita+2006: https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract

depsol = {}
# Depletion of 1 -> no depletion, while 0 -> fully depleted
# Noble gases aren't depleted (eventhough there is some eveidence
# for Argon depletion -> https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.2310G/abstract)
depsol['H']     = 1.0
depsol['He']    = 1.0
depsol['Li']    = 0.16
depsol['Be']    = 0.6
depsol['B']     = 0.13
depsol['C']     = 0.5
depsol['N']     = 0.89 # <----- replaced by Dopita+2013 value, Gutkin+2016 assumes no depletion
depsol['O']     = 0.7
depsol['F']     = 0.3
depsol['Ne']    = 1.0
depsol['Na']    = 0.25
depsol['Mg']    = 0.2
depsol['Al']    = 0.02
depsol['Si']    = 0.1
depsol['P']     = 0.25
depsol['S']     = 1.0
depsol['Cl']    = 0.5
depsol['Ar']    = 1.0
depsol['K']     = 0.3
depsol['Ca']    = 0.003
depsol['Sc']    = 0.005
depsol['Ti']    = 0.008
depsol['V']     = 0.006
depsol['Cr']    = 0.006
depsol['Mn']    = 0.05
depsol['Fe']    = 0.01
depsol['Co']    = 0.01
depsol['Ni']    = 0.04
depsol['Cu']    = 0.1
depsol['Zn']    = 0.25


def metallicity(elements):
    """
    This function determines the mass fraction of the metals, or the metallicity

    :param elements: a dictionary with the absolute elemental abundances

    :return: A single number
    :rtype: float
    """
    # TODO: rewrite this for improved clarity

    return np.sum([A[i]*10**(elements[i]) for i in metals])/np.sum([A[i]*10**(elements[i]) for i in all])


def abundances(Z, d2m=False, scaling='Dopita+2013'):
    """
    This function returns the elemental abundances after removing the depleted amount

    :param Z: float, the total metallicity (includes depletion as well)
    :param d2m: float(?), dust to metal ratio float
    :param scaling: string

    :return: dictionary with different elemental abundances as log10(element/H)
    :rtype: float
    """
    a = {}

    a['H'] = 0.0

    # New scaling applied to match the He abundance at Z_sol
    a['He'] = np.log10(0.0737 + 0.0114*(Z/Z_sol))

    for i in metals:
        # Scale elemental abundances from solar abundances based on given metallicity
        a[i] = sol[i] + np.log10(Z/metallicity(sol))

    if scaling == 'Dopita+2013':
        # Actually from Dopita+2006
        # Scaling applied to match with our solar metallicity, this done by
        # solving the equation to get the adopted solar metallicity
        Z_sol_Dopita = 0.016
        C_fac = Z_sol/1.01973
        N_fac = Z_sol/1.06774

        a['C'] = np.log10(6e-5 * Z/C_fac + 2e-4 * (Z/C_fac)**2)
        a['N'] = np.log10(1.1e-5 * Z/N_fac + 4.9e-5 * (Z/N_fac)**2)

    elif scaling == 'Wilkins+2020':
        a['N'] = -4.47 + np.log10(Z/Z_sol + (Z/Z_sol)**2)

    # rescale abundances to recover correct Z
    cor = np.log10(Z/metallicity(a))

    for i in metals: a[i] += cor

    if d2m:

        dep = depletions(d2m)

        for i in metals: a[i] += np.log10(dep[i])

    return a


def dust_to_metal(a, dep):

    """
    This function returns the dust-to-metal ratio from the depleted amount of metals

    :param a: a dictionary with the absolute elemental abundances
    :param dep: a dictionary with non-depleted fraction of the metals

    :return: the dust-to-metal ratio
    :rtype: float
    """
    # TODO: rewrite this for improved clarity

    return np.sum([A[i]*(1.-dep[i])*10**a[i] for i in metals])/np.sum([A[i]*10**a[i] for i in metals])


def depletions(d2m):

    """
    This function returns the depletion after scaling using the solar abundances and
    depletion patterns from the dust-to-metal ratio.

    :param d2m: float, dust-to-metal ratio

    :return: a dictionary of depletion patterns
    :rtype: float
    """

    dep = {}

    for i in metals:

        if depsol[i] != 1.0:
            # A very crude interpolation, using no depletion, solar depletion patterns and fully depleted metal
            # corresponding to 0, solar and 1 dust-to-metal ratio
            dep[i] = np.interp(d2m, np.array([0.0, dust_to_metal(sol, depsol), 1.0]), np.array([1.0, depsol[i], 0.0]))
        else:
            dep[i] = 1.0

    return dep
