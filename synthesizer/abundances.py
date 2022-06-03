

import numpy as np


all = ['H','He','C','N','O','Ne','Mg','Si','S','Ar','Ca','Fe']

metals = ['C','N','O','Ne','Mg','Si','S','Ar','Ca','Fe']


name = {}
name['H'] = 'Hydrogen'
name['He'] = 'Helium'
name['Li'] = 'Lithium'
name['Be'] = 'Beryllium'
name['B'] = 'Boron'
name['C'] = 'Carbon'
name['N'] = 'Nitrogen'
name['O'] = 'Oxygen'
name['F'] = 'Fluorine'
name['Ne'] = 'Neon'
name['Na'] = 'Argon'
name['Mg'] = 'Magnesium'
name['Al'] = 'Aluminium'
name['Si'] = 'Silicon'
name['P'] = 'Phosphorus'
name['S'] = 'Sulphur'
name['Cl'] = 'Chlorine'
name['Ar'] = 'Argon'
name['K'] = 'Potassium'
name['Ca'] = 'Calcium'
name['Sc'] = 'Scandium'
name['Ti'] = 'Titanium'
name['V'] = 'Vanadium'
name['Cr'] = 'Chromium'
name['Mn'] = 'Manganese'
name['Fe'] = 'Iron'
name['Co'] = 'Cobalt'
name['Ni'] = 'Nickel'
name['Cu'] = 'Copper'
name['Zn'] = 'Zinc'


A = {}
A['H'] = 1.008
A['He'] = 4.003
A['Li'] = 6.940
A['Be'] = 9.012
A['B'] = 10.81
A['C'] = 12.011
A['N'] = 14.007
A['O'] = 15.999
A['F'] = 18.998
A['Ne'] = 20.180
A['Na'] = 22.990
A['Mg'] = 24.305
A['Al'] = 26.982
A['Si'] = 28.085
A['P'] = 30.973
A['S'] = 32.06
A['Cl'] = 35.45
A['Ar'] = 39.948
A['K'] = 39.0983
A['Ca'] = 40.078
A['Sc'] = 44.955
A['Ti'] = 47.867
A['V'] = 50.9415
A['Cr'] = 51.9961
A['Mn'] = 54.938
A['Fe'] = 55.845
A['Co'] = 58.933
A['Ni'] = 58.693
A['Cu'] = 63.546
A['Zn'] = 65.38

# sol = {}
# sol['H'] = 0.0
# sol['He'] = -1.01
# sol['Li'] = -10.99
# sol['Be'] = -10.63
# sol['B'] = -9.47
# sol['C'] = -3.53
# sol['N'] = -4.32
# sol['O'] = -3.17
# sol['F'] = -7.47
# sol['Ne'] = -4.01
# sol['Na'] = -5.7
# sol['Mg'] = -4.45
# sol['Al'] = -5.56
# sol['Si'] = -4.48
# sol['P'] = -6.57
# sol['S'] = -4.87
# sol['Cl'] = -6.53
# sol['Ar'] = -5.63
# sol['K'] = -6.92
# sol['Ca'] = -5.67
# sol['Sc'] = -8.86
# sol['Ti'] = -7.01
# sol['V'] = -8.03
# sol['Cr'] = -6.36
# sol['Mn'] = -6.64
# sol['Fe'] = -4.51
# sol['Co'] = -7.11
# sol['Ni'] = -5.78
# sol['Cu'] = -7.82
# sol['Zn'] = -7.43


#
# # --- Dopita Solar
#
# sol = {}
# sol['H'] = 0.0
# # sol['He'] = -1.01
# sol['He'] = -1.13
# sol['C'] = -3.59
# sol['N'] = -4.22
# sol['O'] = -3.34
# sol['Ne'] = -3.91
# sol['Mg'] = -4.47
# sol['Si'] = -4.49
# sol['S'] = -4.79
# sol['Ar'] = -5.20
# sol['Ca'] = -5.64
# sol['Fe'] = -4.55

#
# # --- Asplund (2006) Solar
#
# sol = {}
# sol['H'] = 0.0
# sol['He'] = -1.07
# sol['C'] = -3.63
# sol['N'] = -4.22
# sol['O'] = -3.34
# sol['Ne'] = -4.16
# sol['Mg'] = -4.47
# sol['Si'] = -4.49
# sol['S'] = -4.86
# sol['Ar'] = -5.82
# sol['Ca'] = -5.69
# sol['Fe'] = -4.55

# --- Asplund (2005) Solar

sol = {}
sol['H'] = 0.0
sol['He'] = -1.07
sol['C'] = -3.61
sol['N'] = -4.22
sol['O'] = -3.34
sol['Ne'] = -4.16
sol['Mg'] = -4.47
sol['Si'] = -4.49
sol['S'] = -4.86
sol['Ar'] = -5.82
sol['Ca'] = -5.69
sol['Fe'] = -4.55

# --- Asplund (2009) Solar

sol = {}
sol['H'] = 0.0
sol['He'] = -1.07
sol['C'] = -3.57
sol['N'] = -4.17
sol['O'] = -3.31
sol['Ne'] = -4.07
sol['Mg'] = -4.40
sol['Si'] = -4.49
sol['S'] = -4.88
sol['Ar'] = -5.60
sol['Ca'] = -5.66
sol['Fe'] = -4.50


# ---------------- Depletion


# --- dopita values

# depsol = {}
# depsol['H'] = 1. # 1.
# depsol['He'] = 1. # 1.
# depsol['C'] = 0.71 # 0.5
# depsol['N'] = 0.59 # 1.0
# depsol['O'] = 0.62 # 0.7
# depsol['Ne'] = 1.0 # 1.
# depsol['Mg'] = 0.08 # 0.2
# depsol['Si'] = 0.15 # 0.1
# depsol['S'] = 0.83 # 1.0
# depsol['Ar'] = 1.0 # 1.0
# depsol['Ca'] = 0.003 # 0.003
# depsol['Fe'] = 0.05 # 0.01


# --- cloudy values

# depsol = {}
# depsol['H'] = 1.
# depsol['He'] = 1.
# depsol['Li'] = 0.16
# depsol['Be'] = 0.6
# depsol['B'] = 0.13
# depsol['C'] = 0.5
# depsol['N'] = 1.0
# depsol['O'] = 0.7
# depsol['F'] = 0.3
# depsol['Ne'] = 1.0
# depsol['Na'] = 0.25
# depsol['Mg'] = 0.2
# depsol['Al'] = 0.02
# depsol['Si'] = 0.1
# depsol['P'] = 0.25
# depsol['S'] = 1.0
# depsol['Cl'] = 0.5
# depsol['Ar'] = 1.0
# depsol['K'] = 0.3
# depsol['Ca'] = 0.003
# depsol['Sc'] = 0.005
# depsol['Ti'] = 0.008
# depsol['V'] = 0.006
# depsol['Cr'] = 0.006
# depsol['Mn'] = 0.05
# depsol['Fe'] = 0.01
# depsol['Co'] = 0.01
# depsol['Ni'] = 0.04
# depsol['Cu'] = 0.1
# depsol['Zn'] = 0.25



# --- ADOPTED VALUES

depsol = {}
depsol['H'] = 1.
depsol['He'] = 1.
depsol['C'] = 0.5
depsol['N'] = 0.59 # <----- replaced by Dopita value
depsol['O'] = 0.7
depsol['Ne'] = 1.0
depsol['Mg'] = 0.2
depsol['Si'] = 0.1
depsol['S'] = 1.0
depsol['Ar'] = 1.0
depsol['Ca'] = 0.003
depsol['Fe'] = 0.01




def metallicity(a):

    # --- determine mass fraction

    return np.sum([A[i]*10**(a[i]) for i in metals])/np.sum([A[i]*10**(a[i]) for i in all])



def abundances(Z, CO = 0.0, d2m = False):

    Z_sol = 0.01316 # Asplund2009 for just these elements

    a = {}

    a['H'] = 0.0
    # a['He'] = sol['He']

    # a['He'] = np.log10(0.0737 + 0.024*(Z/Z_sol)) # Dopita 2006

    a['He'] = np.log10(0.0737 + 0.0114*(Z/Z_sol)) # Mine

    for i in metals:
        a[i] = sol[i] + np.log10(Z/metallicity(sol))

    # a['N'] = np.log10(0.41 * 10**a['O'] * (10**-1.6 + 10**(2.33 + a['O'])))

    # a['N'] = 7.6 + np.log10(Z/Z_sol + (Z/Z_sol)**2) - 12.

    a['N'] = -4.47 + np.log10(Z/Z_sol + (Z/Z_sol)**2)

    a['C'] = a['C'] + CO

    cor = np.log10(Z/metallicity(a)) # --- rescale abundances to recover correct Z

    for i in metals: a[i] += cor

    if d2m:

        dep = depletions(d2m)

        for i in metals: a[i] += np.log10(dep[i])

    return a


def mf(Z):

    Z_sol = 0.01316 # Asplund2009 for just these elements

#     Y = 0.2485 + 1.7756*Z # maybe this should change
#
#     X = 1 - Y - Z

    b = 0.0737 + 0.0114*(Z/Z_sol)

    X = (1.-Z)/(1.+b*A['He']/A['H'])

    Y = 1 - X - Z

    return X,Y,Z


def dust_to_metal(a, dep):

    return np.sum([A[i]*(1.-dep[i])*10**a[i] for i in metals])/np.sum([A[i]*10**a[i] for i in metals])


def depletions(d2m):

    dep = {}

    for i in metals:

        if depsol[i] != 1.0:
            dep[i] = np.interp(d2m, np.array([0.0, dust_to_metal(sol, depsol), 1.0]), np.array([1.0, depsol[i], 0.0]))
        else:
            dep[i] = 1.0

    return dep
