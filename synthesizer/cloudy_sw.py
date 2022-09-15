"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import numpy as np
from scipy import integrate

from unyt import c, h, angstrom






def create_cloudy_input(model_name, lam, lnu, abundances, log10U, output_dir = '', **kwargs):

    params = {'log10radius': -2, # radius in log10 parsecs
              'covering_factor': 1.0, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
              'stop_T': 4000, # K
              'stop_efrac': -2,
              'T_floor': 100, # K
              'log10n_H': 2.5, # Hydrogen density
              'z': 0.,
              'CMB': False,
              'cosmic_rays': False
            }

    for key, value in list(kwargs.items()):
        params[key] = value




    nu = c/(lam * angstrom)
    E = h*nu
    E_Ryd = E.to('Ry').value

    np.savetxt(f'{output_dir}/{model_name}.sed', np.array([E_Ryd[::-1], lnu[::-1]]).T)

    # ----- start CLOUDY input file (as a list)
    cinput = []

    cinput.append(f'table SED "{model_name}.sed" \n')

    # --- Define the chemical composition
    for ele in ['He'] + abundances.metals:
        cinput.append('element abundance '+abundances.name[ele]+' '+str(abundances.a[ele])+'\n')


    # --- add graphite and silicate grains
    # This old version does not actually conserve mass as the command abundances
    # and grains do not really talk with each other
    # graphite, scale by total C abundance relative to ISM
    # scale = 10**a_nodep['C']/2.784000e-04
    # cinput.append(f'grains Orion graphite {scale}'+'\n')
    # # silicate, scale by total Si abundance relative to ISM NOTE: silicates also rely on other elements.
    # scale = 10**a_nodep['Si']/3.280000e-05
    # cinput.append(f'grains Orion silicate {scale}'+'\n')

    """ specifies graphitic and silicate grains with a size distribution and abundance
    appropriate for those along the line of sight to the Trapezium stars in Orion. The Orion size
    distribution is deficient in small particles and so produces the relatively grey extinction
    observed in Orion (Baldwin et al., 1991).
    One problem with the grains approach is metals/element abundances do not talk to the grains command
    and hence there is issues with mass conservation (see cloudy documentation). To alleviate this one
    needs to make the orion grain abundances consistent with the depletion values. Assume 1 per cent of
    C is in PAH's.
    PAHs appear to exist mainly at the interface between the H+ region and the molecular clouds.
    Apparently PAHs are destroyed in ionized gas (Sellgren et al., 1990, AGN3 section 8.5) by
    ionizing photons and by collisions with ions (mainly H+ ) and may be depleted into larger grains
    in molecular regions. Also assume the carbon fraction of PAHs from Abel+2008
    (https://iopscience.iop.org/article/10.1086/591505) assuming 1 percent of Carbon in PAHs.
    The factors in the denominators are the abundances of the carbon, silicon and PAH
    fractions when setting a value of 1 (fiducial abundance) for the orion and PAH grains.
    Another way is to scale the abundance as a function of the metallicity using the Z_PAH vs Z_gas relation
    from Galliano+2008 (https://iopscience.iop.org/article/10.1086/523621, y = 4.17*Z_gas_sol - 7.085),
    which will again introduce issues on mass conservation."""


    if abundances.params['d2m']>0:
        delta_C         = 10**abundances.a_nodep['C'] - 10**abundances.a['C']
        delta_PAH       = 0.01 * (10**abundances.a_nodep['C'])
        delta_graphite  = delta_C - delta_PAH
        delta_Si        = 10**abundances.a_nodep['Si'] - 10**abundances.a['Si']
        orion_C_abund   = -3.6259
        orion_Si_abund  = -4.5547
        PAH_abund       = -4.446
        f_graphite  = delta_graphite/(10**(orion_C_abund))
        f_Si        = delta_Si/(10**(orion_Si_abund))
        f_pah       = delta_PAH/(10**(PAH_abund))
        command = F'grains Orion graphite {f_graphite} \ngrains Orion silicate {f_Si} \ngrains PAH {f_pah}'
        cinput.append(command+'\n')
    else:
        f_graphite, f_Si, f_pah = 0, 0, 0

    # cinput.append('element off limit -7') # should speed up the code

    # # --- Define the ionising luminosity
    # # log10Q = np.log10(calculate_Q(10**log10U, R_inner=10**params['log10radius'] * 3.086e18))
    log10Q = np.log10(calculate_Q_from_U(10**log10U, 10**params["log10n_H"]))
    cinput.append(f'Q(H) = {log10Q}\n')
    # # cinput.append(f'ionization parameter = {log10U} log\n')

    # add background continuum
    if params['cosmic_rays']:
        cinput.append(f'cosmic rays, background\n')
    if params['CMB']:
        cinput.append(f'CMB {params["z"]}\n')

    # --- Define the geometry
    cinput.append(f'hden {params["log10n_H"]} log constant density\n')
    cinput.append(f'radius {params["log10radius"]} log parsecs\n')
    cinput.append(f'sphere\n')
    cinput.append(f'covering factor {params["covering_factor"]} linear\n')

    # --- Processing commands
    cinput.append(f'iterate to convergence\n')
    cinput.append(f'set temperature floor {params["T_floor"]} linear\n')
    cinput.append(f'stop temperature {params["stop_T"]}K\n')
    cinput.append(f'stop efrac {params["stop_efrac"]}\n')

    # --- output commands
    cinput.append(f'save last continuum "{model_name}.cont" units Angstroms no clobber\n')
    cinput.append(f'save last lines, array "{model_name}.lines" units Angstroms no clobber\n')
    cinput.append(f'save overview  "{model_name}.ovr" last\n')

    # --- save input file
    open(f'{output_dir}/{model_name}.in','w').writelines(cinput)

    return cinput



def calculate_Q_from_U(U_avg, n_h):
    """
    Args
    U - units: dimensionless
    n_h - units: cm^-3

    Returns
    Q - units: s^-1
    """
    alpha_B = 2.59e-13 # cm^3 s^-1
    c_cm = 2.99e8 * 100 # cm s^-1
    epsilon = 1.

    return ((U_avg * c_cm)**3 / alpha_B**2) *\
            ((4 * np.pi) / (3 * epsilon**2 * n_h))







def read_lines(output_dir, output_file, lines = []):


    log10M_cluster = 8.

    lam, cID, intrinsic, emergent = np.loadtxt(f'{output_dir}/{output_file}.lines', dtype = str, delimiter='\t', usecols = (0,1,2,3)).T

    lam = lam.astype(float)
    intrinsic = intrinsic.astype(float)
    emergent = emergent.astype(float)

    Lam = []
    ID = []
    Intrinsic = []
    Emergent = []


    for l,id, intrin, emerg in zip(lam, cID, intrinsic, emergent):

        l = int(np.round(l,0))

        li = list(filter(None, id.split(' ')))

        e = li[0]

        i = li[1]
        j = '-'
        if i == '1': j = 'I'
        if i == '2': j = 'II'
        if i == 'II': j = 'II'
        if i == '3': j = 'III'
        if i == '4': j = 'IV'
        if i == '5': j = 'V'
        if i == '6': j = 'VI'
        if i == '7': j = 'VII'
        if i == '8': j = 'VIII'
        if i == '9': j = 'IX'
        if i == '10': j = 'X'
        if i == '11': j = 'XI'
        if i == '12': j = 'XII'
        if i == '13': j = 'XIII'
        if i == '14': j = 'XIV'


        nid = e+j+str(l)

        Lam.append(l)
        ID.append(nid)
        Emergent.append(emerg)
        Intrinsic.append(intrin)


    Lam = np.array(Lam)
    ID = np.array(ID)
    Emergent = np.array(Emergent)
    Intrinsic = np.array(Intrinsic)

    n_Lam = []
    n_emergent = []
    n_intrinsic = []

    for line in lines:

        if line in ID:

            n_emergent.append(Emergent[(ID==line)][0])
            n_intrinsic.append(Intrinsic[(ID==line)][0])
            n_Lam.append(Lam[(ID==line)][0])

        else:

            n_emergent.append(-99.)
            n_intrinsic.append(-99.)
            n_Lam.append(1000.)


    n_Lam = np.array(n_Lam)
    n_emergent = np.array(n_emergent) - log10M_cluster # correct for size of cluster # erg s^-1
    n_intrinsic = np.array(n_intrinsic) - log10M_cluster # correct for size of cluster # erg s^-1


    return n_Lam, n_intrinsic, n_emergent




def read_wavelength(filename):

    """ return just wavelength grid from cloudy file and reverse the order """

    lam = np.loadtxt(f'{filename}.cont', delimiter='\t', usecols = (0)).T
    lam = lam[::-1]  # reverse order

    return lam

def read_continuum(filename, return_dict = False):

    """ read a cloudy continuum file and convert spectra to erg/s/Hz """

    # ----- Open SED

    # 1 = incident, 2 = transmitted, 3 = nebular, 4 = total, 8 = contribution of lines to total
    lam, incident, transmitted, nebular, total, linecont  = np.loadtxt(f'{filename}.cont', delimiter='\t', usecols = (0,1,2,3,4,8)).T

    # --- frequency
    lam = lam[::-1] # reverse array
    lam_m = lam * 1E-10 # m
    nu = c.value / (lam_m)

    # --- nebular continuum is the total nebular emission (nebular) - the line continuum (linecont)
    nebular_continuum = nebular - linecont

    spec_dict = {'lam': lam, 'nu': nu}


    for spec_type in ['incident', 'transmitted', 'nebular', 'nebular_continuum', 'total', 'linecont']:

        sed = locals()[spec_type]
        sed = sed[::-1]  # reverse array
        sed /= 10**7  # convert from W to erg
        sed /= nu  # convert from nu l_nu to l_nu
        spec_dict[spec_type] = sed


    if return_dict:
        return spec_dict
    else:
        return lam, nu, incident, transmitted, nebular, nebular_continuum, total, linecont
