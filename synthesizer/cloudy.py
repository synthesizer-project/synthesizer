"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import subprocess

import numpy as np
from scipy import integrate

from .abundances import abundances

c = 2.9979E8


def create_cloudy_binary(grid, params, verbose=False):
    """
    Args:

    grid: synthesizer _grid_ object
    """
    ages = 10**grid.ages
    ages[ages == 0.0] = 1.
    ages = np.log10(ages)

    # test

    # # ---- TEMP check for negative values and amend
    # # The BPASS binary sed has a couple of erroneous negative values, possibly due to interpolation errors
    # # Here we set the Flux to the average of each neighbouring wavelength value
    #
    # mask = np.asarray(np.where(sed < 0.))
    # for i in range(mask.shape[1]):
    #     sed[mask[0,i],mask[1,i],mask[2,i]] = sed[mask[0,i],mask[1,i],mask[2,i]-1]+sed[mask[0,i],mask[1,i],mask[2,i]+1]/2

    if verbose: print('Writing .ascii')

    output = []
    output.append("20060612\n")  # magic number
    output.append("2\n")  # ndim
    output.append("2\n")  # npar

    ## First parameter MUST be log otherwise Cloudy throws a tantrum
    output.append("age\n")  # label par 1
    output.append("logz\n")  # label par 2

    output.append(str(grid.spectra.shape[0] * grid.spectra.shape[1])+"\n")  # nmod
    output.append(str(len(grid.wl))+"\n")  # nfreq (nwavelength)
    # output.append(str(len(frequency))+"\n")  # nfreq (nwavelength)

    output.append("lambda\n")  # type of independent variable (nu or lambda)
    output.append("1.0\n")  # conversion factor for independent variable
    # output.append("F_nu\n")  # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
    # output.append("3.839e33\n")  # conversion factor for dependent variable
    output.append("F_lambda\n")  # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
    output.append("1.0\n")  # conversion factor for dependent variable

    for z in grid.metallicities:
        for a in ages:  # available SED ages
            output.append(str(a)+' '+str(z)+"\n")  # (npar x nmod) parameters

    # output.append(' '.join(map(str,frequency))+"\n")  # the frequency(wavelength) grid, nfreq points
    output.append(' '.join(map(str,grid.wl))+"\n")  # the frequency(wavelength) grid, nfreq points


    for i,z in enumerate(grid.metallicities):
        for j,a in enumerate(ages):
            output.append(' '.join(map(str,grid.spectra[i,j]))+"\n")


    target = open('model.ascii','w')
    target.writelines(output)
    target.close()

    # ---- compile ascii file
    print('Compiling Cloudy atmosphere file (.ascii)')
    subprocess.call(f'echo -e \'compile stars \"model.ascii\"\' | {params.cloudy_dir}/source/cloudy.exe', shell=True)

    # ---- copy .mod file to cloudy data directory
    print(f'Copying compiled atmosphere to Cloudy directory, {params.cloudy_dir}')
    subprocess.call(f'cp model.mod {params.cloudy_dir}/data/.', shell=True)

    # ---- remove .ascii file
    # os.remove(out_dir+model+'.ascii')

    
# def calculate_log10Q(grid, ia, iZ, **kwargs): 
#     
#     params = {'log10n_H': 2.5, # Hydrogen density
#               'log10U_S_0': -2.0, # ionisation parameter at the reference metallicity and age 
#               ## NOTE: this is not how other people handle this but I think it makes more sense
#               'REF_log10Z': -2.0,
#               'REF_log10age': 6.0
#              }
#     
#     for key, value in list(kwargs.items()):
#         params[key] = value
# 
#     # --- determine the metallicity and age indicies for the reference metallicity and age
#     iZ_REF = (np.abs(grid.metallicities - (params['REF_log10Z']))).argmin()
#     ia_REF = (np.abs(grid.ages - 10**(params['REF_log10age']))).argmin()
# 
#     log10Q_REF = measure_log10Q(grid.wl, grid.spectra[iZ_REF, ia_REF])
# 
#     # --- calculate the ionising photon luminosity for the target SED
#     log10Q_orig = measure_log10Q(grid.wl, grid.spectra[iZ, ia])
# 
#     # --- calculate the actual ionisation parameter for the target SED. Only when the target SED == reference SED will log10U_S = log10U_S_0
#     log10U_S = params['log10U_S_0'] + (log10Q_orig - log10Q_REF)/3.
# 
#     # --- now determine the actual ionising photon luminosity for the target SED. This is the normalising input to CLOUDY
#     log10Q = determine_log10Q(log10U_S, params['log10n_H'])
# 
#     return log10Q


def write_cloudy_input(model_name, grid, ia, iZ, log10U, output_dir='grids/cloudy_output/', **kwargs):

    params = {'log10radius': -2, # radius in log10 parsecs
              'covering_factor': 1.0, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
              'stop_T': 4000, # K
              'stop_efrac': -2,
              'T_floor': 100, # K          
              'log10n_H': 2.5, # Hydrogen density
              'log10U_S_0': -2.0, # ionisation parameter at the reference metallicity and age ## NOTE: this is not how other people handle this but I think it makes more sense
              'CO': 0.0, #
              'd2m': 0.3, # dust-to-metal ratio
              'z': 0.,
              'CMB': False,
              'cosmic_rays': False
            }

    for key, value in list(kwargs.items()):
        params[key] = value

    # --- convert the SED to the CLOUDY format. The normalisation doesn't matter as that is handled by log10Q above
    # CLOUDY_SED = create_CLOUDY_SED(grid.wl, grid.spectra[iZ, ia])

    # --- get metallicity
    Z = 10**grid.metallicities[iZ]

    # ---- initialise abundances object
    abund = abundances()

    # --- determine elemental abundances for given Z, CO, d2m, with depletion taken into account
    a = abund.abundances(Z, params['CO'], params['d2m'])

    # --- determine elemental abundances for given Z, CO, d2m, WITHOUT depletion taken into account
    a_nodep =  abund.abundances(Z, params['CO'], 0.0) # --- determine abundances for no depletion


    # ----- start CLOUDY input file (as a list)
    cinput = []

    # --- Define the incident radiation field shape
    # cinput.append('interpolate{      10.0000     -99.0000}\n')
    # for i1 in range(int(len(CLOUDY_SED)/5)): cinput.append('continue'+''.join(CLOUDY_SED[i1*5:(i1+1)*5])+'\n')
    # cinput.append('print line precision 6\n')
    # cinput.append('table star "model.mod" '+str(grid.ages[ia])+' '+str(grid.metallicity[iZ])+'\n')
    # cinput.append(f'table star "model.mod" {grid.metallicity[iZ]} {grid.ages[ia]}\n')
    cinput.append(f'table star "model.mod" {grid.ages[ia]} {grid.metallicities[iZ]}\n')

    # --- Define the chemical composition
    for ele in ['He'] + abund.metals:
        cinput.append('element abundance '+abund.name[ele]+' '+str(a[ele])+'\n')


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

    delta_C         = 10**a_nodep['C'] - 10**a['C']
    delta_PAH       = 0.01 * (10**a_nodep['C'])
    delta_graphite  = delta_C - delta_PAH
    delta_Si        = 10**a_nodep['Si'] - 10**a['Si']
    orion_C_abund   = -3.6259
    orion_Si_abund  = -4.5547
    PAH_abund       = -4.446
    if params['d2m']>0:
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
    log10Q = np.log10(calculate_Q(10**log10U))
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

    # --- define output filename
    cinput.append(f'save last continuum "{model_name}.cont" units Angstroms no clobber\n')
    cinput.append(f'save last lines, array "{model_name}.lines" units Angstroms no clobber\n')
    # cinput.append(f'save overview "model.ovr" last\n')

    # --- write input file
    open(f'{output_dir}/{model_name}.in','w').writelines(cinput)


    # --- make a dictionary of the parameters including derived parameters

    # derived_parameter = ['log10Q_orig', 'log10U_S', 'log10Q', 'Z']

    # params = {}
    # # for name in cloudy_parameters + derived_parameter:
    # for name in derived_parameter:

    #     if type(locals()[name]) != int:
    #         if type(locals()[name]) != str:
    #             params[name] = float(np.round(locals()[name],2))
    #     else:
    #         params[name] = locals()[name]



    # # --- write parameter file, including derived parameters
    # # yaml.dump({**params, **self.SPS_params}, open(f'params.yaml','w'))
    # yaml.dump({**params}, open(f'params.yaml','w'))

    return cinput


# def calculate_Q(U_0, R_inner=0.01 * 3.086e18, n_h=100):
#     """
#     Q = U_0 * 4 * pi * R_inner^2 * n_H * c
# 
#     U - units: dimensionless
#     R_inner - units: cm
#     n_h - units: cm^-3
# 
#     returns Q - units: s^-1
#     """
#     return U_0 * 4 * np.pi * R_inner**2 * n_h * 2.99e10

def calculate_Q(U_avg, n_h=100):
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


def calculate_U(Q_avg, n_h=100):
    """
    Args
    Q - units: s^-1
    n_h - units: cm^-3

    Returns
    U - units: dimensionless
    """
    alpha_B = 2.59e-13 # cm^3 s^-1
    c_cm = 2.99e8 * 100 # cm s^-1
    epsilon = 1.

    return ((alpha_B**(2./3)) / c_cm) *\
            ((3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi))**(1./3)


def measure_Q(lam, L_AA, limit=100):
    """
    Args
    lam: \AA
    L_AA: erg s^-1 AA^-1
    Returns
    Q: s^-1
    """
    h = 6.626070040E-34 # J s
    h_erg = h * 1e7 # erg s
    c = 2.99E8 # m s-1
    c_AA = c * 1e10 # AA s-1
    f = lambda x: np.interp(x, lam, L_AA * lam) / (h_erg*c_AA)
    return integrate.quad(f, 0, 912, limit=limit)[0]


# def calculate_U(Q, R_inner=0.01 * 3.086e18, n_h=100):
#     """
#     U = Q / (4 * pi * R_inner^2 * n_H * c)
#     
#     Q - units: s^-1
#     R_inner - units: cm
#     n_h - units: cm^-3
# 
#     returns U - units: dimensionless
#     """
#     return Q / (4 * np.pi * R_inner**2 * n_h * 2.99e10)






def write_submission_script_cosma(N, input_prefix, params):#cosma_project, cosma_account, input_file_dir='grids/cloudy_output/'):

    output = []
    output.append(f'#!/bin/bash -l\n')
    output.append(f'#SBATCH --ntasks 1\n')
    output.append(f'#SBATCH -J job_name\n')
    output.append(f'#SBATCH --array=0-{N}\n')
    # output.append(f'#SBATCH -o standard_output_file.%A.%a.out
    # output.append(f'#SBATCH -e standard_error_file.%A.%a.err
    output.append(f'#SBATCH -p {params.cosma_project}\n')
    output.append(f'#SBATCH -A {params.cosma_account}\n')
    output.append(f'#SBATCH --exclusive\n')
    output.append(f'#SBATCH -t 00:15:00\n\n')
    # output.append(f'#SBATCH --mail-type=END # notifications for job done &
    # output.append(f'#SBATCH --mail-user=<email address>
    output.append(f'{params.cloudy_dir}/source/cloudy.exe -r {input_prefix}_$SLURM_ARRAY_TASK_ID\n')

    # --- write input file
    open(f'{params.cloudy_output_dir}/{input_prefix}_run.job','w').writelines(output)

    return






def determine_log10Q(log10U_S, log10n_H):

    alpha_B = 2.59E-13 # cm3 s-1
    c = 3.0E8 # m s-1
    c_cm = c * 100. # cm s-1

    n_H = 10**log10n_H
    U_S = 10**log10U_S

    epsilon = 1.

    Q = ((U_S*3.*c_cm)**3/alpha_B**2)*((4.*np.pi)/(3*epsilon**2*n_H))

    return np.log10(Q)



# def create_CLOUDY_SED(lam, Lnu):
# 
# 
#     nu = 3E8/(lam*1E-10)  # frequency in Hz
#     nu_log10 = np.log10(nu)
# 
#     Lnu_log10 = np.log10(Lnu+1E-99)
#     Lnu_log10 -= np.max(Lnu_log10)
# 
#     # --- reformat for CLOUDY
# 
#     CLOUDY_SED = ['{'+'{0:.5f} {1:.5f}'.format(x,y)+'}' for x,y in zip(nu_log10[::2], Lnu_log10[::2])]
#     CLOUDY_SED = CLOUDY_SED[:19000]
#     CLOUDY_SED = CLOUDY_SED[::-1]
# 
#     return CLOUDY_SED



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




def read_continuum(output_dir, output_file):

    # ----- Open SED

    # 1 = incident, 2 = transmitted, 3 = nebular, 4 = total, 8 = contribution of lines to total
    lam, incident, transmitted, nebular, total, linecont  = np.loadtxt(f'{output_dir}/{output_file}.cont', delimiter='\t', usecols = (0,1,2,3,4,8)).T 

    # --- frequency
    nu = 3E8/(lam*1E-10)

    # --- nebular continuum is the total nebular emission (nebular) - the line continuum (linecont)
    nebular_continuum = nebular - linecont


    for SED_type in ['incident', 'transmitted', 'nebular_continuum', 'total', 'linecont']:

        locals()[SED_type] /= 10**7  #
        locals()[SED_type] /= nu



    return lam, nu, incident, transmitted, nebular_continuum, total, linecont
