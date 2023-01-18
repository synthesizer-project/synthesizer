"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import numpy as np
from scipy import integrate

from unyt import c, h, angstrom


def create_cloudy_input(model_name, lam, lnu, abundances, output_dir='./', **kwargs):

    params = {
        'log10U': -2,
        'log10radius': -2,  # radius in log10 parsecs
        # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
        'covering_factor': 1.0,
        'stop_T': 4000,  # K
        'stop_efrac': -2,
        'T_floor': 100,  # K
        'log10n_H': 2.5,  # Hydrogen density
        'z': 0.,
        'CMB': False,
        'cosmic_rays': False
    }

    for key, value in list(kwargs.items()):
        params[key] = value

    log10U = params['log10U']

    nu = c/(lam * angstrom)
    E = h*nu
    E_Ryd = E.to('Ry').value

    lnu[lnu <= 0.0] = 1E-10  #  get rid of negative models

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

    if abundances.params['d2m'] > 0:
        delta_C = 10**abundances.a_nodep['C'] - 10**abundances.a['C']
        delta_PAH = 0.01 * (10**abundances.a_nodep['C'])
        delta_graphite = delta_C - delta_PAH
        delta_Si = 10**abundances.a_nodep['Si'] - 10**abundances.a['Si']
        orion_C_abund = -3.6259
        orion_Si_abund = -4.5547
        PAH_abund = -4.446
        f_graphite = delta_graphite/(10**(orion_C_abund))
        f_Si = delta_Si/(10**(orion_Si_abund))
        f_pah = delta_PAH/(10**(PAH_abund))
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
    open(f'{output_dir}/{model_name}.in', 'w').writelines(cinput)

    return cinput


def calculate_Q_from_U(U_avg, n_h):
    """
    Args
    U - units: dimensionless
    n_h - units: cm^-3

    Returns
    Q - units: s^-1
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
    epsilon = 1.

    return ((U_avg * c_cm)**3 / alpha_B**2) *\
        ((4 * np.pi) / (3 * epsilon**2 * n_h))

def calculate_U_from_Q(Q_avg, n_h=100):
    """
    Args
    Q - units: s^-1
    n_h - units: cm^-3

    Returns
    U - units: dimensionless
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
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
    h = 6.626070040E-34  # J s
    h_erg = h * 1e7  # erg s
    c = 2.99E8  # m s-1
    c_AA = c * 1e10  # AA s-1
    def f(x): return np.interp(x, lam, L_AA * lam) / (h_erg*c_AA)
    return integrate.quad(f, 0, 912, limit=limit)[0]


def default_lines():
    return [
        'MgII2803', 'MgII2796',
        'ArIII7136', 'ArIII7751', 'ArIV4711', 'ArIV4740',
        'NeIII3869', 'NeIII3967',
        'FeIV3095', 'FeIV2836', 'FeIV2829',
        'CII2325', 'CII2327', 'CIII1909', 'CIII1907', 'CIV1551', 'CIV1548', 'C-1335',
        'OII2470', 'OII3729', 'OII3726', 'OIII4959', 'OIII5007', 'OIII2321', 'OIII4363',
        'OIII1661', 'OIII1666',
        'SiIII1892', 'SiIII1883', 'SiIII1206', 'SiIV1394', 'SiIV1403',
        'NII6583',
        'SII6731', 'SII6716', 'SIII9069', 'SIII9531', 'SIII3722', 'SIII6312',
        'HeII1640', 'HeI10830', 'HeI3889', 'HeI3188', 'HeI2945', 'HeI2829', 'HeI20581',
        'HeI5016', 'HeI3965', 'HeI7065', 'HeI5876', 'HeI4471', 'HeI4026', 'HeI3820',
        'HeI3705', 'HeI6678', 'HeI4922', 'HeI18685',
        'HI1216', 'HI1026', 'HI6563', 'HI4861', 'HI4340', 'HI4102', 'HI3970', 'HI3889',
        'HI3835', 'HI3798', 'HI3771', 'HI3750', 'HI3734', 'HI3722', 'HI3712', 'HI3704',
        'HI3697', 'HI3692', 'HI3687', 'HI3683', 'HI3679', 'HI3671', 'HI3669', 'HI18751',
        'HI12818', 'HI10938', 'HI10049', 'HI9546', 'HI9229', 'HI9015', 'HI8863', 'HI8750',
        'HI8665', 'HI8323', 'HI26251', 'HI21655', 'HI19445', 'HI18174',
    ]


class Line:
    wv = None  # wavelength
    intrinsic = None  #  intrinsic luminosity
    emergent = None  # emergent luminosity


def get_new_id(wv, cloudy_id):
    """ convert the cloudy ID into a new form ID """

    wv = int(np.round(wv, 0))

    li = list(filter(None, cloudy_id.split(' ')))

    e = li[0]

    i = li[1]
    j = '-'
    if i == '1':
        j = 'I'
    if i == '2':
        j = 'II'
    if i == 'II':
        j = 'II'
    if i == '3':
        j = 'III'
    if i == '4':
        j = 'IV'
    if i == '5':
        j = 'V'
    if i == '6':
        j = 'VI'
    if i == '7':
        j = 'VII'
    if i == '8':
        j = 'VIII'
    if i == '9':
        j = 'IX'
    if i == '10':
        j = 'X'
    if i == '11':
        j = 'XI'
    if i == '12':
        j = 'XII'
    if i == '13':
        j = 'XIII'
    if i == '14':
        j = 'XIV'

    return e+j+str(wv)


def read_lines(filename, line_ids=None):

    if not line_ids:
        line_ids = default_lines()

    wavelengths, cloudy_line_ids, intrinsic, emergent = np.loadtxt(
        f'{filename}.lines', dtype=str, delimiter='\t', usecols=(0, 1, 2, 3)).T

    wavelengths = wavelengths.astype(float)
    intrinsic = intrinsic.astype(float) - 7.  # erg s^{-1} magic number
    emergent = emergent.astype(float) - 7.  # erg s^{-1} magic number

    new_line_ids = np.array([get_new_id(wv, cloudy_line_id)
                            for wv, cloudy_line_id in zip(wavelengths, cloudy_line_ids)])

    lines = {}  # dictionary holding the output of the lines

    # for line_id in line_ids:
    #
    #     line = Line()
    #
    #     if line_id in new_line_ids:
    #
    #         s = new_line_ids == line_id
    #
    #         line.wv = wavelengths[s]
    #         line.intrinsic = intrinsic[s]
    #         line.emergent = emergent[s]
    #
    #     lines[line_id] = line

    wavelenths_ = []
    intrinsic_ = []
    emergent_ = []

    for line_id in line_ids:

        if line_id in new_line_ids:

            s = new_line_ids == line_id

            wavelenths_.append(wavelengths[s][0])
            intrinsic_.append(intrinsic[s][0])
            emergent_.append(emergent[s][0])

        else:
            wavelenths_.append(-99)
            intrinsic_.append(-99)
            emergent_.append(-99)

    inds = np.array(wavelenths_).argsort()

    return np.array(line_ids)[inds], np.array(wavelenths_)[inds], np.array(intrinsic_)[inds], np.array(emergent_)[inds]


def make_linecont(filename, wavelength_grid, line_ids=None):
    """ make linecont from lines (hopefully the same as that from the continuum)"""

    if not line_ids:
        line_ids = default_lines()

    line_wavelengths, cloudy_line_ids, intrinsic, emergent = np.loadtxt(
        f'{filename}.lines', dtype=str, delimiter='\t', usecols=(0, 1, 2, 3)).T

    line_wavelengths = line_wavelengths.astype(float)
    intrinsic = intrinsic.astype(float)  # correct for size of cluster # erg s^-1
    emergent = emergent.astype(float)  # correct for size of cluster # erg s^-1

    new_line_ids = np.array([get_new_id(wv, cloudy_line_id)
                            for wv, cloudy_line_id in zip(line_wavelengths, cloudy_line_ids)])

    line_spectra = np.zeros(len(wavelength_grid)) + 1E-100

    for new_line_id, line_wv, line_luminosity in zip(new_line_ids, line_wavelengths, emergent):

        if new_line_id in line_ids:

            line_luminosity += -7.  # erg -> W ??????

            idx = (np.abs(wavelength_grid-line_wv)).argmin()
            dl = 0.5*(wavelength_grid[idx+1] - wavelength_grid[idx-1])
            n = c.value/(line_wv*1E-10)
            line_spectra[idx] += line_wv*((10**line_luminosity)/n)/dl

    return line_spectra


def read_wavelength(filename):
    """ return just wavelength grid from cloudy file and reverse the order """

    lam = np.loadtxt(f'{filename}.cont', delimiter='\t', usecols=(0)).T
    lam = lam[::-1]  # reverse order

    return lam


def read_continuum(filename, return_dict=False):
    """ read a cloudy continuum file and convert spectra to erg/s/Hz """

    # ----- Open SED

    # 1 = incident, 2 = transmitted, 3 = nebular, 4 = total, 8 = contribution of lines to total
    lam, incident, transmitted, nebular, total, linecont = np.loadtxt(
        f'{filename}.cont', delimiter='\t', usecols=(0, 1, 2, 3, 4, 8)).T

    # --- frequency
    lam = lam[::-1]  # reverse array
    lam_m = lam * 1E-10  # m
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



def _create_cloudy_binary(grid, params, verbose=False):
    """
    DEPRECATED create a cloudy binary file

    Args:

    grid: synthesizer _grid_ object
    """

    # # ---- TEMP check for negative values and amend
    # # The BPASS binary sed has a couple of erroneous negative values,
    # # possibly due to interpolation errors
    # # Here we set the Flux to the average of each
    # # neighbouring wavelength value
    #
    # mask = np.asarray(np.where(sed < 0.))
    # for i in range(mask.shape[1]):
    #     sed[mask[0,i],mask[1,i],mask[2,i]] = \
    #           sed[mask[0,i],mask[1,i],mask[2,i]-1]+sed[mask[0,i],mask[1,i],mask[2,i]+1]/2

    if verbose:
        print('Writing .ascii')

    output = []
    output.append("20060612\n")  # magic number
    output.append("2\n")  # ndim
    output.append("2\n")  # npar

    # First parameter MUST be log otherwise Cloudy throws a tantrum
    output.append("age\n")  # label par 1
    output.append("logz\n")  # label par 2

    output.append(str(grid.spectra['stellar'].shape[0] *
                      grid.spectra['stellar'].shape[1])+"\n")  # nmod
    output.append(str(len(grid.lam))+"\n")  # nfreq (nwavelength)
    # output.append(str(len(frequency))+"\n")  # nfreq (nwavelength)

    output.append("lambda\n")  # type of independent variable (nu or lambda)
    output.append("1.0\n")  # conversion factor for independent variable

    # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
    # output.append("F_nu\n")

    # output.append("3.839e33\n")  # conversion factor for dependent variable

    # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
    output.append("F_lambda\n")
    output.append("1.0\n")  # conversion factor for dependent variable

    for a in grid.ages:  # available SED ages
        for z in grid.metallicities:
            output.append(f'{np.log10(a)} {z}\n')  # (npar x nmod) parameters

    # the frequency(wavelength) grid, nfreq points
    output.append(' '.join(map(str, grid.lam))+"\n")

    for i, a in enumerate(grid.ages):
        for j, z in enumerate(grid.metallicities):
            output.append(' '.join(map(str,
                                       grid.spectra['stellar'][i, j]))+"\n")

    with open('model.ascii', 'w') as target:
        target.writelines(output)

    # ---- compile ascii file
    print('Compiling Cloudy atmosphere file (.ascii)')
    subprocess.call(('echo -e \'compile stars \"model.ascii\"\''
                    f'| {params.cloudy_dir}/source/cloudy.exe'), shell=True)

    # ---- copy .mod file to cloudy data directory
    print(('Copying compiled atmosphere to Cloudy directory, '
           f'{params.cloudy_dir}'))
    subprocess.call(f'cp model.mod {params.cloudy_dir}/data/.', shell=True)

    # ---- remove .ascii file
    # os.remove(out_dir+model+'.ascii')

