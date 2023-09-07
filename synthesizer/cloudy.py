"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import numpy as np
from scipy import integrate
import shutil
from unyt import c, h, angstrom, eV, erg, s, Hz, unyt_array


from dataclasses import dataclass


@dataclass
class Ions:

    """
    A dataclass holding the ionisation energy of various ions amongst other
    properties and methods

    Used for calculating ionising photon luminosities (Q).

    Values taken from: https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page)
    """

    energy = {
        "HI": 13.6 * eV,
        "HeI": 24.6 * eV,
        "HeII": 54.4 * eV,
        "CII": 24.4 * eV,
        "CIII": 47.9 * eV,
        "CIV": 64.5 * eV,
        "NI": 14.5 * eV,
        "NII": 29.6 * eV,
        "NIII": 47.4 * eV,
        "OI": 13.6 * eV,
        "OII": 35.1 * eV,
        "OIII": 54.9 * eV,
        "NeI": 21.6 * eV,
        "NeII": 41.0 * eV,
        "NeIII": 63.45 * eV,
    }


class ShapeCommands:

    """
    A class for holding different cloudy shape commands

    """

    def table_sed(model_name, lam, lnu, output_dir="./"):
        """
        A function for creating a cloudy input file using a tabulated SED.

        Arguments
        ----------
        model_name: str
            User defined name of the model used for cloudy inputs and outputs.
        lam: array or unyt_array
            Wavelength grid with or without units (via unyt)
        lnu: array
            Spectral luminosity density
        output_dir: str
            Output directory path

        Returns
        -------
        list
            a list of strings with the cloudy input commands


        TODO
        -------
        - allow the user to instead specify nu and to automatically convert units if provided

        """

        # if lam is not a unyt_array assume it has units of angstrom and convert to a unyt_array
        if type(lam) != unyt_array:
            lam *= angstrom

        # define frequency
        nu = c / lam

        # define energy
        E = h * nu

        # define energy in units of Rydbergs
        E_Ryd = E.to("Ry").value

        # get rid of negative/zero luminosities, which are unphysical and seem to make cloudy break
        lnu[lnu <= 0.0] = 1e-100

        # save tabulated spectrum
        np.savetxt(
            f"{output_dir}/{model_name}.sed", np.array([E_Ryd[::-1], lnu[::-1]]).T
        )

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(f'table SED "{model_name}.sed" \n')

        return shape_commands


    def cloudy_agn(TBB, aox=-1.4, auv=-0.5, ax=-1.35): # values from Calabro CEERS AGN model
        
        """
        A function for specifying the cloudy AGN model. See 6.2 Hazy1.pdf.

        Arguments
        ----------
        model_name: str
            User defined name of the model used for cloudy inputs and outputs.
        TBB: float
            The Big Bump temperature
        aox: float

        auv: float

        ax: float
            
        Returns
        -------
        list
            a list of strings with the cloudy input commands
        """

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(f'AGN T ={TBB} k, a(ox) = {aox}, a(uv)= {auv} a(x)={ax} \n')

        return shape_commands


def create_cloudy_input(model_name, shape_commands, abundances,
                        output_dir='./', **kwargs):

    """
    A generic function for creating a cloudy input file

    Arguments
    ----------
    model_name: str
        The model name. Used in the naming of the outputs
    shape_commands: list
        List of strings describing the cloudy input commands
    abundances: obj
        A synthsizer Abundances object
    output_dir: str
        The output dir

    Returns
    -------
    """

    default_params = {
        "log10U": -2,  # ionisation parameter
        "log10radius": -2,  # radius in log10 parsecs, only important for spherical geometry
        "covering_factor": 1.0,  # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
        "stop_T": 4000,  # K
        "stop_efrac": -2,
        "T_floor": 100,  # K
        "log10n_H": 2.5,  # Hydrogen density
        "z": 0.0,  # redshift
        "CMB": False,  # include CMB heating
        "cosmic_rays": False,  # include cosmic rays
        "grains": False,  # include dust grains
        "geometry": "planeparallel",  # the geometry
        "resolution": 1.0,  # relative resolution the saved continuum spectra
        "output_abundances": True,  # output abundances
        "output_cont": True,  # output continuum
        "output_lines": False,  # output full list of all available lines
        "output_linelist": "linelist.dat",  # output linelist
    }

    # update default_params with kwargs
    params = default_params | kwargs

    # old approach for updated parameters
    # for key, value in list(kwargs.items()):
    #     params[key] = value

    # begin input list
    cinput = []

    # add spectral shape commands
    cinput += shape_commands

    # --- Define the chemical composition
    for ele in ["He"] + abundances.metals:
        cinput.append(
            (
                f"element abundance {abundances.name[ele]} "
                f"{abundances.gas[ele]} no grains\n"
            )
        )

    """
    add graphite and silicate grains

    This old version does not actually conserve mass
    as the commands `abundances` and `grains` do not
    really talk with each other
    """
    # # graphite, scale by total C abundance relative to ISM
    # scale = 10**a_nodep['C']/2.784000e-04
    # cinput.append(f'grains Orion graphite {scale}'+'\n')
    # # silicate, scale by total Si abundance relative to ISM
    # # NOTE: silicates also rely on other elements.
    # scale = 10**a_nodep['Si']/3.280000e-05
    # cinput.append(f'grains Orion silicate {scale}'+'\n')

    """ specifies graphitic and silicate grains with a size
    distribution and abundance appropriate for those along
    the line of sight to the Trapezium stars in Orion. The
    Orion size distribution is deficient in small particles
    and so produces the relatively grey extinction observed
    in Orion (Baldwin et al., 1991). One problem with the
    grains approach is metals/element abundances do not talk
    to the grains command and hence there is issues with mass
    conservation (see cloudy documentation). To alleviate
    this one needs to make the orion grain abundances
    consistent with the depletion values. Assume 1 per cent of
    C is in PAH's.

    PAHs appear to exist mainly at the interface between the
    H+ region and the molecular clouds. Apparently PAHs are
    destroyed in ionized gas (Sellgren et al., 1990, AGN3
    section 8.5) by ionizing photons and by collisions with
    ions (mainly H+ ) and may be depleted into larger grains
    in molecular regions. Also assume the carbon fraction of
    PAHs from Abel+2008
    (https://iopscience.iop.org/article/10.1086/591505)
    assuming 1 percent of Carbon in PAHs. The factors in the
    denominators are the abundances of the carbon, silicon and
    PAH fractions when setting a value of 1 (fiducial abundance)
    for the orion and PAH grains.

    Another way is to scale the abundance as a function of the
    metallicity using the Z_PAH vs Z_gas relation from
    Galliano+2008
    (https://iopscience.iop.org/article/10.1086/523621,
    y = 4.17*Z_gas_sol - 7.085),
    which will again introduce issues on mass conservation.
    """

    if (abundances.d2m > 0) & params["grains"]:
        delta_C = 10 ** abundances.a_nodep["C"] - 10 ** abundances.a["C"]
        delta_PAH = 0.01 * (10 ** abundances.a_nodep["C"])
        delta_graphite = delta_C - delta_PAH
        delta_Si = 10 ** abundances.a_nodep["Si"] - 10 ** abundances.a["Si"]
        orion_C_abund = -3.6259
        orion_Si_abund = -4.5547
        PAH_abund = -4.446
        f_graphite = delta_graphite / (10 ** (orion_C_abund))
        f_Si = delta_Si / (10 ** (orion_Si_abund))
        f_pah = delta_PAH / (10 ** (PAH_abund))
        command = (
            f"grains Orion graphite {f_graphite} "
            f"\ngrains Orion silicate {f_Si} \ngrains "
            f"PAH {f_pah}"
        )
        cinput.append(command + "\n")
    else:
        f_graphite, f_Si, f_pah = 0, 0, 0

    # cinput.append('element off limit -7') # should speed up the code

    log10U = params["log10U"]

    # plane parallel geometry
    if params["geometry"] == "planeparallel":
        cinput.append(f"ionization parameter = {log10U:.3f}\n")
        # inner radius = 10^30 cm and thickness = 10^21.5 cm (==1 kpc) this is essentially plane parallel geometry
        cinput.append(f"radius 30.0 21.5\n")

    if params["geometry"] == "spherical":
        # in the spherical geometry case I think U is some average U, not U at the inner face of the cloud.
        log10Q = np.log10(calculate_Q_from_U(10**log10U, 10 ** params["log10n_H"]))
        cinput.append(f"Q(H) = {log10Q}\n")
        cinput.append(f'radius {params["log10radius"]} log parsecs\n')
        cinput.append("sphere\n")

    # add background continuum
    if params["cosmic_rays"]:
        cinput.append("cosmic rays, background\n")
    if params["CMB"]:
        cinput.append(f'CMB {params["z"]}\n')

    # define hydrogend density
    cinput.append(f'hden {params["log10n_H"]} log constant density\n')

    # cinput.append(f'covering factor {params["covering_factor"]} linear\n')

    # --- Processing commands
    cinput.append("iterate to convergence\n")
    cinput.append(f'set temperature floor {params["T_floor"]} linear\n')
    cinput.append(f'stop temperature {params["stop_T"]}K\n')
    cinput.append(f'stop efrac {params["stop_efrac"]}\n')

    # --- output commands
    # cinput.append(f'print line vacuum\n')  # output vacuum wavelengths
    cinput.append(
        f'set continuum resolution {params["resolution"]}\n'
    )  # set the continuum resolution
    cinput.append(f'save overview  "{model_name}.ovr" last\n')

    # output abundances
    if params["output_abundances"]:
        cinput.append(f'save last abundances "{model_name}.abundances"\n')

    # output continuum (i.e. spectra)
    if params["output_cont"]:
        cinput.append(
            (
                f'save last continuum "{model_name}.cont" '
                f"units Angstroms no clobber\n"
            )
        )
    # output lines
    if params["output_lines"]:
        cinput.append(
            (
                f'save last lines, array "{model_name}.lines" '
                "units Angstroms no clobber\n"
            )
        )

    # output linelist
    if params["output_linelist"]:
        cinput.append(
            f'save linelist column emergent absolute last units angstroms "{model_name}.elin" "linelist.dat"\n'
        )

        # make copy of linelist
        shutil.copyfile(params["output_linelist"], f"{output_dir}/linelist.dat")

    # --- save input file
    open(f"{output_dir}/{model_name}.in", "w").writelines(cinput)

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
    epsilon = 1.0

    return ((U_avg * c_cm) ** 3 / alpha_B**2) * (
        (4 * np.pi) / (3 * epsilon**2 * n_h)
    )


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
    epsilon = 1.0

    return ((alpha_B ** (2.0 / 3)) / c_cm) * (
        (3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi)
    ) ** (1.0 / 3)


# # deprecate in favour of the function in sed.py
# def measure_Q(lam, L_AA, limit=100):
#     """
#     Args
#     lam: \\AA
#     L_AA: erg s^-1 AA^-1
#     Returns
#     Q: s^-1
#     """
#     h = 6.626070040E-34  # J s
#     h_erg = h * 1e7  # erg s
#     c = 2.99E8  # m s-1
#     c_AA = c * 1e10  # AA s-1
#     def f(x): return np.interp(x, lam, L_AA * lam) / (h_erg*c_AA)
#     return integrate.quad(f, 0, 912, limit=limit)[0]


# def get_synthesizer_id(wavelength, cloudy_id):
#     """ convert the cloudy line ID into a new form ID """
#
#     # round wavelength
#     wv = int(np.round(wavelength, 0))
#
#     # split id into different components
#     li = list(filter(None, cloudy_id.split(' ')))
#
#     if len(li) == 2:
#         return [li[0]+str(wv), True]
#
#     elif len(li) == 3:
#
#         # element
#         e = li[0]
#
#         # convert arabic ionisation level to roman
#
#         if li[1].isnumeric():
#             ion = get_roman_numeral(int(li[1]))
#         else:
#             ion = li[1]
#
#         return [e+ion+str(wv), False]


def read_lines(filename, extension="lines"):
    """
    Read a full line list from cloudy

    """

    # open file and read columns
    wavelengths, cloudy_ids, intrinsic, emergent = np.loadtxt(
        f"{filename}.{extension}", dtype=str, delimiter="\t", usecols=(0, 1, 2, 3)
    ).T

    wavelengths = wavelengths.astype(float)
    intrinsic = intrinsic.astype(float) - 7.0  # erg s^{-1}
    emergent = emergent.astype(float) - 7.0  # erg s^{-1}

    # make a new cloudy ID e.g. "H I 4861.33A"
    line_ids = np.array(
        [" ".join(list(filter(None, id.split(" ")))) for id in cloudy_ids]
    )

    # find out the length of the line id when split
    lenid = np.array([len(list(filter(None, id.split(" ")))) for id in cloudy_ids])

    # define a blend as a line with only two entries
    blends = np.ones(len(wavelengths), dtype=bool)
    blends[lenid == 3] = False

    return line_ids, blends, wavelengths, intrinsic, emergent


def convert_cloudy_wavelength(x):
    """
    Convert a cloudy wavelength string (e.g. 6562.81A, 1.008m) to a wavelength float in units of angstroms.
    """

    value = float(x[:-1])
    unit = x[-1]

    # if Angstroms
    if unit == "A":
        return value

    # if microns
    if unit == "m":
        return 1e4 * value


def read_linelist(filename, extension="elin"):
    """
    Read a cloudy linelist file.
    """

    # read file
    with open(f"{filename}.{extension}", "r") as f:
        d = f.readlines()

    line_ids = []
    luminosities = []
    wavelengths = []

    for row in d:
        # ignore invalid rows
        if len(row.split("\t")) > 1:
            # split each row using tab character
            id, lum = row.split("\t")

            # reformat line id to be ELEMENT ION WAVELENGTH
            id = " ".join(list(filter(None, id.split(" "))))

            wavelength = convert_cloudy_wavelength(id.split(" ")[-1])

            # convert luminosity to float
            lum = float(lum)

            print(id, wavelength, lum)

            line_ids.append(id)
            wavelengths.append(wavelength)
            luminosities.append(lum)

    return np.array(line_ids), np.array(wavelengths), np.array(luminosities)


# I DON'T BELIEVE THIS IS USED ANYMORE. I BELIEVE THIS WAS TO GENERATE SPECTRA FROM JUST LINES.

# def make_linecont(filename, wavelength_grid, line_ids=None):
#     """
#     make linecont from lines
#     (hopefully the same as that from the continuum)
#     """
#
#     line_wavelengths, cloudy_line_ids, intrinsic, emergent = np.loadtxt(
#         f'{filename}.lines', dtype=str, delimiter='\t', usecols=(0, 1, 2, 3)).T
#
#     line_wavelengths = line_wavelengths.astype(float)
#
#     # correct for size of cluster # erg s^-1
#     intrinsic = intrinsic.astype(float)
#     emergent = emergent.astype(float)  # correct for size of cluster # erg s^-1
#
#     new_line_ids = np.array([get_new_id(wv, cloudy_line_id)
#                              for wv, cloudy_line_id in zip(line_wavelengths,
#                                                            cloudy_line_ids)])
#
#     line_spectra = np.zeros(len(wavelength_grid)) + 1E-100
#
#     for new_line_id, line_wv, line_luminosity in zip(new_line_ids,
#                                                      line_wavelengths,
#                                                      emergent):
#
#         if new_line_id in line_ids:
#
#             line_luminosity += -7.  # erg -> W ??????
#
#             idx = (np.abs(wavelength_grid-line_wv)).argmin()
#             dl = 0.5*(wavelength_grid[idx+1] - wavelength_grid[idx-1])
#             n = c.value/(line_wv*1E-10)
#             line_spectra[idx] += line_wv*((10**line_luminosity)/n)/dl
#
#     return line_spectra


def read_wavelength(filename):
    """return just wavelength grid from cloudy file and reverse the order"""

    lam = np.loadtxt(f"{filename}.cont", delimiter="\t", usecols=(0)).T
    lam = lam[::-1]  # reverse order

    return lam


def read_continuum(filename, return_dict=False):
    """read a cloudy continuum file and convert spectra to erg/s/Hz"""

    # ----- Open SED

    """
    1 = incident, 2 = transmitted, 3 = nebular,
    4 = total, 8 = contribution of lines to total
    """
    lam, incident, transmitted, nebular, total, linecont = np.loadtxt(
        f"{filename}.cont", delimiter="\t", usecols=(0, 1, 2, 3, 4, 8)
    ).T

    # --- frequency
    lam = lam[::-1]  # reverse array
    lam_m = lam * 1e-10  # m
    nu = c.value / (lam_m)

    """
    nebular continuum is the total nebular emission (nebular)
    minus the line continuum (linecont)
    """
    nebular_continuum = nebular - linecont

    spec_dict = {"lam": lam, "nu": nu}

    for spec_type in [
        "incident",
        "transmitted",
        "nebular",
        "nebular_continuum",
        "total",
        "linecont",
    ]:
        sed = locals()[spec_type]
        sed = sed[::-1]  # reverse array
        sed /= 10**7  # convert from W to erg
        sed /= nu  # convert from nu l_nu to l_nu
        spec_dict[spec_type] = sed

    if return_dict:
        return spec_dict
    else:
        return (
            lam,
            nu,
            incident,
            transmitted,
            nebular,
            nebular_continuum,
            total,
            linecont,
        )


# def _create_cloudy_binary(grid, params, verbose=False):
#     """
#     DEPRECATED create a cloudy binary file
#
#     Args:
#
#     grid: synthesizer _grid_ object
#     """
#
#     # # ---- TEMP check for negative values and amend
#     # # The BPASS binary sed has a couple of erroneous negative values,
#     # # possibly due to interpolation errors
#     # # Here we set the Flux to the average of each
#     # # neighbouring wavelength value
#     #
#     # mask = np.asarray(np.where(sed < 0.))
#     # for i in range(mask.shape[1]):
#     #     sed[mask[0,i],mask[1,i],mask[2,i]] = \
#     #           sed[mask[0,i],mask[1,i],mask[2,i]-1]+sed[mask[0,i],mask[1,i],mask[2,i]+1]/2
#
#     if verbose:
#         print('Writing .ascii')
#
#     output = []
#     output.append("20060612\n")  # magic number
#     output.append("2\n")  # ndim
#     output.append("2\n")  # npar
#
#     # First parameter MUST be log otherwise Cloudy throws a tantrum
#     output.append("age\n")  # label par 1
#     output.append("logz\n")  # label par 2
#
#     output.append(str(grid.spectra["incident"].shape[0] *
#                       grid.spectra["incident"].shape[1])+"\n")  # nmod
#     output.append(str(len(grid.lam))+"\n")  # nfreq (nwavelength)
#     # output.append(str(len(frequency))+"\n")  # nfreq (nwavelength)
#
#     output.append("lambda\n")  # type of independent variable (nu or lambda)
#     output.append("1.0\n")  # conversion factor for independent variable
#
#     # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
#     # output.append("F_nu\n")
#
#     # output.append("3.839e33\n")  # conversion factor for dependent variable
#
#     # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
#     output.append("F_lambda\n")
#     output.append("1.0\n")  # conversion factor for dependent variable
#
#     for a in grid.ages:  # available SED ages
#         for z in grid.metallicities:
#             output.append(f'{np.log10(a)} {z}\n')  # (npar x nmod) parameters
#
#     # the frequency(wavelength) grid, nfreq points
#     output.append(' '.join(map(str, grid.lam))+"\n")
#
#     for i, a in enumerate(grid.ages):
#         for j, z in enumerate(grid.metallicities):
#             output.append(' '.join(map(str,
#                                        grid.spectra["incident"][i, j]))+"\n")
#
#     with open('model.ascii', 'w') as target:
#         target.writelines(output)
#
#     # ---- compile ascii file
#     print('Compiling Cloudy atmosphere file (.ascii)')
#     subprocess.call(('echo -e \'compile stars \"model.ascii\"\''
#                     f'| {params.cloudy_dir}/source/cloudy.exe'), shell=True)
#
#     # ---- copy .mod file to cloudy data directory
#     print(('Copying compiled atmosphere to Cloudy directory, '
#            f'{params.cloudy_dir}'))
#     subprocess.call(f'cp model.mod {params.cloudy_dir}/data/.', shell=True)
#
#     # ---- remove .ascii file
#     # os.remove(out_dir+model+'.ascii')
