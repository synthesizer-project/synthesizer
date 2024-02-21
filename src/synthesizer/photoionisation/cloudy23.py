"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import numpy as np
import shutil
from unyt import c, h, angstrom, unyt_array
from synthesizer.photoionisation import calculate_Q_from_U


class ShapeCommands:

    """
    A class for holding different cloudy shape commands
    """

    def table_sed(model_name, lam, lnu, output_dir="./"):
        """
        A function for creating a cloudy input file using a tabulated SED.

        TODO: allow the user to instead specify nu and to automatically convert
        units if provided

        Args:
            model_name (str)
                User defined name of the model used for cloudy inputs
                and outputs.
            lam (array or unyt_array)
                Wavelength grid with or without units (via unyt)
            lnu (array)
                Spectral luminosity density
            output_dir (str)
                Output directory path

        Returns:
            list
                a list of strings with the cloudy input commands
        """

        # if lam is not a unyt_array assume it has units of angstrom and
        # convert to a unyt_array
        if not isinstance(lam, unyt_array):
            lam *= angstrom

        # define frequency
        nu = c / lam

        # define energy
        E = h * nu

        # define energy in units of Rydbergs
        E_Ryd = E.to("Ry").value

        # get rid of negative/zero luminosities, which are unphysical and seem
        # to make cloudy break
        lnu[lnu <= 0.0] = 1e-100

        # save tabulated spectrum
        np.savetxt(
            f"{output_dir}/{model_name}.sed",
            np.array([E_Ryd[::-1], lnu[::-1]]).T,
        )

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(f'table SED "{model_name}.sed" \n')

        return shape_commands

    def cloudy_agn(TBB, aox=-1.4, auv=-0.5, ax=-1.35):
        """
        A function for specifying the cloudy AGN model. See 6.2 Hazy1.pdf.

        Args:
            model_name (str)
                User defined name of the model used for cloudy inputs and
                outputs
            TBB (float)
                The Big Bump temperature
            aox (float)
                The x-ray slope (default value from Calabro CEERS AGN model)
            auv (float)
                The uv-slope (default value from Calabro CEERS AGN model)
            ax (float)
                Slope normalisation

        Returns:
            list
                a list of strings with the cloudy input commands
        """

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(
            f"AGN T = {TBB} k, a(ox) = {aox}, a(uv)= {auv} \
                              a(x)={ax} \n"
        )

        return shape_commands


def create_cloudy_input(
    model_name, shape_commands, abundances, output_dir="./", **kwargs
):
    """
    A generic function for creating a cloudy input file

    Args:

        model_name (str)
            The model name. Used in the naming of the outputs
        shape_commands (list)
            List of strings describing the cloudy input commands
        abundances: (Abundances object)
            A synthsizer Abundances object
        output_dir (str)
            The output dir

    Returns:
        list
            A list of cloudy commands

    """

    default_params = {
        # ionisation parameter
        "ionisation_parameter": 0.01,
        # radius in log10 parsecs, only important for spherical geometry
        "radius": 0.01,
        # covering factor. Keep as 1 as it is more efficient to simply combine
        # SEDs to get != 1.0 values
        "covering_factor": False,
        # K, if not provided the command is not used
        "stop_T": False,
        # if not provided the command is not used
        "stop_efrac": False,
        # K, if not provided the command is not used
        "T_floor": False,
        # Hydrogen density
        "hydrogen_density": 10 ** (2.5),
        # redshift, only necessary if CMB heating included
        "z": 0.0,
        # include CMB heating
        "CMB": False,
        # include cosmic rays
        "cosmic_rays": False,
        # include dust grains
        "grains": False,
        # the geometry
        "geometry": "planeparallel",
        # relative resolution the saved continuum spectra
        "resolution": 1.0,
        # output abundances
        "output_abundances": True,
        # output continuum
        "output_cont": True,
        # output full list of all available lines
        "output_lines": False,
        # output linelist
        "output_linelist": "linelist.dat",
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

    # The original method of handling depletion and grain
    if params["depletion_model"] in ["ClassicCloudy", "Gutkin2016"]:
        # Define the chemical composition, here we use the depleted (gas-phase)
        # abundances and set "no grains".
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

        # If grain physics are required, add this self-consistently
        if (abundances.dust_to_metal_ratio > 0) & params["grains"]:
            delta_C = 10**abundances.total["C"] - 10**abundances.gas["C"]
            delta_PAH = 0.01 * (10 ** abundances.total["C"])
            delta_graphite = delta_C - delta_PAH
            delta_Si = 10**abundances.total["Si"] - 10**abundances.gas["Si"]
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

    # Jenkins 2009 depletion model as implemented by cloudy.
    # See 2023 release paper section 5
    elif params["depletion_model"] == 'Jenkins2009':

        # Define the chemical composition, because Jenkins2009 applies
        # depletion as a cloudy command we use total abundances and there
        # is no need to use the "no grains" command.
        for ele in ["He"] + abundances.metals:
            cinput.append(
                (
                    f"element abundance {abundances.name[ele]} "
                    f"{abundances.total[ele]}\n"
                )
            )

        if params["fstar"] == 0.5:
            cinput.append("metals deplete Jenkins 2009")

            # turn on grains
            if params["grains"]:
                cinput.append("grains")

        else:
            value = params["fstar"]
            cinput.append(f"metals depletion jenkins 2009 fstar {value}")

            # turn on grains
            if params["grains"]:

                # for non-default (!= 0.5) fstar values, grains needs to be
                # scaled.

                # calculate the dust mass fraction for the default parameter
                # choice.
                default_dust_mass_fraction = abundances.apply_depletion(
                    depletion_model='Jenkins2009').dust_abundance

                # calculate the dust mass fraction for the provided value of
                # fstar.
                dust_mass_fraction = abundances.apply_depletion(
                    depletion_model='Jenkins2009',
                    depletion_scaling=params["fstar"]).dust_abundance
                ratio = dust_mass_fraction / default_dust_mass_fraction
                cinput.append(f"grains {ratio}")

    ionisation_parameter = params["ionisation_parameter"]
    log10ionisation_parameter = np.log10(ionisation_parameter)

    # plane parallel geometry
    if params["geometry"] == "planeparallel":
        cinput.append(
            f"ionization parameter = {log10ionisation_parameter:.3f}\n")

    # spherical geometry
    if params["geometry"] == "spherical":
        # in the spherical geometry case I think U is some average U, not U at
        # the inner face of the cloud.
        log10ionising_luminosity = np.log10(calculate_Q_from_U(
            log10ionisation_parameter,
            params["hydrogen_density"]))
        cinput.append(f"Q(H) = {log10ionising_luminosity}\n")
        cinput.append(f'radius {np.log10(params["radius"])} log parsecs\n')
        cinput.append("sphere\n")

    # add background continuum
    if params["cosmic_rays"]:
        cinput.append("cosmic rays, background\n")

    if params["CMB"]:
        cinput.append(f'CMB {params["z"]}\n')

    # define hydrogend density
    if params["hydrogen_density"]:
        cinput.append(
            (
                f"hden {np.log10(params['hydrogen_density'])}"
                " log constant "
                "density\n"
            )
        )

    # covering factor
    if params["covering_factor"]:
        cinput.append(f'covering factor {params["covering_factor"]} linear\n')

    # Processing commands
    cinput.append("iterate to convergence\n")
    if params["T_floor"]:
        cinput.append(f'set temperature floor {params["T_floor"]} linear\n')

    if params["stop_T"]:
        cinput.append(f'stop temperature {params["stop_T"]}K\n')

    if params["stop_efrac"]:
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
            f'save linelist column emergent absolute last units angstroms \
                  "{model_name}.elin" "linelist.dat"\n'
        )

        # make copy of linelist
        shutil.copyfile(
            params["output_linelist"], f"{output_dir}/linelist.dat"
        )

    # save input file
    if output_dir is not None:
        open(f"{output_dir}/{model_name}.in", "w").writelines(cinput)

    return cinput


def read_lines(filename, extension="lines"):
    """
    Read a full line list from a cloudy output file.

    Args:
        filename (str)
            The cloudy filename
        extension (str)
            The extension of the file

    Returns:
        line_ids (list)
            A list of line identificaitons
        blends (list)
            A list containing flags whether the line is a blend
        wavelengths (list)
            A list of the line wavelengths
        intrinsic (list)
            A list of the intrinsic luminosities
        emergent (list)
            A list of emergent luminosities

    """

    # open file and read columns
    wavelengths, cloudy_ids, intrinsic, emergent = np.loadtxt(
        f"{filename}.{extension}",
        dtype=str,
        delimiter="\t",
        usecols=(0, 1, 2, 3),
    ).T

    wavelengths = wavelengths.astype(float)
    intrinsic = intrinsic.astype(float) - 7.0  # erg s^{-1}
    emergent = emergent.astype(float) - 7.0  # erg s^{-1}

    # make a new cloudy ID e.g. "H I 4861.33A"
    line_ids = np.array(
        [" ".join(list(filter(None, id.split(" ")))) for id in cloudy_ids]
    )

    # find out the length of the line id when split
    lenid = np.array(
        [len(list(filter(None, id.split(" ")))) for id in cloudy_ids]
    )

    # define a blend as a line with only two entries
    blends = np.ones(len(wavelengths), dtype=bool)
    blends[lenid == 3] = False

    return line_ids, blends, wavelengths, intrinsic, emergent


def convert_cloudy_wavelength(x):
    """
    Convert a cloudy wavelength string (e.g. 6562.81A, 1.008m) to a wavelength
    float in units of angstroms.

    Args:
        x (str)
            A cloudy wavelength string

    Returns:
        float
            The wavelength in Angstroms
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
    Args:
        filename (str)
            The cloudy filename
        extension (str)
            The extension of the file

    Returns:
        line_ids (list)
            A list of line identificaitons
        wavelengths (list)
            A list of the line wavelengths
        luminosities (list)
            A list of the luminosities
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

            line_ids.append(id)
            wavelengths.append(wavelength)
            luminosities.append(lum)

    return np.array(line_ids), np.array(wavelengths), np.array(luminosities)


def read_wavelength(filename):
    """
    Extract just the wavelength grid from cloudy .cont file and reverse the
    order

    Args:
        filename (str)
            The cloudy filename

    Returns:
        ndnarray
            The wavelength grid

    """

    lam = np.loadtxt(f"{filename}.cont", delimiter="\t", usecols=(0)).T
    lam = lam[::-1]  # reverse order

    return lam


def read_continuum(filename, return_dict=False):
    """
    Extract just the spectra from a cloudy.cont file

    Args:
        filename (str)
            The cloudy filename

    Returns:
        lam (array-like, float)
            The wavelength grid
        nu (array-like, float)
            The frequency grid
        incident (array-like, float)
            The incident spectrum
        transmitted (array-like, float)
            The transmitted spectrum
        nebular (array-like, float)
            The nebular spectrum
        nebular_continuum (array-like, float)
            The nebular continuum spectrum
        total (array-like, float)
            The total spectrum
        linecont (array-like, float)
            The line contribution spectrum

        alternatively returns a dictionary

    """

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
