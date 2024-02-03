import numpy as np

from ..particle.galaxy import Galaxy

from bigfile import BigFile
import scipy.interpolate as interpolate
from scipy import integrate


class BlueTidesDataHolder:
    """
    A holder for BlueTides data that makes it easier to work with.

    The class can be pickled as follows:

        file_name = f'bluetides_{z}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(
                BlueTidesDataHolder_instance,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )
            print(f'Object successfully saved to "{file_name}"')

    *** everything is in proper distance
    """

    def __init__(
        self,
        z,
        bluetides_data_folder="/fred/oz183/sberger/bluetides/BlueTides/",
        end_of_arr=108001,
    ):
        # Units
        self.m_solar = 1.99 * 10**30  # unit: kg
        self.l_solar = 3.826 * 10**33  # unit: erg/s
        self.g_constant = 6.67408 * 10 ** (-11)  # unit: m^3*kg^(-1)*s^(-2)
        self.mp = 1.673e-27  # kg
        self.kpc = 3.086 * 10**19  # unit: m
        self.k_constant = 1.38 * 10 ** (-23)  # unit: m^2*kg*s^(-2)*K^(-1)

        self.factor = (
            121.14740013761634  # GADGET unit Protonmass / Bolztman const
        )
        self.GAMMA = 5 / 3.0
        self.Xh = 0.76

        self.hh = 0.697  # hubble constant / 100 (dimensionless)
        self.c = 3e8  # m/s
        self.eta = 0.1
        self.yr = 365 * 24 * 3600

        self.mpc_to_km = 3.086e19
        self.sec_to_yr = 3.17098e-8  # convert to years
        self.omk = 0
        self.oml = 0.7186
        self.omm = 0.2814
        self.h = 0.697

        # Helper functions
        def get_placed(number, offarray):
            dummy = np.where(offarray <= number)[0]
            return dummy[-1]

        def get_normalized_hubble_parameter(a):  # normalized hubble parameter
            return np.sqrt(self.omm * a ** (-3) + self.oml)

        def calculate_integrand(a):
            return 1.0 / (a * get_normalized_hubble_parameter(a))

        def get_age(a0, a00):  # get age or time between two scale factors
            return (
                1.0
                / (self.h * 100.0)
                * integrate.quad(calculate_integrand, a0, a00)[0]
            )

        self.accretion_unit_conversion = 1e10 / (
            self.kpc / 1e3 / self.yr
        )  # to solar masses per year
        self.acrtolum = (
            self.eta * self.c * self.c * (self.m_solar / self.yr) * 10**7
        )  # Accounts for unit conversion to physical units already happening
        self.z = z

        # Load all BlueTides data
        if self.z == 7:
            sunset = BigFile(bluetides_data_folder + "sunset_208")
            pig_zoo = [bluetides_data_folder + "PIG_208"]
            length_of_bhar = end_of_arr
        elif self.z == 6.5:
            sunset = BigFile(bluetides_data_folder + "sunset_271")
            pig_zoo = [bluetides_data_folder + "PIG_271"]
            length_of_bhar = end_of_arr
        elif self.z == 7.5:
            pig_zoo = [bluetides_data_folder + "PIG_129"]
            sunset = BigFile(bluetides_data_folder + "sunset_129")
            length_of_bhar = end_of_arr
        elif self.z == 8:
            pig_zoo = [bluetides_data_folder + "PIG_086"]
            sunset = BigFile(bluetides_data_folder + "sunset_086")
            length_of_bhar = end_of_arr
        else:
            print("z value not supported")
            exit()

        # Determine & store galaxy properties
        file = pig_zoo[0]  # filename
        pig = BigFile(file)  # load data

        print(f"Chosen redshift = {self.z}")
        self.z = self.redshift = float(
            1.0 / pig.open("Header").attrs["Time"] - 1
        )
        print(f"Actual bluetides redshift = {self.z}")

        # If there's no OffsetByType field, use below to get the same output:
        # get FOF info to know which stars correspond to which galaxies
        length_by_type = pig.open("FOFGroups/LengthByType")[0:length_of_bhar]
        offset_by_type = np.cumsum(length_by_type, axis=0)
        a1 = np.zeros((1, 6)).astype(int)
        offset_by_type = np.append(a1, offset_by_type, axis=0).astype(int)
        offset_by_type = np.transpose(offset_by_type)

        self.bh_offset = np.transpose(offset_by_type[5])
        self.StarOffset = StarOffset = np.transpose(offset_by_type[4])

        a_cur = 1.0 / (1.0 + self.redshift)
        sft_space = np.linspace(0, a_cur, 100)
        mpc_to_km = 3.086e19
        sec_to_yr = 3.17098e-8
        age_space = [
            get_age(SFT, 1.0 / (1.0 + self.redshift)) * mpc_to_km * sec_to_yr
            for SFT in sft_space
        ]  # yr
        self.gen_SFT_to_age = interpolate.interp1d(
            sft_space, age_space, fill_value="extrapolate"
        )  # get an interpolator to translate

        self.bhar = (
            pig.open("5/BlackholeAccretionRate")[0:length_of_bhar]
            * self.accretion_unit_conversion
        )  # solar masses per year

        # Indexes of BH with largest accretion rate,
        # everything is sorted by this array
        self.luminous_BH_indices = np.argsort(-self.bhar)

        # load galaxy black hole data
        self.quasarLuminosity = (
            self.bhar[self.luminous_BH_indices] * self.acrtolum
        )  # transfer from accretion rate to AGN luminosity
        self.bh_mass = (
            pig.open("5/BlackholeMass")[0:length_of_bhar][
                self.luminous_BH_indices
            ]
            * 1e10
            / self.hh
        )  # masses of BHs with largest accretion rate

        self.bhid = pig.open("5/ID")[0:length_of_bhar][
            self.luminous_BH_indices
        ]

        self.bh_position = (
            np.transpose(
                pig.open("5/Position")[0:length_of_bhar][
                    self.luminous_BH_indices
                ]
            )
            / self.hh
            / (1 + self.z)
        )

        self.bhar = self.bhar[self.luminous_BH_indices]

        # Set particle location indices for the BHs in order in pig
        offset_index = []  # index in offset
        for i in self.luminous_BH_indices:
            offset_index.append(get_placed(i, self.bh_offset))
        self.offsetIndex = np.array(offset_index)

        # Load stellar particle information
        self.metallicity_ind = pig.open("4/Metallicity")[
            0 : StarOffset[offset_index[np.argmax(StarOffset[offset_index])]]
        ]
        self.star_formation_time_ind = pig.open("4/StarFormationTime")[
            0 : StarOffset[offset_index[np.argmax(StarOffset[offset_index])]]
        ]
        self.position_ind = (
            pig.open("4/Position")[
                0 : StarOffset[
                    offset_index[np.argmax(StarOffset[offset_index])]
                ]
            ]
            / self.hh
            / (1 + self.z)
        )
        self.velocity_ind = (
            pig.open("4/Velocity")[
                0 : StarOffset[
                    offset_index[np.argmax(StarOffset[offset_index])]
                ]
            ]
            / self.hh
        )
        self.met_surface_densities_ind = sunset.open("4/MetSurfaceDensity")[
            0 : StarOffset[offset_index[np.argmax(StarOffset[offset_index])]]
        ]


def load_BlueTides(
    redshift,
    dataholder=None,
    galaxy_bhid=[],
    end_arr=108001,
    bluetides_data_folder="",
):
    """
    Load BlueTides galaxies into a galaxy object

    Args:
        redshift (float):
            desired redshift to extract BlueTides galaxies
        dataholder (BlueTidesDataHolder object):
            contains all data for a particular simulation of BlueTides
            (default is `None` and dataholder class will be generated from
            a BlueTides simulation file for the given redshift)
        galaxy_bhid (array):
            contains all the galaxy BHIDs to include in your galaxy object
            Note: each BlueTides dataholder has arrays sorted in descending
            bhar (default is an empty list)
        end_arr (int):
            last galaxy to include in BlueTides dataholder, selection of this
            value should be done with care to avoid galaxies with BHs too close
            to the seed mass (10^5.8 solar masses; default is 108001)
        bluetides_data_folder (str):
            location of BlueTides pig/sunset files. Only required if
            dataholder is `None` (default is an empty string)

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing specified
            stars and gas components
    """
    if dataholder is None:
        print(f"Loading in BlueTides data from z = {redshift}...")
        dataholder = BlueTidesDataHolder(
            redshift,
            bluetides_data_folder=bluetides_data_folder,
            end_of_arr=end_arr,
        )

    galaxies_length = len(
        dataholder.bh_mass
    )  # holder array for galaxies of the same length
    galaxies = [None] * galaxies_length
    smoothing_length_proper_bluetides = (
        1.5 / dataholder.hh * dataholder.kpc
    ) / (1 + dataholder.z)

    if len(galaxy_bhid) == 0:
        galaxy_cycle = np.arange(galaxies_length)  # take every galaxy
    else:
        # Get galaxies with the BHIDs you desire
        # (since the indices are different for each redshift)
        _, bh_indices, _ = np.intersect1d(
            dataholder.bhid, galaxy_bhid, return_indices=True
        )
        galaxy_cycle = bh_indices

    # Grab stellar particles associated with each galaxy and
    # assign to galaxy in galaxies array
    for ii, bh_index in enumerate(galaxy_cycle):
        # Get the first index of the star particle for the particular galaxy
        idx = int(dataholder.offsetIndex[bh_index])
        star_off = dataholder.StarOffset[idx : idx + 2]
        galaxies[ii] = Galaxy()
        galaxies[ii].redshift = dataholder.redshift

        star_time = dataholder.star_formation_time_ind[
            star_off[0] : star_off[1]
        ]
        ages = (
            dataholder.gen_SFT_to_age(star_time) / 1e6
        )  # translate galaxy star formation time to ages in years

        initial_gas_particle_mass = np.full(ages.shape, 2.36e6) / dataholder.hh
        imasses = initial_gas_particle_mass / 4
        masses = (
            np.ones(ages.shape) * 1e10 * 5.90556119e-05 / 0.697
        )  # mass of each star particle in M_sol
        metallicities = dataholder.metallicity_ind[star_off[0] : star_off[1]]

        star_pos = dataholder.position_ind[star_off[0] : star_off[1]]

        # Centering all stars around the BH such that (0, 0, 0) is
        # the BH position
        star_relpos = (
            star_pos - dataholder.bh_position[:, bh_index]
        )  # same for BH_position
        x = star_relpos[:, 0]
        y = star_relpos[:, 1]
        z = star_relpos[:, 2]

        smoothing_lengths = np.full(
            ages.shape, smoothing_length_proper_bluetides
        )

        coords = np.transpose([x, y, z])
        galaxies[ii].load_stars(
            initial_masses=imasses * dataholder.m_solar,
            ages=ages * dataholder.yr,
            metals=metallicities,
            coordinates=coords * dataholder.kpc,
            current_masses=masses * dataholder.m_solar,
            smoothing_lengths=smoothing_lengths,
        )

    return galaxies