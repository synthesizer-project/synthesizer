import h5py
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from unyt import Msun, kpc, yr

from ..particle.galaxy import Galaxy
from synthesizer.load_data.utils import get_len

import numpy as np
import math
from bigfile import BigFile
import scipy.interpolate as interpolate
from scipy import integrate
import pickle
import os
import sys

class BlueTidesDataHolder:
    """
    Just a holder for BlueTides data that makes it easier to work with. Feel free to pickle the class as follows:
    file_name = f'bluetides_{z}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(BlueTidesDataHolder_instance, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Object successfully saved to "{file_name}"')
    """
    def __init__(self, z, bluetides_data_folder="/fred/oz183/sberger/bluetides/BlueTides/", end_of_arr=108001):
        ## Units
        self.Msolar = 1.99 * 10 ** 30  # unit: kg
        self.Lsolar = 3.826 * 10 ** 33  # unit: erg/s
        self.Gconstant = 6.67408 * 10 ** (-11)  # unit: m^3*kg^(-1)*s^(-2)
        self.mp = 1.673e-27  # kg
        self.Kpc = 3.086 * 10 ** 19  # unit: m
        self.kconstant = 1.38 * 10 ** (-23)  # unit: m^2*kg*s^(-2)*K^(-1)

        self.factor = 121.14740013761634  # GADGET unit Protonmass / Bolztman const
        self.GAMMA = 5 / 3.
        self.Xh = 0.76

        self.hh = 0.697  # hubble constant / 100 (dimensionless)
        self.c = 3e8  # m/s
        self.eta = 0.1
        self.yr = 365 * 24 * 3600

        self.Mpc_to_km = 3.086e+19
        self.sec_to_yr = 3.17098e-8  # convert to years
        self.omk = 0
        self.oml = 0.7186
        self.omm = 0.2814
        self.h = 0.697

        # helper functions
        def placed(number, offarray):
            dummy = np.where(offarray <= number)[0]
            return dummy[-1]

        def E(self, a):  # normalized hubble parameter
            return np.sqrt(self.omm * (a) ** (-3) + self.oml)

        def integrand(a):
            return 1.0 / (a * E(a))

        def age(a0, a1):
            return 1. / (self.h * 100.0) * integrate.quad(integrand, a0, a1)[0]

        def E(a):  # normalized hubble parameter
            return np.sqrt(self.omm * (a) ** (-3) + self.oml)

        self.accretion_unit_conversion = 1e10 / (self.Kpc / 1e3 / self.yr)  # to solar masses per year
        self.acrtolum = self.eta * self.c * self.c * (
                    self.Msolar / self.yr) * 10 ** 7  ##Accounts for unit conversion to physical units already happening
        self.z = z

        # Load all BlueTides data
        if self.z == 7:
            sunset = BigFile(bluetides_data_folder + 'sunset_208')
            pig_zoo = [bluetides_data_folder + 'PIG_208']
            length_of_BHAR = end_of_arr
        elif self.z == 6.5:
            sunset = BigFile(bluetides_data_folder + 'sunset_271')
            pig_zoo = [bluetides_data_folder + 'PIG_271']
            length_of_BHAR = end_of_arr
        elif self.z == 7.5:
            pig_zoo = [bluetides_data_folder + 'PIG_129']
            sunset = BigFile(bluetides_data_folder + 'sunset_129')
            length_of_BHAR = end_of_arr
        elif self.z == 8:
            pig_zoo = [bluetides_data_folder + 'PIG_086']
            sunset = BigFile(bluetides_data_folder + 'sunset_086')
            length_of_BHAR = end_of_arr
        else:
            print("z value not supported")
            exit()


        # Determine & store galaxy properties
        file = pig_zoo[0]  # filename
        pig = BigFile(file)  # load data

        print(f"Chosen redshift = {self.z}")
        self.z = self.redshift = float(1./pig.open('Header').attrs['Time']-1)
        print(f"Actual bluetides redshift = {self.z}")


        ##If there's no OffsetByType field, use below to get the same output:
        # get FOF info to know which stars correspond to which galaxies
        LengthByType = (pig.open('FOFGroups/LengthByType')[0:length_of_BHAR])
        OffsetByType = np.cumsum(LengthByType, axis=0)
        a1 = np.zeros((1, 6)).astype(int)
        OffsetByType = np.append(a1, OffsetByType, axis=0).astype(int)
        OffsetByType = np.transpose(OffsetByType)

        self.BHoffset = np.transpose(OffsetByType[5])
        self.StarOffset = StarOffset = np.transpose(OffsetByType[4])

        a_cur = 1. / (1. + self.redshift)
        SFT_space = np.linspace(0, a_cur, 100)
        Mpc_to_km = 3.086e+19
        sec_to_yr = 3.17098e-8
        age_space = [age(SFT, 1. / (1. + self.redshift)) * Mpc_to_km * sec_to_yr for SFT in SFT_space]  # yr
        self.gen_SFT_to_age = interpolate.interp1d(SFT_space, age_space, fill_value='extrapolate')

        self.BHAR = pig.open('5/BlackholeAccretionRate')[
                    0:length_of_BHAR] * self.accretion_unit_conversion  # solar masses per year


        self.luminous_BH_indices = (np.argsort(-self.BHAR))  # indexes of BH with largest accretion rate, everything is sorted by this array

        #### load galaxy black hole data
        self.quasarLuminosity = self.BHAR[
                                    self.luminous_BH_indices] * self.acrtolum  # transfer from accretion rate to AGN luminosity
        self.BHmass = pig.open('5/BlackholeMass')[0:length_of_BHAR][
                          self.luminous_BH_indices] * 1e10 / self.hh  # masses of BHs with largest accretion rate

        self.BHID = pig.open('5/ID')[0:length_of_BHAR][self.luminous_BH_indices]

        self.BHposition = np.transpose(pig.open('5/Position')[0:length_of_BHAR][self.luminous_BH_indices]) / self.hh / (
                    1 + self.z)

        self.BHAR = self.BHAR[self.luminous_BH_indices]

        #### Set particle location indices for the BHs in order in pig
        offsetIndex = []  ## index in offset
        for i in self.luminous_BH_indices:
            offsetIndex.append(placed(i, self.BHoffset))
        self.offsetIndex = np.array(offsetIndex)

        #### load stellar particle information
        self.metallicity_IND = pig.open('4/Metallicity')[
                               0:StarOffset[offsetIndex[np.argmax(StarOffset[offsetIndex])]]]
        self.starFormationTime_IND = pig.open('4/StarFormationTime')[
                                     0:StarOffset[offsetIndex[np.argmax(StarOffset[offsetIndex])]]]
        self.Position_IND = pig.open('4/Position')[
                            0:StarOffset[offsetIndex[np.argmax(StarOffset[offsetIndex])]]] / self.hh / (1 + self.z)
        self.Velocity_IND = pig.open('4/Velocity')[
                            0:StarOffset[offsetIndex[np.argmax(StarOffset[offsetIndex])]]] / self.hh
        self.MetSurfaceDensities_IND = sunset.open('4/MetSurfaceDensity')[
                                       0:StarOffset[offsetIndex[np.argmax(StarOffset[offsetIndex])]]]



def load_BlueTides(redshift, dataholder=None, galaxy_BHID=[], end_arr=108001, bluetides_data_folder=""):
    """
    Load BlueTides galaxies into a galaxy object

    Args:
        redshift (float):
            desired redshift to extract BlueTides galaxies
        dataholder (BlueTidesDataHolder object):
            contains all data for a particular simulation of BlueTides
            (default is None and dataholder class will be generated for the given redshift)
        galaxy_nums (array):
            contains all the galaxy BHIDs to include in your galaxy object
            Note: each BlueTides dataholder has arrays sorted in descending BHAR
            (default is empty list and all galaxies up to
        end_arr (int):
            last galaxy to include in BlueTides dataholder, selection of this value should be
            done with care to avoid galaxies with BHs too close to seed mass (10^5.8 solar masses)
            (default is 108001)
        bluetides_data_folder (str):
            location of BlueTides pig/sunset files, only needed if dataholder=None
            (default is "")

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing specified
            stars and gas objects
    """
    if dataholder == None:
        print(f"Loading in BlueTides data from z = {redshift}...")
        dataholder = BlueTidesDataHolder(redshift, bluetides_data_folder=bluetides_data_folder, end_of_arr=end_arr)

    galaxies_length = len(dataholder.BHmass)  # holder array for galaxies of the same length
    galaxies = [None] * galaxies_length
    smoothing_length_comoving_bluetides = 1.5 / dataholder.hh * dataholder.Kpc

    if len(galaxy_BHID) == 0:
        galaxy_cycle = np.arange(galaxies_length) # take every galaxy
    else:
        # get galaxies with the BHIDs you desire since the indices are different for each redshift
        _, bh_indices, _ = np.intersect1d(dataholder.BHID, galaxy_BHID, return_indices=True)
        galaxy_cycle = bh_indices

    #### grab stellar particles associated with each galaxy and assign to galaxy in galaxies array
    for ii, bh_index in enumerate(galaxy_cycle):
        idx = int(dataholder.offsetIndex[bh_index]) # this gives you the first index of the star particle for the particular galaxy
        star_off = dataholder.StarOffset[idx:idx + 2]
        galaxies[ii] = Galaxy()
        galaxies[ii].redshift = dataholder.redshift

        star_Time = dataholder.starFormationTime_IND[
                    star_off[0]:star_off[1]]
        Ages = dataholder.gen_SFT_to_age(star_Time) / 1e6

        initial_gas_particle_mass = np.full(Ages.shape, 2.36e6) / dataholder.hh
        imasses = initial_gas_particle_mass / 4
        Masses = np.ones(Ages.shape) * 1E10 * 5.90556119E-05 / 0.697  # mass of each star particle in M_sol
        Metallicities = dataholder.metallicity_IND[star_off[0]:star_off[1]]

        star_pos = (dataholder.Position_IND[star_off[0]:star_off[1]]) * (
                    1 + dataholder.redshift)  # Position_IND is in proper, so converting back to comoving

        ### centering all stars around the BH such that (0, 0, 0) is the BH position 
        star_relpos = star_pos - (dataholder.BHposition[:, bh_index] * (1 + dataholder.redshift))  # same for BH_position
        X = star_relpos[:, 0]
        Y = star_relpos[:, 1]
        Z = star_relpos[:, 2]

        smoothing_lengths = np.full(Ages.shape, smoothing_length_comoving_bluetides)

        coords = np.transpose([X, Y, Z])
        galaxies[ii].load_stars(
            initial_masses=imasses * dataholder.Msolar,
            ages=Ages * dataholder.yr,
            metals=Metallicities,
            #         s_oxygen=s_oxygen[b:e],
            #         s_hydrogen=s_hydrogen[b:e],
            coordinates=coords * dataholder.Kpc,
            current_masses=Masses * dataholder.Msolar,
            smoothing_lengths=smoothing_lengths,
        )

    return galaxies

