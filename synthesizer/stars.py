import warnings

import numpy as np

from .particles import Particles


class Stars(Particles):
    def __init__(self, masses, ages, metallicities, **kwargs):
        self.masses = masses
        self.ages = ages
        self.metallicities = metallicities

        self.tauV = None  # V-band dust optical depth
        self.alpha = None  # alpha-enhancement [alpha/Fe]
        self.a3 = None

        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        self.resampled = False

        self.attributes = ['masses', 'ages', 'metallicities',
                           'log10ages', 'log10metallicities']

        if 'coordinates' in kwargs.keys():
            self.coordinates = kwargs['coordinates']
            self.attributes.append('coordinates')

        if 'initial_masses' in kwargs.keys():
            self.initial_masses = kwargs['initial_masses']
            self.attributes.append('initial_masses')

        if 's_oxygen' in kwargs.keys():
            self.abundance_oxygen = kwargs['s_oxygen']
            self.attributes.append('abundance_oxygen')

        if 's_hydrogen' in kwargs.keys():
            self.abundance_hydrogen = kwargs['s_hydrogen']
            self.attributes.append('abundance_hydrogen')

    def _power_law_sample(self, a, b, g, size=1):
        """
        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b
        """
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    def resample_young_stars(self, min_age=1e8, min_mass=700, max_mass=1e6,
                             power_law_index=-1.3, n_samples=1e3,
                             force_resample=False, verbose=False):
        """
        Resample young stellar particles into HII regions, with a
        power law distribution of masses
        """
        if self.resampled & (~force_resample):
            warnings.warn("Warning, galaxy stars already resampled. \
                    To force resample, set force_resample=True. Returning...")
            return None

        if verbose:
            print("masking resample stars")

        resample_idxs = np.where(self.ages < min_age)[0]

        if len(resample_idxs) == 0:
            return None

        new_ages = {}
        new_masses = {}

        if verbose:
            print("loop through resample stars")

        for _idx in resample_idxs:
            rvs = self._power_law_sample(min_mass, max_mass,
                                         power_law_index, int(n_samples))

            # ---- if not enough mass sampled, sample again
            while np.sum(rvs) < self.masses[_idx]:
                n_samples *= 2
                rvs = self._power_law_sample(min_mass, max_mass,
                                             power_law_index, int(n_samples))

            # --- sum masses up to total mass limit
            _mask = np.cumsum(rvs) < self.masses[_idx]
            _masses = rvs[_mask]

            # ---- scale up to original mass
            _masses *= (self.masses[_idx] / np.sum(_masses))

            # sample uniform distribution of ages
            _ages = np.random.rand(len(_masses)) * min_age

            new_ages[_idx] = _ages
            new_masses[_idx] = _masses

        new_lens = [len(new_ages[_idx]) for _idx in resample_idxs]
        new_ages = np.hstack([new_ages[_idx] for _idx in resample_idxs])
        new_masses = np.hstack([new_masses[_idx] for _idx in resample_idxs])

        if verbose:
            print("concatenate new arrays to existing")

        for attr, new_arr in zip(['masses', 'ages'],
                                 [new_masses, new_ages]):
            attr_array = getattr(self, attr)
            setattr(self, attr, np.append(attr_array, new_arr))

        if verbose:
            print("duplicate existing attributes")

        gen = (attr for attr in self.attributes if attr not in ['masses', 'ages'])
        for attr in gen:
            attr_array = getattr(self, attr)[resample_idxs]
            setattr(self, attr, np.append(getattr(self, attr),
                                          np.repeat(attr_array, new_lens, axis=0)))

        if verbose:
            print("delete old particles")

        for attr in self.attributes:
            attr_array = getattr(self, attr)
            attr_array = np.delete(attr_array, resample_idxs)
            setattr(self, attr, attr_array)

        # ---- recalculate log attributes
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # ---- set resampled flag
        self.resampled = True
