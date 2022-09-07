


import numpy as np

import os
this_dir, this_filename = os.path.split(__file__)

import pandas as pd

import matplotlib.pyplot as plt
import cmasher as cmr




class FilterCollection:

    """ Create a collection of filters. While this could be just a dictionary or list the class includes methods for also making plots. """

    def __init__(self, fs, new_lam = False, read_from_SVO = True, local_filter_path = f'{this_dir}/data/filters'):

        self.filter = {}
        for f in fs:
            F = Filter(f, new_lam = new_lam, read_from_SVO = read_from_SVO, local_filter_path = local_filter_path)
            self.filter[F.filter_code] = F
        self.filters = list(self.filter.keys())


    def transmission_curve_ax(self, ax, add_filter_label = True):

        """ add filter transmission curves to a give axes """

        # --- add colours

        for filter_code, F in self.filter.items():
            ax.plot(F.l, F.t)

            # --- add label with automatic placement
            #

        ax.set_xlabel(r'$\rm \lambda/\AA$')
        ax.set_ylabel(r'$\rm T_{\lambda}$')


    def plot_transmission_curves(self, show = True):

        """ Create a filter transmission curve plot """

        fig = plt.figure(figsize = (5., 3.5))

        left  = 0.1
        height = 0.8
        bottom = 0.15
        width = 0.85

        ax = fig.add_axes((left, bottom, width, height))

        self.transmission_curve_ax(ax, add_filter_label = True)

        if show: plt.show()

        return fig, ax



class Filter:

    """
    A class representing a filter.

    Attributes
    ----------
    filter_code : str
        filter code of the filter {observatory}/{instrument}.{filter}
    f : str
        filter code tuple (observatory, instrument, filter)


    Methods
    -------
    pivwv:
        Calculate pivot wavelength
    """


    def __init__(self, f, new_lam = False, read_from_SVO = True, local_filter_path = f'{this_dir}/data/filters'):


        if type(f) == tuple:
            self.f = f
            self.observatory, self.instrument, self.filter = f
            self.filter_code = f'{self.observatory}/{self.instrument}.{self.filter}'
        else:
            self.filter_code = f
            self.observatory = f.split('/')[0]
            self.instrument = f.split('/')[1].split('.')[0]
            self.filter = f.split('.')[-1]
            self.f = (self.observatory, self.instrument, self.filter)

        if read_from_SVO:
            """ read directly from the SVO archive """
            svo_url = f'http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id={self.observatory}/{self.instrument}.{self.filter}'
            df = pd.read_csv(svo_url, sep = ' ')
            self.original_lam = df.values[:,0] # original wavelength
            self.original_t = df.values[:,1] # original transmission
        else:
            """ read from local files instead """
            # l, t are the original wavelength and transmission grid
            self.original_lam, self.original_t = np.loadtxt(f'{local_filter_path}/{self.filter_code}.dat').T

        # self.original_t[self.original_t<0.05] = 0

        # --- if a new wavelength grid is provided interpolate the transmission curve on to that grid
        if isinstance(new_lam, np.ndarray):
            self.lam = new_lam
            self.t = np.interp(self.lam, self.original_lam, self.original_t, left = 0.0, right = 0.0) # setting left and right is necessary for when the filter transmission function is mapped to a new wavelength grid
        else:
            self.lam = self.original_lam
            self.t = self.original_t


    #http://stsdas.stsci.edu/stsci_python_epydoc/SynphotManual.pdf   #got to page 42 (5.1)

    def pivwv(self): # -- calculate pivot wavelength using original l, t

        """ calculate the pivot wavelength """

        return np.sqrt(np.trapz(self.original_lam * self.original_t , x = self.original_lam)/np.trapz(self.original_t / self.original_lam, x = self.original_lam))


    def pivT(self): # -- calculate pivot wavelength using original l, t

        """ calculate the transmission at the pivot wavelength """

        return np.interp(self.pivwv(), self.original_lam, self.original_t)

    def meanwv(self):

        """ calculate the mean wavelength """

        return np.exp(np.trapz(np.log(self.original_lam) * self.original_t / self.original_lam, x = self.original_lam)/np.trapz(self.original_t / self.original_lam, x = self.original_lam))

    def bandw(self):

        """ calculate the bandwidth """

        return self.meanwv() * np.sqrt(np.trapz((np.log(self.original_lam/self.meanwv())**2)*self.original_t/self.original_lam,x=self.original_lam))/np.sqrt(np.trapz(self.original_t/self.original_lam,x=self.original_lam))

    def fwhm(self):

        """ calculate the FWHM """

        return np.sqrt(8.*np.log(2))*self.bandw()

    def Tpeak(self):

        """ calculate the peak transmission """

        return np.max(self.original_t)

    def rectw(self):

        """ calculate the rectangular width """

        return np.trapz(self.original_t, x=self.original_lam)/self.Tpeak()


    def max(self):

        """ calculate the longest wavelength where the transmission is still >0.01 """

        return self.original_lam[self.original_t>1E-2][-1]

    def min(self):

        """ calculate the shortest wavelength where the transmission is still >0.01 """

        return self.original_lam[self.original_t>1E-2][0]

    def mnmx(self):

        """ calculate the minimum and maximum wavelengths """

        return (self.min(), self.max())
