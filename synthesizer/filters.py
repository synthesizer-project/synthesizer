


import numpy as np

import os
this_dir, this_filename = os.path.split(__file__)

import pandas as pd

import matplotlib.pyplot as plt
import cmasher as cmr





def add_filters(filters, new_lam = False,  filter_path = f'{this_dir}/data/filters/'):

    F = {f: filter(f, new_lam, filter_path) for f in filters}

    F['filters'] = filters

    return F


class FilterCollection:

    def __init__(self, fs, new_lam = False, read_from_SVO = True, local_filter_path = f'{this_dir}/data/filters'):

        """ create a collection of filters """

        self.filter = {}
        for f in fs:
            F = Filter(f, new_lam = new_lam, read_from_SVO = read_from_SVO, local_filter_path = local_filter_path)
            self.filter[F.filter_code] = F
        self.filters = list(self.filter.keys())


    def transmission_curve_ax(self, ax):

        for filter_code, F in self.filter.items():
            ax.plot(F.l, F.t)

        ax.set_xlabel(r'$\rm \lambda/\AA$')
        ax.set_ylabel(r'$\rm T_{\lambda}$')


    def plot_transmission_curves(self, show = True):

        fig = plt.figure(figsize = (5., 3.5))

        left  = 0.1
        height = 0.8
        bottom = 0.15
        width = 0.85

        ax = fig.add_axes((left, bottom, width, height))

        self.transmission_curve_ax(ax)
        
        if show: plt.show()

        return fig, ax



class Filter:

    def __init__(self, f, new_lam = False, read_from_SVO = True, local_filter_path = f'{this_dir}/data/filters'):

        """ initialise a filter object. The new naming scheme uses the SVO (http://svo2.cab.inta-csic.es/theory/fps/) naming scheme
            filter_name = {observatory}/{instrument}.{filter}.dat
        """

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
            self.l = df.values[:,0]
            self.t = df.values[:,1]
        else:
            """ read from local files instead """
            # l, t are the original wavelength and transmission grid
            self.l, self.t = np.loadtxt(f'{local_filter_path}/{self.filter_code}.dat').T

        self.t[self.t<0] = 0

        # if a new wavelength grid is provided interpolate the transmission curve on to that grid
        if isinstance(new_lam, np.ndarray):
            self.lam = new_lam
            self.T = np.interp(self.lam, self.l, self.t)
        else:
            self.lam = self.l
            self.T = self.t


    #http://stsdas.stsci.edu/stsci_python_epydoc/SynphotManual.pdf   #got to page 42 (5.1)

    def pivwv(self): # -- calculate pivot wavelength using original l, t

        """ calcualate the pivot wavelength """

        return np.sqrt(np.trapz(self.l * self.t , x = self.l)/np.trapz(self.t / self.l, x = self.l))


    def pivT(self): # -- calculate pivot wavelength using original l, t

        return np.interp(self.pivwv(), self.l, self.t)

    def meanwv(self):

        return np.exp(np.trapz(np.log(self.l) * self.t / self.l, x = self.l)/np.trapz(self.t / self.l, x = self.l))

    def bandw(self):

        return self.meanwv() * np.sqrt(np.trapz((np.log(self.l/self.meanwv())**2)*self.t/self.l,x=self.l))/np.sqrt(np.trapz(self.t/self.l,x=self.l))

    def fwhm(self):

        return np.sqrt(8.*np.log(2))*self.bandw()

    def Tpeak(self):

        return np.max(self.t)

    def rectw(self):

        return np.trapz(self.t, x=self.l)/self.Tpeak()

    def mnmx(self):

        return (self.min(), self.max())

    def max(self):

        return self.l[self.T>1E-2][-1]

    def min(self):

        return self.l[self.T>1E-2][0]








#
# # ----------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------
#
# # below is deprecated in favour of the observatories sub-module
#
#
# # --- artificial
#
# FAKE = ['FAKE.FAKE.'+f for f in ['1500','2500','Uth','Bth','Vth','Ith','Zth','Yth','Jth','Hth']]
# TH = ['FAKE.TH.'+f for f in ['FUV','MUV', 'NUV','U','B','V','R','I','Z','Y','J','H','K']]
#
# # --- Roman
#
# Roman = ['Roman.Roman.'+f for f in ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184']]
# WFIRST = Roman
#
# # --- Hubble
#
# WFC3UVIS_W = ['Hubble.WFC3.'+f for f in ['f225w','f275w','f336w']]
# WFC3NIR_W = ['Hubble.WFC3.'+f for f in ['f105w', 'f125w', 'f140w', 'f160w']]
# WFC3 = WFC3UVIS_W + WFC3NIR_W
#
# ACS_W = ['HST.ACS.'+f for f in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]
# ACS = ACS_W
#
# HST = ACS + WFC3
# Hubble = HST
#
# # --- Spitzer
#
# # IRAC = ['Spitzer.IRAC.'+f for f in ['ch1', 'ch2', 'ch3', 'ch4']]
# IRAC = ['Spitzer.IRAC.'+f for f in ['ch1', 'ch2']]
# Spitzer = IRAC
#
# # --- Euclid
#
# Euclid_VIS = ['Euclid.VIS.'+f for f in ['VIS']]
# Euclid_NISP = ['Euclid.NISP.'+f for f in ['Y','J','H']]
# Euclid = Euclid_VIS + Euclid_NISP
#
# # --- Subaru
#
# HSC = ['Subaru.HSC.'+f for f in ['g','r','i','z','y']]
# Subaru = HSC
#
# # --- Webb
#
# NIRCam_s_W = ['JWST.NIRCAM.'+f for f in ['F070W','F090W','F115W','F150W','F200W']]
# NIRCam_s_M = ['JWST.NIRCAM.'+f for f in ['F140M','F162M','F182M','F210M']]
# NIRCam_l_W = ['JWST.NIRCAM.'+f for f in ['F277W','F356W','F444W']]
# NIRCam_l_M = ['JWST.NIRCAM.'+f for f in ['F250M','F300M','F360M','F410M','F430M','F460M','F480M']]
# NIRCam_s = NIRCam_s_W + NIRCam_s_M
# NIRCam_l = NIRCam_l_W + NIRCam_l_M
# NIRCam_W = NIRCam_s_W + NIRCam_l_W
# NIRCam =  NIRCam_s + NIRCam_l
# MIRI = ['JWST.MIRI.'+f for f in ['F560W','F770W','F1000W','F1130W','F1280W','F1500W','F1800W','F2100W','F2550W']]
# Webb = NIRCam + MIRI
# JWST = Webb
#
# CEERS = ['JWST.NIRCAM.'+f for f in ['F115W','F150W','F200W', ]]
#
# # --- All *real* filters
#
# all_filters = FAKE + TH + Hubble + Spitzer + Euclid + Subaru + Webb + Roman
