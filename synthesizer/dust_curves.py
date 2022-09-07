


import os
this_dir, this_filename = os.path.split(__file__)

import numpy as np
from scipy import interpolate


curves = ['simple','MW_N18','MW_Pei92','SMC_Pei92','Starburst_Calzetti2000']
named_curves = ['MW_N18','MW_Pei92','SMC_Pei92','Starburst_Calzetti2000']


class power_law():

    def __init__(self, params = {'slope':-1.}):

        self.description = 'simple power law dust curve'
        self.params = params

    def tau(self, lam):

        return (lam/5500.)**self.params['slope']


class MW_N18():

    def __init__(self, params = {}):

        self.description = 'MW extinction curve from Desika'

        self.d = np.load(f'{this_dir}/data/MW_N18.npz')

        self.tau_lam_V = np.interp(5500., self.d.f.mw_df_lam[::-1], self.d.f.mw_df_chi[::-1])

    def tau(self, lam, interp = 'cubic'):

        f = interpolate.interp1d(self.d.f.mw_df_lam[::-1], self.d.f.mw_df_chi[::-1], kind = interp, fill_value = 'extrapolate')

        return f(lam)/self.tau_lam_V


class MW_Pei92():


    def __init__(self, params = {}):

        self.description = 'Empirical MW extiction curve taken from Pei et al. 1992'

        self.R_V = 3.08

        self.x  = np.array([0.21,0.29,0.45,0.61,0.80,1.11,1.43,1.82,2.27,2.50,2.91,3.65,4.00,4.17,4.35,4.57,4.76,5.00,5.26,5.56,5.88,6.25,6.71,7.18,7.60,8.00,8.50,9.00,9.50,10.00])

        self.ElvEbv = np.array([-3.02,-2.91,-2.76,-2.58,-2.23,-1.60,0.78,0.0,1.0,1.3,1.8,3.10,4.19,4.9,5.77,6.57,6.23,5.52,4.9,4.65,4.60,4.73,4.99,5.36,5.91,6.55,7.45,8.45,9.80,11.30])

    def tau(self, lam, interp = 'cubic'):

        lam_mu = lam/1E4

        x=1./lam_mu # l in mu m

        f = interpolate.interp1d(self.x, self.ElvEbv, kind = interp, fill_value = 'extrapolate')

        k = f(x) + self.R_V

        k_V = f(1/0.55) + self.R_V

        return (k/k_V)*0.4/0.43


class SMC_Pei92():


    def __init__(self, params = {}):

        self.description = 'Empirical SMC extiction curve taken from Pei et al. 1992'

        self.R_V = 2.93
        self.x = np.array([0.45,0.61,0.8,1.82,2.27,2.70,3.22,3.34,3.46,3.60,3.75,3.92,4.09,4.28,4.50,4.73,5.00,5.24,5.38,5.52,5.70,5.88,6.07,6.27,6.48,6.72,6.98,7.23,7.52,8.22])
        self.ElvEbv = np.array([-2.61,-2.47,-2.12,0.0,1.00,1.67,2.29,2.65,3.00,3.15,3.49,3.91,4.24,4.53,5.30,5.85,6.38,6.76,6.90,7.17,7.71,8.01,8.49,9.06,9.28,9.84,10.80,11.51,12.52,13.54])


    def tau(self, lam, interp = 'slinear'):

        lam_mu = lam/1E4

        x=1./lam_mu # l in mu m

        f = interpolate.interp1d(self.x, self.ElvEbv, kind = interp, fill_value = 'extrapolate')

        k = f(x) + self.R_V

        k_V = f(1/0.55) + self.R_V

        return (k/k_V)*0.4/0.43



class Starburst_Calzetti2000():

    def __init__(self, params = {}):

        self.description = 'Starburst ATTENUATION curve from Calzetti 2000'
        self.R_V = 4.05


    def tau(self, lam):

        lam_mu = lam/1E4

        x=1./lam_mu #l in mu m

        k = np.zeros(x.shape)

        k[lam_mu>0.63] = 2.659*(-1.857 + 1.040*x[lam_mu>0.63]) + self.R_V
        k[lam_mu<=0.63] = 2.659*(-2.156+1.509*x[lam_mu<=0.63]-0.198*x[lam_mu<=0.63]**2+0.011*x[lam_mu<=0.63]**3) + self.R_V

        k_V = 2.659*(-2.156+1.509*(1/0.55)-0.198*(1/0.55)**2+0.011*(1/0.55)**3) + self.R_V

        return (k/k_V)*0.4/0.43





# class MW_Cardelli89():
#
#
#     def __init__(self, params = {}):
#
#         self.description = 'Cardelli89 MW'
#         self.R_V = 3.1
#
#     def tau(self, lam):
#
#         lam_mu = lam/1E4
#
#         x=1./lam_mu #l in mu m
#
#
#         a = np.zeros(x.shape)
#         b = np.zeros(x.shape)
#
#         s = x<1.1
#         a[s] = 0.574*x[s]**1.61
#         b[s] = -0.527*x[s]**1.61
#
#         s = (x>1.1)&(x<3.3)
#         y = x[s] - 1.82
#         a[s] = (1.)+(0.17699*y)-(0.50447*y**2)-(0.02427*y**3)+(0.72085*y**4)+(0.01979*y**5)-(0.77530*y**6)+(0.32999*y**7)
#         b[s] = (1.41338*y)+(2.28305*y**2)+(1.07233*y**3)-(5.38434*y**4)-(0.62261*y**5)+(5.30260*y**6)-(2.09002*y**7)
#
#         s = x>=3.3
#
#         a[s] = 1.752-0.316*x[s]-0.104/((x[s]-4.67)**2+0.263)
#         b[s] = -2.090+1.825*x[s]+1.206/((x[s]-4.62)**2+0.263)
#
#         k = a+b/(self.R_V)/(1.337-1.) #hopefully
#
#         k_V = np.interp(0.55, lam_mu, k)
#
#         return (k/k_V)*0.4/0.43
