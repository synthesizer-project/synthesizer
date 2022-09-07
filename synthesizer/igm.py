 #from pylab import*

from numpy import *
import numpy as np

# --- calculates the absorption due to the intergalactic medium using e.g. Madau et al. formalism.




# --- there are alernatives to the Madau 96 formalism which we should add.





def madau96(lambdas, z):

	"""
    Return the IGM absorption for the Madau96 model as a function of wavelength and redshift

            Parameters:
                    lambdas (ndarray): wavelength grid in \AA
                    z (float): redshift

            Returns:
                    Expteff (ndarray): IGM transmission as a function of wavelength
    """

	# --- NB this was written by an undergraduate student ~10 years ago. Should be re-written.

	LAMBDA= [1216.0,1026.0,973.0,950.0]
	A=[0.0036,0.0017,0.0012,0.00093]

	if not isinstance(lambdas, (list, np.ndarray)):

		l=lambdas
		if l>LAMBDA[0]*(1+z):
			return 1
		if l<=LAMBDA[-1]*(1+z)-2000:
			return 0

		teff=0
		for i in range(0,len(LAMBDA)-1,1):
			teff+=A[i]*(l/LAMBDA[i])**3.46
			if LAMBDA[i+1]*(1+z)<l<=LAMBDA[i]*(1+z):
				return exp(-teff)

		return 	exp(-(teff+ 0.25*(l/LAMBDA[-1])**3*((1+z)**0.46-(l/LAMBDA[-1])**0.46)+9.4*(l/LAMBDA[-1])**1.5*((1+z)**0.18-(l/LAMBDA[-1])**0.18)-0.7*(l/LAMBDA[-1])**3*((l/LAMBDA[-1])**(-1.32)-(1+z)**(-1.32))+0.023*((l/LAMBDA[-1])**1.68-(1+z)**1.68)))


	Expteff=array([])
	for l in lambdas:

		if l>LAMBDA[0]*(1+z):
			Expteff=append(Expteff,1)
			continue

		if l<=LAMBDA[-1]*(1+z)-1500:
			Expteff=append(Expteff,0)
			continue

		teff=0
		for i in range(0,len(LAMBDA)-1,1):
			teff+=A[i]*(l/LAMBDA[i])**3.46
			if LAMBDA[i+1]*(1+z)<l<=LAMBDA[i]*(1+z):
				Expteff=append(Expteff,exp(-teff))
				continue

		if l<=LAMBDA[-1]*(1+z):
			Expteff=append(Expteff,exp(-(teff+ 0.25*(l/LAMBDA[-1])**3*((1+z)**0.46-(l/LAMBDA[-1])**0.46)+9.4*(l/LAMBDA[-1])**1.5*((1+z)**0.18-(l/LAMBDA[-1])**0.18)-0.7*(l/LAMBDA[-1])**3*((l/LAMBDA[-1])**(-1.32)-(1+z)**(-1.32))+0.023*((l/LAMBDA[-1])**1.68-(1+z)**1.68))))
			continue

	return Expteff
