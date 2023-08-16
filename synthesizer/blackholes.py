
import numpy as np
from scipy.integrate import quad
from unyt import c, Msun, yr, Angstrom, K, unyt_array




def calculate_Lbol(mdot, epsilon=0.1):

    """
    Create the black hole bolometric luminosity

    Parameters
    ----------
    mdot: unyt_array or float
        The black hole accretion rate. If not unyt_array assume Msun/yr

    epsilon: float
        The radiative efficiency, by default 0.1.

    Returns
    ----------
    unyt_array
        The black hole bolometric luminosity
    
    """

    # if not a unyt_array instance convert to one assume the value is Msol/yr
    if not isinstance(mdot, unyt_array):
        mdot *= Msun/yr

    return epsilon * mdot * c**2

def calculate_TBB(mbh, mdot):

    """
    Create the black hole "Big bump" temperature

    Parameters
    ----------
    mbh: unyt_array or float
        The black hole mass. If not unyt_array assume Msun.

    mdot: unyt_array or float
        The black hole accretion rate. If not unyt_array assume Msun/yr.

    Returns
    ----------
    unyt_array
        The black hole "big bump" temperature
    """

    # if mbh not a unyt_array instance convert to one assume the value is Msol
    if not isinstance(mbh, unyt_array):
        mbh *= Msun

    # if mdot not a unyt_array instance convert to one assume the value is Msol/yr
    if not isinstance(mdot, unyt_array):
        mdot *= Msun/yr

    return 2.24E9 * (mdot.to('Msun/yr').value)**(1/4) * (mbh.to('Msun').value)**(-1/2) * K






class Cloudy:

    """
    A class to hold routines for employing the Cloudy AGN model.
    """

    def __init__(self):
        return None




class Feltre16:

    """
    A class to hold routines for employing the Feltre16 AGN model.
    """

    def __init__(self):
        return None


    def intrinsic(lam, alpha, luminosity = 1):

        """
        Create intrinsic narrow-line AGN spectra as utilised by Feltre et al. (2016). This is utilised to build the cloudy grid.

        Parameters
        ----------
        lam : array
            Wavelength grid (array) in angstrom or unyt 

        alpha: float
            UV/optical power-law index. Expected to be -2.0<alpha<-1.2

        luminosity: float
            Bolometric luminosity. Set to unity. 

            
        Returns
        -------
        lnu
            Spectral luminosity density.
        """

        # create empty luminosity array
        lnu = np.zeros(lam.shape)
        
        # calculate frequency
        nu = c/lam

        # define edges
        edges = [10., 2500., 100000., 1000000.] * Angstrom  # Angstrom

        # define indices
        indices = [alpha, -0.5, 2.]

        # define normalisations
        norms = [1.]

        # calcualte remaining normalisations
        for i, (edge_lam, ind1, ind2) in enumerate(zip(edges[1:], indices, indices[1:])):

            edge_nu = c/edge_lam

            norm = norms[i]*(edge_nu**ind1)/(edge_nu**ind2)
            norms.append(norm)

        # now construct spectra
        for e1, e2, ind, norm in zip(edges[0:], edges[1:], indices, norms):

            # identify indices within the wavelength range
            s = (lam>=e1)&(lam<e2)

            lnu[s] = norm * nu[s]**ind

        # normalise -- not yet implemented

        return lnu

