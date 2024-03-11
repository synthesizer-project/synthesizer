import os
import numpy as np
from scipy import interpolate
from unyt import Angstrom, unyt_quantity, unyt_array

from dust_extinction.grain_models import WD01

from synthesizer import exceptions


this_dir, this_filename = os.path.split(__file__)

__all__ = ["PowerLaw", "MW_N18", "Calzetti2000", "GrainsWD01", "LMC_M23"]

def N09_tau(lam, slope, cent_lam, ampl, gamma):
    """
    Attenuation curve using a modified version of the Calzetti
    attenuation (Calzetti+2000) law allowing for a varying UV slope
    and the presence of a UV bump; from Noll+2009

    Args:
        lam (array-like, float)
            The input wavelength array (expected in AA units, 
            global unit).

        slope (float)
            The slope of the attenuation curve.

        cent_lam (float)
            The central wavelength of the UV bump, expected in microns.

        ampl (float)
            The amplitude of the UV-bump.

        gamma (float)
            The width (FWHM) of the UV bump, in microns.

    Returns:
        (array-like, float)
            V-band normalised optical depth for given wavelength
    """

    # Wavelength in microns
    if isinstance(lam, (unyt_quantity, unyt_array)):
        lam_micron = lam.to('um').v
    else:
        lam_micron = lam/1e4
        
    lam_v = 0.55
    k_lam = np.zeros_like(lam_micron)

    # Masking for different regimes in the Calzetti curve
    ok1 = (lam_micron >= 0.12) * (lam_micron < 0.63) # 0.12um<=lam<0.63um
    ok2 = (lam_micron >= 0.63) * (lam_micron < 31.) # 0.63um<=lam<=31um
    ok3 = (lam_micron < 0.12) # lam<0.12um
    if np.sum(ok1) > 0: # equation 1
        k_lam[ok1] = -2.156 + (1.509 / lam_micron[ok1]) \
            - (0.198 / lam_micron[ok1]**2) \
            + (0.011 / lam_micron[ok1]**3)
        func = interpolate.interp1d(lam_micron[ok1], k_lam[ok1],
                                fill_value="extrapolate")
    if np.sum(ok2) > 0: # equation 2
        k_lam[ok2] = -1.857 + (1.040 / lam_micron[ok2])
    if np.sum(ok3) > 0:
        # Extrapolating the 0.12um<=lam<0.63um regime
        k_lam[ok3] = func(lam_micron[ok3])
    
    # Using the Calzetti attenuation curve normalised 
    # to Av=4.05
    k_lam = 4.05 + 2.659 * k_lam
    k_v = 4.05 + 2.659 * (-2.156 + (1.509 / lam_v)
        - (0.198 / lam_v**2)
        + (0.011 / lam_v**3))

    # UV bump feature expression from Noll+2009
    D_lam = ampl * ((lam_micron * gamma) ** 2) \
        / ((lam_micron ** 2 - cent_lam ** 2) ** 2 + (lam_micron * gamma) ** 2)
    
    # Normalising with the value at 0.55um, to obtain 
    # normalised optical depth
    tau_x_v = (k_lam + D_lam)/k_v              
    
    print()
    
    return tau_x_v * (lam_micron/lam_v)**slope


class AttenuationLaw:
    """
    A generic parent class for dust attenuation laws to hold common attributes
    and methods

    Attributes:
        description (str)
            A description of the type of model. Defined on children classes.
    """

    def __init__(self, description):
        """
        Initialise the parent and set common attributes.
        """

        # Store the description of the model.
        self.description = description

    def get_tau(self, *args):
        """
        A prototype method to be overloaded by the children defining various
        models.
        """
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models ("
            + ", ".join(__all__) + ")"
        )

    def get_transmission(self, tau_v, lam):
        """
        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/array-like, float)
                Optical depth in the V-band. Can either be a single float or
                array.

            lam (array-like, float)
                The wavelengths (with units) at which to calculate transmission.

        Returns:
            array-like
                The transmission at each wavelength. Either (lam.size,) in shape
                for singular tau_v values or (tau_v.size, lam.size) tau_v
                is an array.
        """

        # Get the optical depth at each wavelength
        tau_x_v = self.get_tau(lam)

        # Include the V band optical depth in the exponent
        # For a scalar we can just multiply but for an array we need to
        # broadcast
        if np.isscalar(tau_v):
            exponent = tau_v * tau_x_v
        else:
            exponent = tau_v[:, None] * tau_x_v

        return np.exp(-exponent)


class PowerLaw(AttenuationLaw):
    """
    Custom power law dust curve.

    Attributes:
        slope (float)
            The slope of the power law.
    """

    def __init__(self, slope=-1.0):
        """
        Initialise the power law slope of the dust curve.

        Args:
            params (dict)
                A dictionary containing the parameters for the model.
        """

        description = "simple power law dust curve"
        AttenuationLaw.__init__(self, description)
        self.slope = slope

    def get_tau_at_lam(self, lam):
        """
        Calculate optical depth at a wavelength.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/array-like, float
                The optical depth.
        """

        return (lam / 5500.0) ** self.slope

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/array-like, float
                The optical depth.
        """

        return self.get_tau_at_lam(lam) / self.get_tau_at_lam(5500.0)


class MW_N18(AttenuationLaw):
    """
    Milky Way attenuation curve used in Narayanan+2018.

    Attributes:
        data (array-like, float)
            The data describing the dust curve, loaded from MW_N18.npz.
        tau_lam_v (float)
            The V band optical depth.
    """

    def __init__(self):
        """
        Initialise the dust curve by loading the data and get the V band
        optical depth by interpolation.
        """

        description = "MW extinction curve from Desika"
        AttenuationLaw.__init__(self, description)
        self.data = np.load(f"{this_dir}/../data/MW_N18.npz")
        self.tau_lam_v = np.interp(
            5500.0, self.data.f.mw_df_lam[::-1], self.data.f.mw_df_chi[::-1]
        )

    def get_tau(self, lam, interp="cubic"):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str)
                The type of interpolation to use. Can be ‘linear’, ‘nearest’,
                ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and
                ‘cubic’ refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float
                The optical depth.
        """

        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )

        if isinstance(lam, (unyt_quantity, unyt_array)):
            _lam = lam.to('Angstrom').v
        else:
            _lam = lam

        return func(lam) / self.tau_lam_v


class Calzetti2000(AttenuationLaw):
    """
    Calzetti attenuation curve; with option for the slope and UV-bump
    implemented in Noll et al. 2009.

    Attributes:
        slope (float)
            The slope of the attenuation curve.

        cent_lam (float)
            The central wavelength of the UV bump, expected in microns.

        ampl (float)
            The amplitude of the UV-bump.

        gamma (float)
            The width (FWHM) of the UV bump, in microns.

    """

    def __init__(self, slope=0, cent_lam=0.2175, ampl=0, gamma=0.035):
        """
        Initialise the dust curve.

        Args:
            slope (float)
                The slope of the attenuation curve.

            cent_lam (float)
                The central wavelength of the UV bump, expected in microns.

            ampl (float)
                The amplitude of the UV-bump.

            gamma (float)
                The width (FWHM) of the UV bump, in microns.
        """
        description = (
            "Calzetti attenuation curve; with option"
            "for the slope and UV-bump implemented"
            "in Noll et al. 2009"
        )
        AttenuationLaw.__init__(self, description)

        # Define the parameters of the model.
        self.slope = slope
        self.cent_lam = cent_lam
        self.ampl = ampl
        self.gamma = gamma

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth. (Uses the N09_tau function
        defined above.)

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        return N09_tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )


class GrainsWD01:
    """
    Weingarter and Draine 2001 dust grain extinction model
    for MW, SMC and LMC or any available in WD01.

    NOTE: this model does not inherit from AttenuationLaw because it is
          distinctly different.

    Attributes:
        model (str)
            The dust grain model used.
        emodel (function)
            The function that describes the model from WD01 imported above.
    """

    def __init__(self, model="SMCBar"):
        """
        Initialise the dust curve

        Args:
            model (str)
                The dust grain model to use.

        """

        self.description = (
            "Weingarter and Draine 2001 dust grain extinction"
            " model for MW, SMC and LMC"
        )

        # Get the correct model string
        if "MW" in model:
            self.model = "MWRV31"
        elif "LMC" in model:
            self.model = "LMCAvg"
        elif "SMC" in model:
            self.model = "SMCBar"
        else:
            self.model = model
        self.emodel = WD01(self.model)

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        if isinstance(lam, (unyt_quantity, unyt_array)):
            _lam = lam.to('Angstrom').v
        else:
            _lam = lam
        return self.emodel((_lam*Angstrom).to_astropy())

    def get_transmission(self, tau_v, lam):
        """
        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/array-like, float)
                Optical depth in the V-band. Can either be a single float or
                array.

            lam (array-like, float)
                The wavelengths (with units) at which to calculate transmission.

        Returns:
            array-like
                The transmission at each wavelength. Either (lam.size,) in shape
                for singular tau_v values or (tau_v.size, lam.size) tau_v
                is an array.
        """
        if isinstance(lam, (unyt_quantity, unyt_array)):
            _lam = lam.to('Angstrom').v
        else:
            _lam = lam
        return self.emodel.extinguish(x=(_lam*Angstrom).to_astropy(), Av=1.086 * tau_v)



def Li_08(lam, c1, c2, c3, c4):
        """
        Drude-like parametric expression for the attenuation curve from Li+08
        
        Args:
            
            lam (array-like, float)
                The wavelengths (with unit == micron) at which to calculate transmission.
        
            c1, c2, c3, c4 (float)
                Dimensionless parameters describing the FUV rise, optical/NIR extinction and UV bump strength.
                Values for the Calzetti, SMC, MW and LMC attenuation curve (Li+08):
                    Calzetti SMC    MW      LMC
                c1  44.9     38.7   14.4    4.47
                c2  7.56     3.83   6.52    2.39
                c3  61.2     6.34   2.04    –0.988
                c4  0        0      0.0519  0.0221
        
        
        Returns:
            array-like
                tau/tau_v at each input wavelength (lam)
        
        """
        
        # Wavelength in microns
        if isinstance(lam, (unyt_quantity, unyt_array)):
            lam_micron = lam.to('um').v
        else:
            lam_micron = lam/1e4
            
        
        
        #Attenuation curve normalized to Av
        A_lam_v = c1 / ((lam_micron/0.08)**c2 + (lam_micron/0.08)**-c2 + c3)  +   (233. * (1 - c1/(6.88**c2 + 0.145**c2 +c3) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.)  +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)
        
        
        return A_lam_v
        

class Markov23(AttenuationLaw):
    """
    Parametric attenuation curve;
    implemented in Li+08, Evolution up to high-z (z=12) studied in: Markov+23a,b

    Attributes:
            c1, c2, c3, c4 (float)
                Dimensionless parameters describing the FUV rise, optical/NIR regime and 2175 Angstrom bump of the attenuation curve.

    """

    def __init__(self, c1=1., c2=1., c3=1., c4=0.):
        """
        Initialise the dust curve.
        """
        description = (
            "Parametric attenuation curve; with option"
            "for multiple slopes and varying UV-bump strength"
            "implemented in Li+08, see Markov+23 for high-z application to JWST data z=2-12"
        )
        AttenuationLaw.__init__(self, description)

        # Define the parameters of the model.
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth. (Uses the Li_08 function
        defined above.)

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        return Li_08(
                lam=lam,
                c1=self.c1,
                c2=self.c2,
                c3=self.c3,
                c4=self.c4,
            )
