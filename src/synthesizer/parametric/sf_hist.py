"""A submodule for creating and manipulating star formation histories.

NOTE: This module is imported as SFH in parametric.__init__ enabling the syntax
      shown below.

Example usage:

    from synthesizer.parametric import SFH

    print(SFH.parametrisations)

    sfh = SFH.Constant(...)
    sfh = SFH.Exponential(...)
    sfh = SFH.LogNormal(...)

    sfh.calculate_sfh()

"""

import io
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from unyt import yr

from synthesizer import exceptions
from synthesizer.synth_warnings import warn
from synthesizer.utils.stats import weighted_mean, weighted_median

# Define a list of the available parametrisations
parametrisations = (
    "Constant",
    "Gaussian",
    "Exponential",
    "TruncatedExponential",
    "DecliningExponential",
    "DelayedExponentialLogNormal",
    "DoublePowerLaw",
    "DenseBasis",
)


class Common:
    """The parent class for all SFH parametrisations.

    Attributes:
        name (str):
            The name of this SFH. This is set by the child and encodes
            the type of the SFH. Possible values are defined in
            parametrisations above.
        parameters (dict):
            A dictionary containing the parameters of the model.
    """

    def __init__(self, name, **kwargs):
        """Initialise the parent.

        Args:
            name (str):
                The name of this SFH. This is set by the child and encodes
                the type of the SFH. Possible values are defined in
                parametrisations above.
            **kwargs (dict):
                A dictionary containing the parameters of the model.
        """
        # Set the name string
        self.name = name

        # Store the model parameters (defined as kwargs)
        self.parameters = kwargs

    def _sfr(self, age):
        """Prototype for child defined SFR functions."""
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            "How did you get here!?"
        )

    def get_sfr(self, age):
        """Calculate the star formation in each bin.

        Args:
            age (float):
                The age at which to calculate the SFR.

        Returns:
            sfr (float/np.ndarray of float)
                The SFR at the passed age. Either as a single value
                or an array for each age in age.
        """
        # If we have been handed an array we need to loop
        if isinstance(age, (np.ndarray, list)):
            return np.array([self._sfr(a) for a in age])

        return self._sfr(age)

    def calculate_sfh(self, t_range=(0, 10**10), dt=10**6):
        """Calculate the star formation history over a specified time range.

        Args:
            t_range (tuple, float):
                The age limits over which to calculate the SFH.
            dt (float):
                The interval between age bins.

        Returns:
            t (np.ndarray of float):
                The age bins.
            sfh (np.ndarray of float):
                The SFH in units of 1 / yr.
        """
        # Define the age array
        t = np.arange(*t_range, dt)

        # Evaluate the array
        sfh = self.get_sfr(t)

        return t, sfh

    def plot_sfh(self, t_range=(0, 10**10), dt=10**6, log=False):
        """Create a quick plot of the star formation history.

        Args:
            t_range (tuple, float):
                The age limits over which to calculate the SFH.
            dt (float):
                The interval between age bins.
            log (bool):
                If True, plot the SFH in log-log space.
        """
        t, sfr = self.calculate_sfh(t_range=t_range, dt=dt)
        plt.plot(t, sfr)

        if log:
            plt.xscale("log")
            plt.yscale("log")

        plt.show()

    def calculate_median_age(self, t_range=(0, 10**10), dt=10**6):
        """Calculate the median age of a given star formation history.

        Args:
            t_range (tuple, float):
                The age limits over which to calculate the SFH.
            dt (float):
                The interval between age bins.

        Returns:
            unyt_array:
                The median age of the star formation history.
        """
        # Get the SFH first
        t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

        return weighted_median(t, sfh) * yr

    def calculate_mean_age(self, t_range=(0, 10**10), dt=10**6):
        """Calculate the median age of a given star formation history.

        Args:
            t_range (tuple, float):
                The age limits over which to calculate the SFH.
            dt (float):
                The interval between age bins.

        Returns:
            unyt_array:
                The mean age of the star formation history.
        """
        # Get the SFH first
        t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

        return weighted_mean(t, sfh) * yr

    def calculate_moment(self, n):
        """Calculate the n-th moment of the star formation history."""
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def plot(self, show=True, save=False, **kwargs):
        """Plot the star formation history, returned by `calculate_sfh`.

        Args:
            show (bool):
                display plot to screen directly
            save (bool, string):
                if False, don't save. If string, save to directory
            kwargs (dict):
                key word arguments describing sfh
        """
        t, sfh = self.calculate_sfh(**kwargs)

        plt.plot(t, sfh)
        plt.xlabel("age (yr)")
        plt.ylabel(r"SFR ($\mathrm{M_{\odot} \,/\, yr}$)")
        plt.xlim(
            0,
        )
        plt.ylim(
            0,
        )

        if show:
            plt.show()

        if save:
            plt.savefig(save, bbox_inches="tight")
            plt.close()

    def __str__(self):
        """Print basic summary of the parameterised star formation history."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF PARAMETERISED STAR FORMATION HISTORY" + "\n"
        pstr += str(self.__class__) + "\n"
        for parameter_name, parameter_value in self.parameters.items():
            pstr += f"{parameter_name}: {parameter_value}" + "\n"
        pstr += (
            f"median age: {self.calculate_median_age().to('Myr'):.2f}" + "\n"
        )
        pstr += f"mean age: {self.calculate_mean_age().to('Myr'):.2f}" + "\n"
        pstr += "-" * 10 + "\n"

        return pstr


class Constant(Common):
    """A constant star formation history.

    The SFR is defined such that:
        sfr = 1; min_age<t<=max_age
        sfr = 0; t>max_age, t<min_age

    Attributes:
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, max_age=100 * yr, min_age=0 * yr, duration=None):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
                If min_age = 0 then this is the duration of star formation.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
            duration (unyt_quantity):
                The duration of the star formation history. This is deprecated
                in favour of max_age.

        """
        if duration is not None:
            warn(
                "The use of duration is deprecated in favour of max_age",
                DeprecationWarning,
                stacklevel=2,
            )
            max_age = duration

        # Initialise the parent
        Common.__init__(
            self, name="Constant", min_age=min_age, max_age=max_age
        )

        # Set the model parameters
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        # Set the SFR based on the duration.
        if (age <= self.max_age) & (age >= self.min_age):
            return 1.0
        return 0.0


class Gaussian(Common):
    """A Gaussian star formation history.

    Attributes:
        peak_age (unyt_quantity):
            The age at which the star formation peaks, i.e. the age at which
            the gaussian is centred.
        sigma (unyt_quantity):
            The standard deviation of the gaussian function.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, peak_age, sigma, max_age=1e11 * yr, min_age=0 * yr):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            peak_age (unyt_quantity):
                The age at which the star formation peaks, i.e. the age at
                which the gaussian is centred.
            sigma (unyt_quantity):
                The standard deviation of the gaussian function.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        # Initialise the parent
        Common.__init__(
            self,
            name="Gaussian",
            peak_age=peak_age,
            sigma=sigma,
            min_age=min_age,
            max_age=max_age,
        )

        # Set the model parameters
        self.peak_age = peak_age.to("yr").value
        self.sigma = sigma.to("yr").value
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        # Set the SFR based on the duration.
        if (age <= self.max_age) & (age >= self.min_age):
            return np.exp(-np.power((age - self.peak_age) / self.sigma, 2.0))
        return 0.0


class Exponential(Common):
    """A exponential star formation history that can be truncated.

    Note: synthesizer uses age (not t) as its parameter such that:
        t = max_age - age

    Attributes:
        tau (unyt_quantity):
            The "stretch" parameter of the exponential.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, max_age, min_age=0.0 * yr):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (unyt_quantity):
                The "stretch" parameter of the exponential. Here a positive
                tau produces a declining exponential, i.e. the star formation
                rate decreases with time, but decreases with age. A negative
                tau indicates an increasing star formation history.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        # Raise exception if tau is zero
        if tau == 0.0:
            raise exceptions.InconsistentArguments("tau must be non-zero!")

        # Initialise the parent
        Common.__init__(
            self,
            name="Exponential",
            tau=tau,
            max_age=max_age,
            min_age=min_age,
        )

        # Set the model parameters
        self.tau = tau.to("yr").value
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        # define time coordinate
        t = self.max_age - age

        if (age < self.max_age) and (age >= self.min_age):
            return np.exp(-(t - self.max_age) / self.tau)
            # equivalent to:
            # return np.exp(age / self.tau)

        return 0.0


class TruncatedExponential(Exponential):
    """A truncated exponential star formation history.

    Attributes:
        tau (unyt_quantity):
            The "stretch" parameter of the exponential.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, max_age, min_age):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (unyt_quantity):
                The "stretch" parameter of the exponential. Here a positive
                tau produces a declining exponential, i.e. the star formation
                rate decreases with time, but decreases with age. A negative
                tau indicates an increasing star formation history.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        Exponential.__init__(self, tau, max_age, min_age=min_age)


class DecliningExponential(Exponential):
    """A declining exponential star formation history.

    Attributes:
        tau (unyt_quantity):
            The "stretch" parameter of the exponential.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.

    """

    def __init__(self, tau, max_age, min_age=0.0 * yr):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (unyt_quantity):
                The "stretch" parameter of the exponential. Here a positive
                tau produces a declining exponential, i.e. the star formation
                rate decreases with time, but decreases with age. A negative
                tau indicates an increasing star formation history.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        if tau <= 0.0:
            raise exceptions.InconsistentArguments(
                "For a declining exponential tau must be positive!"
            )
        Exponential.__init__(self, tau, max_age, min_age=min_age)


class DelayedExponential(Common):
    """A delayed exponential star formation history that can be truncated.

    The "delayed" element results in a more gradual increase at earlier
    times.

    Attributes:
        tau (unyt_quantity):
            The "stretch" parameter of the exponential.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, max_age, min_age=0.0 * yr):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (unyt_quantity):
                The "stretch" parameter of the exponential. Here a positive
                tau produces a declining exponential, i.e. the star formation
                rate decreases with time, but decreases with age. A negative
                tau indicates an increasing star formation history.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        # Raise exception if tau is zero
        if tau == 0.0:
            raise exceptions.InconsistentArguments("tau must be non-zero!")

        # Initialise the parent
        Common.__init__(
            self,
            name="DelayedExponential",
            tau=tau,
            max_age=max_age,
            min_age=min_age,
        )

        # Set the model parameters
        self.tau = tau.to("yr").value
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        # define time coordinate
        t = self.max_age - age

        if (age < self.max_age) and (age >= self.min_age):
            return t * np.exp(-(t - self.max_age) / self.tau)
        return 0.0


class LogNormal(Common):
    """A log-normal star formation history.

    Attributes:
        tau (float):
            The dimensionless "width" of the log normal distribution.
        peak_age (float):
            The peak of the log normal distribution.
        max_age (unyt_quantity):
            The maximum age of the log normal distribution. In addition to
            truncating the star formation history this is used to define
            the peak.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, peak_age, max_age, min_age=0 * yr):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (float):
               The dimensionless "width" of the log normal distribution.
            peak_age (unyt_quantity):
                The peak of the log normal distribution.
            max_age (unyt_quantity):
                The maximum age of the log normal distribution. In addition to
                truncating the star formation history this is used to define
                the peak.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        # Initialise the parent
        Common.__init__(
            self,
            name="LogNormal",
            tau=tau,
            peak_age=peak_age,
            max_age=max_age,
            min_age=min_age,
        )

        # Set the model parameters
        self.peak_age = peak_age.to("yr").value
        self.tau = tau
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

        # Calculate the relative ages and peak for the calculation
        self.tpeak = self.max_age - self.peak_age
        self.t_0 = np.log(self.tpeak) + self.tau**2

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        if (age < self.max_age) & (age >= self.min_age):
            norm = 1.0 / (self.max_age - age)
            exponent = (
                (np.log(self.max_age - age) - self.t_0) ** 2 / 2 / self.tau**2
            )
            return norm * np.exp(-exponent)

        return 0.0


class DoublePowerLaw(Common):
    """A double power law star formation history.

    Attributes:
        peak_age (unyt_quantity):
            The age at which the star formation history peaks.
        alpha (float):
            The first power.
        beta (float):
            The second power.
        max_age (unyt_quantity):
            The age above which the star formation history is truncated.
        min_age (unyt_quantity):
            The age below which the star formation history is truncated.
    """

    def __init__(
        self, peak_age, alpha, beta, min_age=0 * yr, max_age=1e11 * yr
    ):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            peak_age (unyt_quantity):
                The age at which the star formation history peaks.
            alpha (float):
                The first power.
            beta (float):
                The second power.
            max_age (unyt_quantity):
                The age above which the star formation history is truncated.
            min_age (unyt_quantity):
                The age below which the star formation history is truncated.
        """
        # Initialise the parent
        Common.__init__(
            self,
            name="DoublePowerLaw",
            peak_age=peak_age,
            alpha=alpha,
            beta=beta,
            max_age=max_age,
            min_age=min_age,
        )

        # Set the model parameters
        self.peak_age = peak_age.to("yr").value
        self.alpha = alpha
        self.beta = beta
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """Get the amount SFR weight in a single age bin.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.
        """
        if (age < self.max_age) & (age >= self.min_age):
            term1 = (age / self.peak_age) ** self.alpha
            term2 = (age / self.peak_age) ** self.beta
            return (term1 + term2) ** -1

        return 0.0


class DenseBasis(Common):
    """Dense Basis representation of a SFH.

    See here for more details on the Dense Basis method.

    https://dense-basis.readthedocs.io/en/latest/

    Attributes:
        db_tuple (tuple):
            Dense basis parameters describing the SFH
            1) total mass formed
            2) SFR at the time of observation (see redshift)
            3) number of tx parameters
            4) times when the galaxy formed fractions of its mass,
                units of fraction of the age of the universe [0-1]
        redshift (float):
            redshift at which to scale the SFH
    """

    def __init__(self, db_tuple, redshift):
        """Initialise the parent and this parametrisation of the SFH.

        Args:
            db_tuple (tuple):
                Dense basis parameters describing the SFH
                1) total mass formed
                2) SFR at the time of observation (see redshift)
                3) number of tx parameters
                4) times when the galaxy formed fractions of its mass,
                    units of fraction of the age of the universe [0-1]
            redshift (float):
                redshift at which to scale the SFH
        """
        # Initialise the parent
        Common.__init__(
            self, name="DenseBasis", db_tuple=db_tuple, redshift=redshift
        )

        self.db_tuple = db_tuple
        self.redshift = redshift

        # convert dense basis parameters (1db_tuple`) into SFH
        self._convert_db_to_sfh()

    def _convert_db_to_sfh(
        self, interpolator="gp_george", min_age=5, max_age=10.3
    ):
        """Convert dense basis representation to a binned SFH.

        Args:
            interpolator (str):
                Dense basis interpolator to use. Options:
                [gp_george, gp_sklearn, linear, and pchip].
                Note that gp_sklearn requires sklearn to be installed.
            min_age (float):
                minimum age of SFH grid
            max_age (float):
                maximum age of SFH grid
        """
        # Attempt to import dense_basis, if missing they need to
        # install the optional dependency
        try:
            # This import is in quarantine because it insists on printing
            # "Starting dense_basis" to the console on import! It can stay here
            # until a PR is raised to fix this.
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            import dense_basis as db

            sys.stdout = original_stdout
        except ImportError:
            raise exceptions.UnmetDependency(
                "dense_basis not found. Please install Synthesizer with "
                " dense_basis support by running `pip install "
                ".[dense_basis]`"
            )

        # Convert the dense basis tuple arguments to sfh in mass fraction units
        tempsfh, temptime = db.tuple_to_sfh(
            self.db_tuple, self.redshift, interpolator=interpolator, vb=False
        )

        # Define a new finer grid, and time differences between steps
        self.finegrid = np.linspace(min_age, max_age, 1000)
        tbw = np.mean(np.diff(self.finegrid))

        # define these intervals in log space
        finewidths = 10 ** (self.finegrid + tbw / 2) - 10 ** (
            self.finegrid - tbw / 2
        )

        # Interpolate the SFH on to finer grid in units of SFR
        self.intsfh = self._interp_sfh(
            tempsfh, temptime, 10**self.finegrid / 1e9
        ) / (finewidths / 1e9)

    def _interp_sfh(self, sfh, tax, newtax):
        """Interpolate a dense basis SFH.

        Args:
            sfh (np.ndarray):
                star formation history array
            tax (np.ndarray):
                time axis
            newtax (np.ndarray):
                new time axis

        Returns:
            sfh_interp (np.ndarray):
                array of interpolated sfh values
        """
        sfh_cdf = cumtrapz(sfh, x=tax, initial=0)
        cdf_interp = np.interp(newtax, tax, np.flip(sfh_cdf, 0))
        sfh_interp = np.zeros_like(cdf_interp)
        sfh_interp[0:-1] = -np.diff(cdf_interp)
        return sfh_interp

    def _sfr(self, age):
        """Return SFR at given `age`.

        Args:
            age (float):
                Age to query SFH, units of years

        Returns:
            sfr (float):
                Star formation rate at `age`
        """
        sfr = np.interp(age, 10**self.finegrid, self.intsfh)
        return sfr
