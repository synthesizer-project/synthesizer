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
from unyt import Gyr, unyt_array, yr

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
    "DelayedExponential",
    "LogNormal",
    "DoublePowerLaw",
    "DenseBasis",
    "Continuity",
    "ContinuityFlex",
    "Dirichlet",
    "ContinuityPSB",
    "CombinedSFH",
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
            if hasattr(self, "_sfrs"):
                # If the child has defined a vectorised version of _sfr
                return self._sfrs(age)
            else:
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

    def __add__(self, other):
        """Combine two SFH models by summing their SFRs."""
        if not isinstance(other, Common):
            raise TypeError(
                f"Cannot add {type(other)} to {type(self)}. "
                "Both must be instances of Common or its subclasses."
            )

        # Create a new instance of CombinedSFH - if one of the models
        # is a CombinedSFH, we can just add the other model to it

        if isinstance(self, CombinedSFH):
            new_sfh = self.sfh_models + [other]
        elif isinstance(other, CombinedSFH):
            new_sfh = other.sfh_models + [self]
        else:
            new_sfh = CombinedSFH([self, other])

        # Return the new combined SFH
        return new_sfh


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

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        sfrs = np.zeros_like(ages)
        mask = (ages <= self.max_age) & (ages >= self.min_age)
        sfrs[mask] = 1.0
        return sfrs


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

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        # Set the SFR based on the duration.
        sfrs = np.zeros_like(ages)
        mask = (ages <= self.max_age) & (ages >= self.min_age)
        sfrs[mask] = np.exp(
            -np.power((ages[mask] - self.peak_age) / self.sigma, 2.0)
        )
        return sfrs


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
            return np.exp(-t / self.tau)
            # equivalent to:
            # return np.exp(age / self.tau)

        return 0.0

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        # define time coordinate
        t = self.max_age - ages

        # Set the SFR based on the duration.
        sfrs = np.zeros_like(ages)
        mask = (ages < self.max_age) & (ages >= self.min_age)
        sfrs[mask] = np.exp(-t[mask] / self.tau)

        return sfrs


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
            return t * np.exp(-t / self.tau)
        return 0.0

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        # define time coordinate
        t = self.max_age - ages

        # Set the SFR based on the duration.
        sfrs = np.zeros_like(ages)
        mask = (ages < self.max_age) & (ages >= self.min_age)
        sfrs[mask] = t[mask] * np.exp(-t[mask] / self.tau)

        return sfrs


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

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        # Set the SFR based on the duration.
        sfrs = np.zeros_like(ages)
        mask = (ages < self.max_age) & (ages >= self.min_age)
        norm = 1.0 / (self.max_age - ages[mask])
        exponent = (
            (np.log(self.max_age - ages[mask]) - self.t_0) ** 2
            / 2
            / self.tau**2
        )
        sfrs[mask] = norm * np.exp(-exponent)

        return sfrs


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

    def _sfrs(self, ages):
        """Vectorised version of _sfr for multiple ages.

        Args:
            ages (np.ndarray of float):
                The ages (in years) at which to evaluate the SFR.
        """
        # Set the SFR based on the duration.
        sfrs = np.zeros_like(ages)
        mask = (ages < self.max_age) & (ages >= self.min_age)
        term1 = (ages[mask] / self.peak_age) ** self.alpha
        term2 = (ages[mask] / self.peak_age) ** self.beta
        sfrs[mask] = (term1 + term2) ** -1

        return sfrs


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
        self.log_finegrid = np.linspace(min_age, max_age, 1000)
        self.finegrid = 10**self.log_finegrid
        tbw = np.mean(np.diff(self.log_finegrid))

        # define these intervals in log space
        finewidths = 10 ** (self.log_finegrid + tbw / 2) - 10 ** (
            self.finegrid - tbw / 2
        )

        # Interpolate the SFH on to finer grid in units of SFR
        self.intsfh = self._interp_sfh(
            tempsfh, temptime, self.finegrid / 1e9
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
        sfr = np.interp(age, self.finegrid, self.intsfh)
        return sfr

    def _sfrs(self, ages):
        return self._sfr(ages)


class Continuity(Common):
    """Non-parametric SFH model of mass in fixed bins with a smoothness prior.

    See Leja et al. 2019 for details on the model.
    This model uses log SFR ratios between adjacent bins with a Student-t prior
    to ensure smoothness in the star formation history.

    Attributes:
    - logsfr_ratios (np.ndarray):
        Array of log_10(SFR_j / SFR_{j+1}) values for adjacent bins.

    -   agebins (unyt_array, optional):
        Array of shape (N, 2) with age bin edges in time units (e.g., years.)
        If None, uses default 3-bin setup: [[0,8], [8,9], [9,10]] (log10 yr).

    masses (np.ndarray):
        The stellar mass formed in each age bin (computed from parameters).

    min_age (unyt_quantity):
        The minimum age for truncation.

    max_age (unyt_quantity):
        The maximum age for truncation.

    """

    def __init__(
        self,
        logsfr_ratios=None,
        agebins=None,
        min_age=0 * yr,
        max_age=1e11 * yr,
    ):
        """Initialize the Continuity SFH model.

        Args:
            logsfr_ratios (array-like, optional):
                Array of log_10(SFR_j / SFR_{j+1}) values.
                If None, defaults to 0.
            agebins (unyt_array, optional):
                Array of shape (N, 2) with age bin edges in time units
                (e.g.years.) If None, uses default 3-bin setup:
                [[0,8],[8,9],[9,10]] (log10 yr).
            min_age (unyt_quantity):
                The minimum age for SFH truncation.
            max_age (unyt_quantity):
                The maximum age for SFH truncation.

        """
        if agebins is None:
            agebins = np.array([[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]])

        else:
            assert isinstance(agebins, unyt_array), (
                "agebins must be a unyt_array or convertible to one"
            )

            assert agebins.ndim == 2 and agebins.shape[1] == 2, (
                "agebins must be a 2D array with shape (N, 2)"
            )

            agebins = np.log10(agebins.to("yr").value)

        nbins = agebins.shape[0]
        # Set default logsfr_ratios if not provided (nbins-1 ratios needed)
        if logsfr_ratios is None:
            logsfr_ratios = np.zeros(nbins - 1)
        else:
            logsfr_ratios = np.array(logsfr_ratios)

        if len(logsfr_ratios) != nbins - 1:
            raise ValueError(
                f"logsfr_ratios must have {nbins - 1} elements"
                f"for {nbins} age bins"
            )

        # Initialize parent

        Common.__init__(
            self,
            name="Continuity",
            logsfr_ratios=logsfr_ratios,
            agebins=agebins,
            min_age=min_age,
            max_age=max_age,
        )

        # Store parameters

        self.logsfr_ratios = logsfr_ratios
        self.agebins = agebins
        self.min_age = min_age.to("yr").value
        self.max_age = max_age.to("yr").value
        # Calculate masses using transformation function
        self.masses = self._logsfr_ratios_to_masses()

        # Create interpolation arrays for SFR calculation
        self._setup_sfr_interpolation()

    @classmethod
    def init_from_prior(
        cls, agebins, df=2, scale=0.3, limits=(-10, 10), **kwargs
    ):
        """Initialize Continuity SFH from a student t prior distribution."""
        from scipy.stats import t

        # generate uniform random variables between 0 and 1 of len(agebins)- 1
        value = np.random.uniform(size=agebins.shape[0] - 1)

        uniform_min = t.cdf(limits[0], df=df, scale=scale)
        uniform_max = t.cdf(limits[1], df=df, scale=scale)
        value = (uniform_max - uniform_min) * value + uniform_min
        value = t.ppf(value, df=df, scale=scale)
        log_sfr_ratios = value

        return cls(agebins=agebins, logsfr_ratios=log_sfr_ratios, **kwargs)

    def _logsfr_ratios_to_masses(self):
        """Convert log SFR ratios to masses in each bin."""
        nbins = self.agebins.shape[0]
        sratios = 10 ** np.clip(self.logsfr_ratios, -10, 10)
        dt = 10 ** self.agebins[:, 1] - 10 ** self.agebins[:, 0]
        coeffs = np.array(
            [
                (1.0 / np.prod(sratios[:i]))
                * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
                for i in range(nbins)
            ]
        )
        m1 = 1 / coeffs.sum()
        return m1 * coeffs

    def _setup_sfr_interpolation(self):
        """Setup arrays for SFR interpolation."""
        # Calculate SFR in each bin (mass / time_width)
        dt = 10 ** self.agebins[:, 1] - 10 ** self.agebins[:, 0]
        self.sfrs = self.masses / dt
        # Create age arrays for interpolation
        self.bin_centers = 10 ** (
            (self.agebins[:, 0] + self.agebins[:, 1]) / 2
        )
        self.bin_edges = np.concatenate(
            [10 ** self.agebins[:, 0], [10 ** self.agebins[-1, 1]]]
        )

    def _sfr(self, age):
        """Get the SFR at a given age.

        Args:
            age (float):
                The age (in years) at which to evaluate the SFR.

        Returns:
            float: The star formation rate at the given age.
        """
        if (age < self.min_age) or (age > self.max_age):
            return 0.0

        # Find which bin this age falls into
        bin_idx = np.searchsorted(self.bin_edges[1:], age)
        if bin_idx >= len(self.sfrs):
            return 0.0

        return self.sfrs[bin_idx]


class ContinuityFlex(Common):
    """Continuity Flex SFH model. See Leja et al. 2019 for details.

    Parameters:
    -----------------
    logsfr_ratio_young (float):
        Log SFR ratio for the youngest bin (default: 0.0).
    logsfr_ratio_old (float):
        Log SFR ratio for the oldest bin (default: 0.0).
    logsfr_ratios (array-like, optional):
        Array of log SFR ratios for flexible bins. If None, defaults to [0.
        0].
    fixed_young_bin (list, optional):
        Fixed age bin for the youngest bin as [start, end] in log10 years.
        If None, defaults to [0.0, 7.5] (log10 years).
    fixed_old_bin (list, optional):
        Fixed age bin for the oldest bin as [start, end] in log10 years.
        If None, defaults to [9.7, 10.136] (log10 years).
    min_age (unyt_quantity):
        The minimum age for truncation (default: 0 yr).
    max_age (unyt_quantity):
        The maximum age for truncation (default: 1e11 yr).
    -----------------

    Raises:
    ValueError: If fixed_young_bin or fixed_old_bin are not lists
                of length 2.
    TypeError: If logsfr_ratios is not an array-like type or convertible
                to an array.
    AssertionError: If fixed_young_bin or fixed_old_bin are not lists of
                    length 2.
    TypeError: If fixed_young_bin or fixed_old_bin are not unyt_arrays or

    """

    def __init__(
        self,
        logsfr_ratio_young=0.0,
        logsfr_ratio_old=0.0,
        logsfr_ratios=None,
        fixed_young_bin=None,
        fixed_old_bin=None,
        min_age=0 * yr,
        max_age=1e11 * yr,
    ):
        """Initialize the Continuity Flexible SFH model.

        Parameters:
        -----------------
        logsfr_ratio_young (float):
            Log SFR ratio for the youngest bin (default: 0.0).
        logsfr_ratio_old (float):
            Log SFR ratio for the oldest bin (default: 0.0).
        logsfr_ratios (array-like, optional):
            Array of log SFR ratios for flexible bins. If None, defaults to [0.
            0].
        fixed_young_bin (list, optional):
            Fixed age bin for the youngest bin as [start, end] in log10 years.
            If None, defaults to [0.0, 7.5] (log10 years).
        fixed_old_bin (list, optional):
            Fixed age bin for the oldest bin as [start, end] in log10 years.
            If None, defaults to [9.7, 10.136] (log10 years).
        min_age (unyt_quantity):
            The minimum age for truncation (default: 0 yr).
        max_age (unyt_quantity):
            The maximum age for truncation (default: 1e11 yr).
        -----------------

        Raises:
        ValueError: If fixed_young_bin or fixed_old_bin are not lists
                    of length 2.
        TypeError: If logsfr_ratios is not an array-like type or convertible
                    to an array.
        AssertionError: If fixed_young_bin or fixed_old_bin are not lists of
                        length 2.
        TypeError: If fixed_young_bin or fixed_old_bin are not unyt_arrays or

        """
        # Set defaults

        if logsfr_ratios is None:
            logsfr_ratios = np.array([0.0])
        else:
            logsfr_ratios = np.array(logsfr_ratios)

        if fixed_young_bin is None:
            fixed_young_bin = [0.0, 7.5]
        else:
            assert len(fixed_young_bin) == 2, (
                "fixed_young_bin must be a list of [start, end] ages"
            )
            if isinstance(fixed_young_bin, unyt_array):
                fixed_young_bin = fixed_young_bin.to("yr").value
                fixed_young_bin = np.log10(fixed_young_bin)

        if fixed_old_bin is None:
            fixed_old_bin = [9.7, 10.136]

        else:
            assert len(fixed_old_bin) == 2, (
                "fixed_old_bin must be a list of [start, end] ages"
            )
            if isinstance(fixed_old_bin, unyt_array):
                fixed_old_bin = fixed_old_bin.to("yr").value
                fixed_old_bin = np.log10(fixed_old_bin)

        # Initialize parent

        Common.__init__(
            self,
            name="ContinuityFlex",
            logsfr_ratio_young=logsfr_ratio_young,
            logsfr_ratio_old=logsfr_ratio_old,
            logsfr_ratios=logsfr_ratios,
            min_age=min_age,
            max_age=max_age,
        )

        # Store parameters

        self.logsfr_ratio_young = logsfr_ratio_young
        self.logsfr_ratio_old = logsfr_ratio_old
        self.logsfr_ratios = logsfr_ratios
        self.fixed_young_bin = np.array(fixed_young_bin)
        self.fixed_old_bin = np.array(fixed_old_bin)
        self.min_age = min_age.to("yr").value
        self.max_age = max_age.to("yr").value

        # Calculate age bins and masses
        self.agebins = self._logsfr_ratios_to_agebins()
        self.masses = self._logsfr_ratios_to_masses_flex()

        # Setup SFR interpolation
        self._setup_sfr_interpolation()

    def _logsfr_ratios_to_agebins(self):
        """Transform SFR ratios to age bins assuming constant mass per bin."""
        logsfr_ratios = np.clip(self.logsfr_ratios, -10, 10)
        # Fixed bin times
        lower_time = (
            10 ** self.fixed_young_bin[1] - 10 ** self.fixed_young_bin[0]
        )
        upper_time = 10 ** self.fixed_old_bin[1] - 10 ** self.fixed_old_bin[0]

        # Total flexible time available
        tflex = 10 ** self.fixed_old_bin[1] - upper_time - lower_time
        # Calculate flexible bin widths
        n_ratio = len(logsfr_ratios)
        sfr_ratios = 10**logsfr_ratios
        dt1 = tflex / (
            1
            + np.sum([np.prod(sfr_ratios[: (i + 1)]) for i in range(n_ratio)])
        )
        # Build age limits vector
        agelims = [1, lower_time]
        agelims.append(dt1 + lower_time)
        for i in range(n_ratio):
            agelims.append(dt1 * np.prod(sfr_ratios[: (i + 1)]) + agelims[-1])

        agelims.append(10 ** self.fixed_old_bin[1])
        # Convert to log space age bins
        agebins = np.log10([agelims[:-1], agelims[1:]]).T

        return agebins

    def _logsfr_ratios_to_masses_flex(self):
        """Convert SFR ratios to masses for flexible bin model."""
        logsfr_ratio_young = np.clip(self.logsfr_ratio_young, -10, 10)
        logsfr_ratio_old = np.clip(self.logsfr_ratio_old, -10, 10)

        abins = self.agebins
        nbins = abins.shape[0] - 2  # Number of flexible bins

        syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
        dtyoung, dt1 = 10 ** abins[:2, 1] - 10 ** abins[:2, 0]
        dtn, dtold = 10 ** abins[-2:, 1] - 10 ** abins[-2:, 0]

        mbin = 1 / (syoung * dtyoung / dt1 + sold * dtold / dtn + nbins)
        myoung = syoung * mbin * dtyoung / dt1
        mold = sold * mbin * dtold / dtn
        n_masses = np.full(nbins, mbin)
        return np.concatenate([[myoung], n_masses, [mold]])

    def _setup_sfr_interpolation(self):
        """Setup arrays for SFR interpolation."""
        dt = 10 ** self.agebins[:, 1] - 10 ** self.agebins[:, 0]
        self.sfrs = self.masses / dt
        self.bin_centers = 10 ** (
            (self.agebins[:, 0] + self.agebins[:, 1]) / 2
        )
        self.bin_edges = np.concatenate(
            [10 ** self.agebins[:, 0], [10 ** self.agebins[-1, 1]]]
        )

    def _sfr(self, age):
        """Get the SFR at a given age."""
        if (age < self.min_age) or (age > self.max_age):
            return 0.0

        bin_idx = np.searchsorted(self.bin_edges[1:], age)

        if bin_idx >= len(self.sfrs):
            return 0.0

        return self.sfrs[bin_idx]


class Dirichlet(Common):
    """A non-parametric SFH with Dirichlet prior on SFR fractions.

    See Leja et al. 2017, 2019 for details on the model.

    This model uses independent Beta-distributed variables that are transformed

    to SFR fractions following a Dirichlet distribution.

    Attributes:
        z_fraction (array-like, optional):
            Array of latent variables from Beta(1,1) distributions.
            Length should be (nbins-1). If None, defaults to zeros.
        agebins (unyt_array, optional):
            Array of shape (N, 2) with age bin edges in log(years).
            If None, uses default 3-bin setup.
        min_age (unyt_quantity):
            The minimum age for SFH truncation.
        max_age (unyt_quantity):
            The maximum age for SFH truncation.

    """

    def __init__(
        self, z_fraction=None, agebins=None, min_age=0 * yr, max_age=1e11 * yr
    ):
        """Initialize the Dirichlet SFH model.

        Args:
            z_fraction (array-like, optional):
                Array of latent variables from Beta(1,1) distributions.
                Length should be (nbins-1). If None, defaults to zeros.
            agebins (unyt_array, optional):
                Array of shape (N, 2) with age bin edges in log(years).
                If None, uses default 3-bin setup.
            min_age (unyt_quantity):
                The minimum age for SFH truncation.
            max_age (unyt_quantity):
                The maximum age for SFH truncation.
        """
        # Set default age bins
        if agebins is None:
            agebins = np.array([[0.0, 8.0], [8.0, 9.0], [9.0, 10.0]])
        else:
            assert isinstance(agebins, unyt_array), (
                "agebins must be a unyt_array or convertible to one"
            )
            assert agebins.ndim == 2 and agebins.shape[1] == 2, (
                "agebins must be a 2D array with shape (N, 2)"
            )

            agebins = agebins.to("yr").value
            agebins = np.log10(agebins)

        nbins = agebins.shape[0]
        # Set default z_fraction

        if z_fraction is None:
            z_fraction = np.zeros(nbins - 1)
        else:
            z_fraction = np.array(z_fraction)

        if len(z_fraction) != nbins - 1:
            raise ValueError(
                f"z_fraction must have {nbins - 1} elements"
                "for {nbins} age bins"
            )

        # Validate z_fraction values are in [0,1]
        if np.any((z_fraction < 0) | (z_fraction > 1)):
            raise ValueError("z_fraction values must be between 0 and 1")

        # Initialize parent

        Common.__init__(
            self,
            name="Dirichlet",
            z_fraction=z_fraction,
            agebins=agebins,
            min_age=min_age,
            max_age=max_age,
        )

        # Store parameters
        self.z_fraction = z_fraction
        self.agebins = agebins
        self.min_age = min_age.to("yr").value
        self.max_age = max_age.to("yr").value

        # Calculate masses
        self.masses = self._zfrac_to_masses()

        # Setup SFR interpolation
        self._setup_sfr_interpolation()

    @classmethod
    def init_from_prior(cls, agebins, **kwargs):
        """Initialize Dirichlet SFH from a prior distribution."""
        from scipy.stats import beta

        # Generate uniform random variables between 0 and 1
        value = np.random.uniform(size=agebins.shape[0] - 1)

        # Transform to Beta(1,1) distribution
        z_fraction = beta.ppf(value, a=1.0, b=1.0)

        return cls(agebins=agebins, z_fraction=z_fraction, **kwargs)

    def _zfrac_to_masses(self):
        """Transform z_fraction variables to masses via Dirichlet prior."""
        # Convert to SFR fractions following Dirichlet transformation

        sfr_fraction = np.zeros(len(self.z_fraction) + 1)
        sfr_fraction[0] = 1.0 - self.z_fraction[0]

        for i in range(1, len(self.z_fraction)):
            sfr_fraction[i] = np.prod(self.z_fraction[:i]) * (
                1.0 - self.z_fraction[i]
            )

        sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

        # Convert to mass fractions
        time_per_bin = np.diff(10**self.agebins, axis=-1)[:, 0]
        mass_fraction = sfr_fraction * np.array(time_per_bin)
        mass_fraction /= mass_fraction.sum()
        # Check for negative masses
        if (mass_fraction < 0).any():
            idx = mass_fraction < 0
        else:
            idx = np.zeros_like(mass_fraction, dtype=bool)

        if np.allclose(mass_fraction[idx], 0, rtol=1e-8):
            mass_fraction[idx] = 0.0
        else:
            raise ValueError(
                "The input z_fractions are returning negative masses!"
            )

        masses = 1.0 * mass_fraction

        return masses

    def _setup_sfr_interpolation(self):
        """Setup arrays for SFR interpolation."""
        dt = 10 ** self.agebins[:, 1] - 10 ** self.agebins[:, 0]
        self.sfrs = self.masses / dt
        self.bin_centers = 10 ** (
            (self.agebins[:, 0] + self.agebins[:, 1]) / 2
        )
        self.bin_edges = np.concatenate(
            [10 ** self.agebins[:, 0], [10 ** self.agebins[-1, 1]]]
        )

    def _sfr(self, age):
        """Get the SFR at a given age."""
        if (age < self.min_age) or (age > self.max_age):
            return 0.0

        bin_idx = np.searchsorted(self.bin_edges[1:], age)

        if bin_idx >= len(self.sfrs):
            return 0.0

        return self.sfrs[bin_idx]

    def get_sfr_fractions(self):
        """Get the SFR fractions in each bin.

        Returns:
        np.ndarray: The fraction of total SFR in each age bin.
        """
        time_per_bin = np.diff(10**self.agebins, axis=-1)[:, 0]
        sfr_per_bin = self.masses / time_per_bin
        return sfr_per_bin / sfr_per_bin.sum()

    def get_mass_fractions(self):
        """Get the mass fractions in each bin.

        Returns:
        np.ndarray: The fraction of total mass in each age bin.
        """
        return self.masses / self.masses.sum()


class ContinuityPSB(Common):
    """A non-parametric SFH model for post-starburst galaxies.

    This model, based on Suess et al. (2021), uses a combination of fixed-width
    and flexible-boundary time bins to model the star formation history (SFH).
    A smoothness prior, based on a Student's t-distribution, is applied to the
    logarithmic ratios of SFRs in adjacent time bins.

    The time bins are structured from youngest to oldest as follows:
    1. A single "youngest" bin of variable width `tlast`.
    2. A set of `nflex` bins spanning the time from `tlast` to `tflex`.
    3. A set of `nfixed` bins spanning the time from `tflex` to `max_age`.

    Attributes:
        logsfr_ratio_young (float):
            The log10 ratio of SFR in the youngest bin to the subsequent bin.
        logsfr_ratio_old (np.ndarray):
            An array of log10 SFR ratios for the `nfixed` oldest bins.
        logsfr_ratios (np.ndarray):
            An array of log10 SFR ratios for the `nflex-1`
            intermediate flexible bins.
        tlast (unyt.unyt_quantity):
            The temporal width of the youngest age bin.
        tflex (unyt.unyt_quantity):
            The lookback time marking the transition
            from flexible to fixed bins.
        nflex (int):
            The number of flexible bins.
        nfixed (int):
            The number of fixed oldest bins.
        max_age (unyt.unyt_quantity):
            The age of the Universe at the time of observation,
            which truncates the SFH.
        min_age (unyt.unyt_quantity):
            The age below which the SFH is truncated (typically 0).
    """

    def __init__(
        self,
        logsfr_ratio_young: float = 0.0,
        logsfr_ratio_old: np.ndarray = None,
        logsfr_ratios: np.ndarray = None,
        tlast: unyt_array = 0.2 * 1e9 * yr,
        tflex: unyt_array = 2.0 * 1e9 * yr,
        nflex: int = 5,
        nfixed: int = 3,
        min_age: unyt_array = 0 * yr,
        max_age: unyt_array = 13.8 * 1e9 * yr,
    ):
        """Initialise the ContinuityPSB SFH model.

        Args:
            logsfr_ratio_young (float):
                Log10(SFR_young/SFR_flex_1). Defaults to 0.0.
            logsfr_ratio_old (np.ndarray, optional):
                Log10(SFR) ratios for the oldest bins.
                Defaults to zeros of size `nfixed`.
            logsfr_ratios (np.ndarray, optional):
                Log10(SFR) ratios for the flexible bins.
                Defaults to zeros of size `nflex-1`.
            tlast (unyt.unyt_quantity):
                Width of the youngest bin. Defaults to 0.2 Gyr.
            tflex (unyt.unyt_quantity):
                Transition time from flexible to fixed bins. Defaults to 2 Gyr.
            nflex (int):
                Number of flexible bins. Defaults to 5.
            nfixed (int):
                Number of fixed oldest bins. Defaults to 3.
            min_age (unyt.unyt_quantity):
                Minimum age for SFH truncation. Defaults to 0 yr.
            max_age (unyt.unyt_quantity):
                Maximum age for SFH truncation (i.e., age of the Universe).
                Defaults to 13.8 Gyr.
        """
        # Set default ratio arrays if not provided
        if logsfr_ratios is None:
            logsfr_ratios = np.zeros(nflex - 1)
        if logsfr_ratio_old is None:
            logsfr_ratio_old = np.zeros(nfixed)

        # Validate the size of input arrays
        if len(logsfr_ratios) != nflex - 1:
            raise ValueError(
                f"logsfr_ratios must have {nflex - 1} "
                f"elements for nflex={nflex}"
            )
        if len(logsfr_ratio_old) != nfixed:
            raise ValueError(
                f"logsfr_ratio_old must have {nfixed} "
                f"elements for nfixed={nfixed}"
            )

        # Initialise the parent class
        Common.__init__(
            self,
            name="ContinuityPSB",
            logsfr_ratio_young=logsfr_ratio_young,
            logsfr_ratio_old=logsfr_ratio_old,
            logsfr_ratios=logsfr_ratios,
            tlast=tlast,
            tflex=tflex,
            nflex=nflex,
            nfixed=nfixed,
            min_age=min_age,
            max_age=max_age,
        )

        # Store parameters, converting time quantities to years
        self.logsfr_ratio_young = logsfr_ratio_young
        self.logsfr_ratio_old = np.array(logsfr_ratio_old)
        self.logsfr_ratios = np.array(logsfr_ratios)
        self.tlast = tlast.to("yr").value
        self.tflex = tflex.to("yr").value
        self.nflex = nflex
        self.nfixed = nfixed
        self.min_age = min_age.to("yr").value
        self.max_age = max_age.to("yr").value

        # Calculate age bins and masses from parameters
        self.agebins = self._calculate_agebins()
        self.masses = self._calculate_masses()

        # Set up attributes for SFR interpolation
        self._setup_sfr_interpolation()

    @classmethod
    def init_from_prior(
        cls,
        nflex: int = 5,
        nfixed: int = 3,
        tflex: unyt_array = 2.0 * Gyr,
        max_age: unyt_array = 13.8 * Gyr,
        student_t_df: float = 2.0,
        student_t_scale: float = 0.3,
        tlast_limits_gyr: tuple = (0.01, 1.5),
        **kwargs,
    ):
        """Initialize ContinuityPSB SFH by drawing parameters from priors.

        This method creates an example SFH based on the priors
        defined in the model.
        The `logsfr_ratio*` parameters are drawn from a Student's t-dist,
        and `tlast` is drawn from a uniform (top-hat) distribution.

        Args:
            nflex (int):
                Number of flexible bins. Defaults to 5.
            nfixed (int):
                Number of fixed oldest bins. Defaults to 3.
            tflex (unyt.unyt_quantity):
                Transition time from flexible to fixed bins. Defaults to 2 Gyr.
            max_age (unyt.unyt_quantity):
                Maximum age for SFH truncation. Defaults to 13.8 Gyr.
            student_t_df (float):
                Degrees of freedom for the Student's t-distribution prior.
                Defaults to 2.0.
            student_t_scale (float):
                Scale parameter for the Student's t-distribution prior.
                 Defaults to 0.3.
            tlast_limits_gyr (tuple):
                The (min, max) limits for the uniform prior on `tlast`, in Gyr.
                Defaults to (0.01, 1.5).
            **kwargs:
                Additional keyword arguments to pass to the constructor
                (e.g., `min_age`).

        Returns:
            ContinuityPSB: An instance of the class with parameters
            drawn from priors.
        """
        from scipy.stats import t, uniform

        # Draw SFR ratios from a Student's t-distribution
        logsfr_ratio_young = t.rvs(df=student_t_df, scale=student_t_scale)

        n_flex_ratios = nflex - 1
        logsfr_ratios = (
            t.rvs(df=student_t_df, scale=student_t_scale, size=n_flex_ratios)
            if n_flex_ratios > 0
            else np.array([])
        )

        logsfr_ratio_old = t.rvs(
            df=student_t_df, scale=student_t_scale, size=nfixed
        )

        # Draw tlast from a uniform (top-hat) distribution in Gyr
        tlast_min, tlast_max = tlast_limits_gyr
        tlast_gyr = uniform.rvs(loc=tlast_min, scale=tlast_max - tlast_min)
        tlast = tlast_gyr * 1e9 * yr  # Convert to unyt_array in years

        # Instantiate the class with the drawn parameters
        return cls(
            logsfr_ratio_young=logsfr_ratio_young,
            logsfr_ratios=logsfr_ratios,
            logsfr_ratio_old=logsfr_ratio_old,
            tlast=tlast,
            tflex=tflex,
            nflex=nflex,
            nfixed=nfixed,
            max_age=max_age,
            **kwargs,
        )

    def _calculate_agebins(self) -> np.ndarray:
        """Calculate the edges of the time bins in years."""
        if self.tlast >= self.tflex:
            raise ValueError("tlast must be less than tflex")
        if self.tflex >= self.max_age:
            raise ValueError("tflex must be less than max_age")

        # Create age bin edges, which are linearly spaced
        #  in the flex and fixed regions
        age_edges = [0.0, self.tlast]
        age_edges.extend(
            np.linspace(self.tlast, self.tflex, self.nflex + 1)[1:]
        )
        age_edges.extend(
            np.linspace(self.tflex, self.max_age, self.nfixed + 1)[1:]
        )
        age_edges = np.array(age_edges)

        return np.array([age_edges[:-1], age_edges[1:]]).T

    def _calculate_masses(self) -> np.ndarray:
        """Convert SFR ratios into the stellar mass formed in each bin."""
        all_logsfr_ratios = np.concatenate(
            [
                [self.logsfr_ratio_young],
                self.logsfr_ratios,
                self.logsfr_ratio_old,
            ]
        )
        sfr_ratios = 10**all_logsfr_ratios

        dt = np.diff(self.agebins, axis=-1).flatten()
        n_bins = len(dt)

        # Calculate SFR in each bin relative to the SFR of the oldest bin
        # We define SFR_i = C_i * SFR_oldest, where C_i are the coefficients
        sfr_coeffs = np.ones(n_bins)
        for i in range(n_bins - 2, -1, -1):
            sfr_coeffs[i] = sfr_coeffs[i + 1] * sfr_ratios[i]

        # Normalize to the total mass (1)
        sfr_oldest = 1 / np.sum(sfr_coeffs * dt)
        sfrs = sfr_coeffs * sfr_oldest
        masses = sfrs * dt

        return masses

    def _setup_sfr_interpolation(self):
        """Set up arrays for efficient SFR interpolation."""
        dt = np.diff(self.agebins, axis=1).flatten()
        # Prevent division by zero for bins with no width
        dt[dt == 0] = 1e-12

        self.sfrs = self.masses / dt
        self.bin_edges = np.concatenate(
            [self.agebins[:, 0], [self.agebins[-1, 1]]]
        )

    def _sfr(self, age: float) -> float:
        """Get the SFR for a given age in years.

        Args:
            age (float): The age at which to evaluate the SFR, in years.

        Returns:
            float: The star formation rate at the given age.
        """
        if not (self.min_age <= age < self.max_age):
            return 0.0

        # Find the appropriate bin for the given age
        bin_idx = np.searchsorted(self.bin_edges, age, side="right") - 1

        if bin_idx < 0:
            return 0.0

        return self.sfrs[bin_idx]


class CombinedSFH(Common):
    """A combined SFH model that uses multiple SFH models.

    This class allows for the combination of different SFH models
    (e.g., Continuity, Dirichlet) into a single SFH model.
    It provides methods to calculate the SFR and mass formed in each bin.

    Attributes:
        sfh_models (list):
            List of SFH model instances to combine.
        name (str):
            Name of the combined SFH model.
    """

    def __init__(self, sfh_models, weights=None):
        """Initialize the CombinedSFH model.

        Args:
            sfh_models (list):
                List of SFH model instances to combine.
            weights (array-like, optional):
                Weights for each model in the combination.
                If None, equal weights are assigned to each model.
        """
        if not isinstance(sfh_models, list):
            raise TypeError("sfh_models must be a list of SFH model instances")

        for model in sfh_models:
            if not isinstance(model, Common):
                raise TypeError(
                    f"Model {model} is not a valid SFH model instance"
                )

        self.sfh_models = sfh_models
        name = "Combined"

        self.weights = (
            weights
            if weights is not None
            else np.ones(len(sfh_models)) / len(sfh_models)
        )

        Common.__init__(self, name=name)

    def _sfr(self, age):
        """Calculate the SFR at a given age by summing models contributions.

        Args:
            age (float):
                Age in years to calculate the SFR.

        Returns:
            float: The total SFR at the given age.
        """
        total_sfr = 0.0
        for model, weight in zip(self.sfh_models, self.weights):
            total_sfr += model._sfr(age) * weight
        return total_sfr

    def get_sfr(self, age):
        """Get the total SFR at a given age.

        Args:
            age (float):
                Age in years to calculate the SFR.

        Returns:
            float: The total SFR at the given age.
        """
        return np.sum(
            [
                model.get_sfr(age) * weight
                for model, weight in zip(self.sfh_models, self.weights)
            ],
            axis=0,
        )

    def __add__(self, other):
        """Combine two CombinedSFH instances."""
        if isinstance(other, CombinedSFH):
            combined_models = self.sfh_models + other.sfh_models
            combined_weights = np.concatenate([self.weights, other.weights])
            return CombinedSFH(combined_models, weights=combined_weights)
        elif isinstance(other, Common):
            # Add a single model to the combined SFH
            return CombinedSFH(
                self.sfh_models + [other],
                weights=np.concatenate([self.weights, [1.0]]),
            )
        else:
            raise TypeError(f"Cannot add {type(other)} to CombinedSFH")
