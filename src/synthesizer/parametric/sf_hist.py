"""A module for creating and manipulating star formation histories.

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

import numpy as np
from unyt import yr

from synthesizer import exceptions
from synthesizer.utils.stats import weighted_mean, weighted_median

# Define a list of the available parametrisations
parametrisations = (
    "Constant",
    "Exponential",
    "LogNormal",
    "DoublePowerLaw",
)


class Common:
    """
    The parent class for all SFH parametrisations.

    Attributes:
        name (string)
            The name of this SFH. This is set by the child and encodes
            the type of the SFH. Possible values are defined in
            parametrisations above.
        parameters (dict)
            A dictionary containing the parameters of the model.
    """

    def __init__(self, name, **kwargs):
        """
        Initialise the parent.

        Args:
            name (string)
                The name of this SFH. This is set by the child and encodes
                the type of the SFH. Possible values are defined in
                parametrisations above.
        """

        # Set the name string
        self.name = name

        # Store the model parameters (defined as kwargs)
        self.parameters = kwargs

    def _sfr(self, age):
        """
        Prototype for child defined SFR functions.
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            "How did you get here!?"
        )

    def get_sfr(self, age):
        """
        Calculate the star formation in each bin.

        Args:
            age (float)
                The age at which to calculate the SFR.

        Returns:
            sfr (float/array-like, float)
                The SFR at the passed age. Either as a single value
                or an array for each age in age.
        """

        # If we have been handed an array we need to loop
        if isinstance(age, (np.ndarray, list)):
            return np.array([self._sfr(a) for a in age])

        return self._sfr(age)

    def calculate_sfh(self, t_range=(0, 10**10), dt=10**6):
        """
        Calcualte the age of a given star formation history.

        Args:
            t_range (tuple, float)
                The age limits over which to calculate the SFH.
            dt (float)
                The interval between age bins.

        Returns:
            t (array-like, float)
                The age bins.
            sfh (array-like, float)
                The SFH in units of 1 / yr.
        """

        # Define the age array
        t = np.arange(*t_range, dt)

        # Evaluate the array
        sfh = self.get_sfr(t)

        return t, sfh

    def calculate_median_age(self, t_range=(0, 10**10), dt=10**6):
        """
        Calcualte the median age of a given star formation history.

        Args:
            t_range (tuple, float)
                The age limits over which to calculate the SFH.
            dt (float)
                The interval between age bins.

        Returns:
            t (array-like, float)
                The age bins.
            sfh (array-like, float)
                The SFH.
        """

        # Get the SFH first
        t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

        return weighted_median(t, sfh) * yr

    def calculate_mean_age(self, t_range=(0, 10**10), dt=10**6):
        """
        Calcualte the median age of a given star formation history.

        Args:
            t_range (tuple, float)
                The age limits over which to calculate the SFH.
            dt (float)
                The interval between age bins.

        Returns:
            t (array-like, float)
                The age bins.
            sfh (array-like, float)
                The SFH.
        """

        # Get the SFH first
        t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

        return weighted_mean(t, sfh) * yr

    def calculate_moment(self, n):
        """
        Calculate the n-th moment of the star formation history.
        """

        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def __str__(self):
        """
        Print basic summary of the parameterised star formation history.
        """

        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF PARAMETERISED STAR FORMATION HISTORY" + "\n"
        pstr += str(self.__class__) + "\n"
        for parameter_name, parameter_value in self.parameters.items():
            pstr += f"{parameter_name}: {parameter_value}" + "\n"
        pstr += (
            f'median age: {self.calculate_median_age().to("Myr"):.2f}' + "\n"
        )
        pstr += f'mean age: {self.calculate_mean_age().to("Myr"):.2f}' + "\n"
        pstr += "-" * 10 + "\n"

        return pstr


class Constant(Common):
    """
    A constant star formation history.

    The SFR is defined such that:
        sfr = 1; min_age<t<=max_age
        sfr = 0; t>max_age, t<min_age

    Attributes:
       max_age (unyt_quantity)
            The age above which the star formation history is truncated.
        min_age (unyt_quantity)
            The age below which the star formation history is truncated.
    """

    def __init__(self, max_age, min_age=0 * yr):
        """
        Initialise the parent and this parametrisation of the SFH.

        Args:
            max_age (unyt_quantity)
                The age above which the star formation history is truncated.
                If min_age = 0 then this is the duration of star formation.
            min_age (unyt_quantity)
                The age below which the star formation history is truncated.
        """

        # Initialise the parent
        Common.__init__(
            self, name="Constant", min_age=min_age, max_age=max_age
        )

        # Set the model parameters
        self.max_age = max_age.to("yr").value
        self.min_age = min_age.to("yr").value

    def _sfr(self, age):
        """
        Get the amount SFR weight in a single age bin.

        Args:
            age (float)
                The age (in years) at which to evaluate the SFR.
        """

        # Set the SFR based on the duration.
        if (age <= self.max_age) & (age > self.min_age):
            return 1.0
        return 0.0


class Exponential(Common):
    """
    A truncated exponential star formation history.

    Attributes:
        tau (unyt_quantity)
            The "stretch" parameter of the exponential.
        max_age (unyt_quantity)
            The age above which the star formation history is truncated.
        min_age (unyt_quantity)
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, max_age=1e11 * yr, min_age=0.0 * yr):
        """
        Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (unyt_quantity)
                The "stretch" parameter of the exponential.
            max_age (unyt_quantity)
                The age above which the star formation history is truncated.
            min_age (unyt_quantity)
                The age below which the star formation history is truncated.
        """

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
        """
        Get the amount SFR weight in a single age bin.

        Args:
            age (float)
                The age (in years) at which to evaluate the SFR.
        """

        if (age < self.max_age) and (age > self.min_age):
            return np.exp(-age / self.tau)
        return 0.0


# included for backwards compatability
TruncatedExponential = Exponential


class LogNormal(Common):
    """
    A log-normal star formation history.

    Attributes:
        tau (float)
            The dimensionless "width" of the log normal distribution.
        peak_age (float)
            The peak of the log normal distribution.
        max_age (unyt_quantity)
            The maximum age of the log normal distribution. In addition to
            truncating the star formation history this is used to define
            the peak.
        min_age (unyt_quantity)
            The age below which the star formation history is truncated.
    """

    def __init__(self, tau, peak_age, max_age, min_age=0 * yr):
        """
        Initialise the parent and this parametrisation of the SFH.

        Args:
            tau (float)
               The dimensionless "width" of the log normal distribution.
            peak_age (unyt_quantity)
                The peak of the log normal distribution.
            max_age (unyt_quantity)
                The maximum age of the log normal distribution. In addition to
                truncating the star formation history this is used to define
                the peak.
            min_age (unyt_quantity)
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
        """
        Get the amount SFR weight in a single age bin.

        Args:
            age (float)
                The age (in years) at which to evaluate the SFR.
        """
        if (age < self.max_age) & (age > self.min_age):
            norm = 1.0 / (self.max_age - age)
            exponent = (
                (np.log(self.max_age - age) - self.t_0) ** 2 / 2 / self.tau**2
            )
            return norm * np.exp(-exponent)

        return 0.0


class DoublePowerLaw(Common):
    """
    A double power law star formation history.

    Attributes:
        peak_age (unyt_quantity)
            The age at which the star formation history peaks.
        alpha (float)
            The first power.
        beta (float)
            The second power.
        max_age (unyt_quantity)
            The age above which the star formation history is truncated.
        min_age (unyt_quantity)
            The age below which the star formation history is truncated.
    """

    def __init__(
        self, peak_age, alpha, beta, min_age=0 * yr, max_age=1e11 * yr
    ):
        """
        Initialise the parent and this parametrisation of the SFH.

        Args:
            peak_age (unyt_quantity)
                The age at which the star formation history peaks.
            alpha (float)
                The first power.
            beta (float)
                The second power.
            max_age (unyt_quantity)
                The age above which the star formation history is truncated.
            min_age (unyt_quantity)
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
        """
        Get the amount SFR weight in a single age bin.

        Args:
            age (float)
                The age (in years) at which to evaluate the SFR.
        """

        if (age < self.max_age) & (age > self.min_age):
            term1 = (age / self.peak_age) ** self.alpha
            term2 = (age / self.peak_age) ** self.beta
            return (term1 + term2) ** -1

        return 0.0
