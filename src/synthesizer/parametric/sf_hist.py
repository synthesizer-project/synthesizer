""" A module for creating and manipulating star formation histories.


"""

class SFH:

    """A collection of classes describing parametric star formation histories"""

    class Common:
        def sfr(self, age):
            if isinstance(age, np.ndarray) | isinstance(age, list):
                return np.array([self.sfr_(a) for a in age])
            else:
                return self.sfr_(age)

        def calculate_sfh(self, t_range=[0, 1e10], dt=1e6):
            """calcualte the age of a given star formation history"""

            t = np.arange(*t_range, dt)
            sfh = self.sfr(t)
            return t, sfh

        def calculate_median_age(self, t_range=[0, 1e10], dt=1e6):
            """calcualte the median age of a given star formation history"""

            t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

            return weighted_median(t, sfh) * yr

        def calculate_mean_age(self, t_range=[0, 1e10], dt=1e6):
            """calcualte the median age of a given star formation history"""

            t, sfh = self.calculate_sfh(t_range=t_range, dt=dt)

            return weighted_mean(t, sfh) * yr

        def calculate_moment(self, n):
            """calculate the n-th moment of the star formation history"""

            print("WARNING: not yet implemnted")
            return

        def __str__(self):
            """print basic summary of the parameterised star formation history"""

            pstr = ""
            pstr += "-" * 10 + "\n"
            pstr += "SUMMARY OF PARAMETERISED STAR FORMATION HISTORY" + "\n"
            pstr += str(self.__class__) + "\n"
            for parameter_name, parameter_value in self.parameters.items():
                pstr += f"{parameter_name}: {parameter_value}" + "\n"
            pstr += f'median age: {self.calculate_median_age().to("Myr"):.2f}' + "\n"
            pstr += f'mean age: {self.calculate_mean_age().to("Myr"):.2f}' + "\n"
            pstr += "-" * 10 + "\n"
            return pstr

    class Constant(Common):

        """
        A constant star formation history
            sfr = 1; t<=duration
            sfr = 0; t>duration
        """

        def __init__(self, parameters):
            self.name = "Constant"
            self.parameters = parameters
            self.duration = self.parameters["duration"].to("yr").value

        def sfr_(self, age):
            if age <= self.duration:
                return 1.0
            else:
                return 0.0

    class Exponential(Common):

        """
        An exponential star formation history
        """

        def __init__(self, parameters):
            self.name = "Exponential"
            self.parameters = parameters
            self.tau = self.parameters["tau"].to("yr").value

        def sfr_(self, age):
            return np.exp(-age / self.tau)

    class TruncatedExponential(Common):

        """
        A truncated exponential star formation history
        """

        def __init__(self, parameters):
            self.name = "Truncated Exponential"
            self.parameters = parameters
            self.tau = self.parameters["tau"].to("yr").value

            if "max_age" in self.parameters:
                self.max_age = self.parameters["max_age"].to("yr").value
            else:
                self.max_age = self.parameters["duration"].to("yr").value

        def sfr_(self, age):
            if age < self.max_age:
                return np.exp(-age / self.tau)
            else:
                return 0.0

    class LogNormal(Common):
        """
        A log-normal star formation history
        """

        def __init__(self, parameters):
            self.name = "Log Normal"
            self.parameters = parameters
            self.peak_age = self.parameters["peak_age"].to("yr").value
            self.tau = self.parameters["tau"]
            self.max_age = self.parameters["max_age"].to("yr").value

            self.tpeak = self.max_age - self.peak_age
            self.T0 = np.log(self.tpeak) + self.tau**2

        def sfr_(self, age):
            """age is lookback time"""

            if age < self.max_age:
                return (1.0 / (self.max_age - age)) * np.exp(
                    -((np.log(self.max_age - age) - self.T0) ** 2) / (2 * self.tau**2)
                )
            else:
                return 0.0
