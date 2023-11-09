""" A module for creating and manipulating metallicity histories.


"""
import numpy as np


class ZH:

    """A collection of classes describing the metallicity history (and distribution)"""

    class Common:
        def __str__(self):
            """print basic summary of the parameterised star formation history"""

            pstr = ""
            pstr += "-" * 10 + "\n"
            pstr += "SUMMARY OF PARAMETERISED METAL ENRICHMENT HISTORY" + "\n"
            pstr += str(self.__class__) + "\n"
            for parameter_name, parameter_value in self.parameters.items():
                pstr += f"{parameter_name}: {parameter_value}" + "\n"
            pstr += "-" * 10 + "\n"
            return pstr

    class deltaConstant(Common):

        """return a single metallicity as a function of age."""

        def __init__(self, parameters):
            self.name = "Constant"
            self.dist = "delta"  # set distribution type
            self.parameters = parameters
            if "Z" in parameters.keys():
                self.Z_ = parameters["Z"]
                self.log10Z_ = np.log10(self.Z_)
            elif "log10Z" in parameters.keys():
                self.log10Z_ = parameters["log10Z"]
                self.Z_ = 10**self.log10Z_

        def Z(self, age):
            return self.Z_

        def log10Z(self, age):
            return self.log10Z_
