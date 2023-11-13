""" A module for creating and manipulating metallicity histories.


"""
import numpy as np


class ZH:

    """A collection of classes describing the metallicity history (and distribution)"""

    # Define a list of the available parametrisations
    parametrisations = (
        "deltaConstant",
    )

    class Common:
        def __str__(self):
            """print basic summary of the parameterised star formation history"""

            pstr = ""
            pstr += "-" * 10 + "\n"
            pstr += "SUMMARY OF PARAMETERISED METAL ENRICHMENT HISTORY" + "\n"
            pstr += str(self.__class__) + "\n"
            # for parameter_name, parameter_value in self.parameters.items():
            #     pstr += f"{parameter_name}: {parameter_value}" + "\n"
            pstr += "-" * 10 + "\n"
            return pstr

    class deltaConstant(Common):

        """return a single metallicity as a function of age."""

        def __init__(self, parameters):
            self.name = "Constant"
            self.dist = "delta"  # set distribution type
            self.parameters = parameters
            if "Z" in parameters.keys():
                self.metallicity_ = parameters["Z"]
                self.log10metallicity_ = np.log10(self.metallicity_)
            elif "log10Z" in parameters.keys():
                self.log10metallicity_ = parameters["log10Z"]
                self.metallicity_ = 10**self.log10metallicity_

        def metallicity(self, *args):
            return self.metallicity_

        def log10metallicity(self, *args):
            return self.log10metallicity_
