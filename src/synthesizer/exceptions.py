"""A module for Synthesizer specific errors.

This module contains the definitions for the exceptions that are raised
by the Synthesizer package.

Exceptions:
    MissingArgument: A generic exception class for when an argument is missing.
    IncorrectUnits: A generic exception class for when incorrect units are
                    provided.
    MissingUnits: A generic exception class for when expected units aren't
                  provided.
    InconsistentParameter: A generic exception class for inconsistent
                           parameters.
    InconsistentArguments: A generic exception class for inconsistent
                           combinations of arguments.
    UnimplementedFunctionality: A generic exception class for functionality not
                                yet implemented.
    UnknownImageType: A generic exception class for functionality not yet
                      implemented.
    InconsistentAddition: A generic exception class for when adding two objects
                          is impossible.
    InconsistentCoordinates: A generic exception class for when coordinates are
                             inconsistent.
    SVOFilterNotFound: An exception class for when an SVO filter code does not
                       match one in the database.
    InconsistentWavelengths: An exception class for when array dimensions don't
                             match.
    MissingSpectraType: An exception class for when an SPS grid is missing.
    MissingImage: An exception class for when an image has not yet been made.
    WavelengthOutOfRange: An exception class for when a wavelength is not
                          accessible to Filters in a FilterCollection.
    SVOInaccessible: A generic exception class for when SVO is inaccessible.
    UnrecognisedOption: A generic exception class for when a string argument is
                        not a recognised option.
    MissingAttribute: A generic exception class for when a required attribute
                      is missing on an object.
    GridError: A generic exception class for anything to with grid issues, such
               as particles not lying within a grid, missing axes etc.
"""


class MissingArgument(Exception):
    """Generic exception class for when an argument is missing."""

    def __init__(self, *args):
        """
        Initialise the MissingArgument class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Missing argument"


class IncorrectUnits(Exception):
    """Generic exception class for when incorrect units are provided."""

    def __init__(self, *args):
        """
        Initialise the IncorrectUnits class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent units"


class MissingUnits(Exception):
    """Generic exception class for when expected units aren't provided."""

    def __init__(self, *args):
        """
        Initialise the MissingUnits class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Units are missing"


class InconsistentParameter(Exception):
    """Generic exception class for inconsistent parameters."""

    def __init__(self, *args):
        """
        Initialise the InconsistentParameter class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent parameter choice"


class InconsistentArguments(Exception):
    """Generic exception class for inconsistent combinations of arguments."""

    def __init__(self, *args):
        """
        Initialise the InconsistentArguments class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent parameter choice"


class UnimplementedFunctionality(Exception):
    """Generic exception class for functionality not yet implemented."""

    def __init__(self, *args):
        """
        Initialise the UnimplementedFunctionality class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Unimplemented functionality!"


class UnknownImageType(Exception):
    """Generic exception class for functionality not yet implemented."""

    def __init__(self, *args):
        """
        Initialise the UnknownImageType class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            return "Inconsistent parameter choice"

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent units"


class InconsistentAddition(Exception):
    """Generic exception class for when adding two objects is impossible."""

    def __init__(self, *args):
        """
        Initialise the InconsistentAddition class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Unable to add"


class InconsistentCoordinates(Exception):
    """Generic exception class for when coordinates are inconsistent."""

    def __init__(self, *args):
        """
        Initialise the InconsistentCoordinates class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Coordinates are inconsistent"


class SVOFilterNotFound(Exception):
    """Exception class for when an SVO filter code is not in the database."""

    def __init__(self, *args):
        """
        Initialise the SVOFilterNotFound class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Filter not found!"


class InconsistentWavelengths(Exception):
    """Exception class for when array dimensions don't."""

    def __init__(self, *args):
        """
        Initialise the InconsistentWavelengths class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Coordinates are inconsistent"


class MissingSpectraType(Exception):
    """Exception class for when an SPS grid is missing."""

    def __init__(self, *args):
        """
        Initialise the MissingSpectraType class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Spectra type not in grid!"


class MissingImage(Exception):
    """Exception class for when an image has not yet been made."""

    def __init__(self, *args):
        """
        Initialise the MissingImage class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Image not yet created!"


class WavelengthOutOfRange(Exception):
    """Exception class for when a wavelength is outside a Filter."""

    def __init__(self, *args):
        """
        Initialise the WavelengthOutOfRange class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "The provided wavelength is out of the filter range!"


class SVOInaccessible(Exception):
    """Generic exception class for when SVO is inaccessible."""

    def __init__(self, *args):
        """
        Initialise the SVOInaccessible class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "SVO database is down!"


class UnrecognisedOption(Exception):
    """Generic exception class for when an argument is not recognised."""

    def __init__(self, *args):
        """
        Initialise the UnrecognisedOption class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Unrecognised option."


class MissingAttribute(Exception):
    """Generic exception class for when a required attribute is missing."""

    def __init__(self, *args):
        """
        Initialise the MissingAttribute class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Missing attribute"


class GridError(Exception):
    """
    Generic exception class for anything to with grid issues.

    e.g. particles not lying within a grid, missing axes etc.
    """

    def __init__(self, *args):
        """
        Initialise the GridError class.

        Args:
            *args: A list of arguments. The first argument is the message to
                   display.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the message to display."""
        if self.message:
            return "{0} ".format(self.message)
        return "Theres an issues with the grid."
