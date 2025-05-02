"""The definitions for Synthesizer specific errors."""


class MissingArgument(Exception):
    """Generic exception class for when an argument is missing."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Missing argument"


class IncorrectUnits(Exception):
    """Generic exception class for when incorrect units are provided."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent units"


class MissingUnits(Exception):
    """Generic exception class for when expected units aren't provided."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Units are missing"


class InconsistentParameter(Exception):
    """Generic exception class for inconsistent parameters."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent parameter choice"


class InconsistentArguments(Exception):
    """Generic exception class for inconsistent combinations of arguments."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Inconsistent parameter choice"


class UnimplementedFunctionality(Exception):
    """Generic exception class for functionality not yet implemented."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Unimplemented functionality!"


class UnknownImageType(Exception):
    """Generic exception class for functionality not yet implemented."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Unknown image type!"


class InconsistentAddition(Exception):
    """Generic exception class for when adding two objects is impossible."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Unable to add"


class InconsistentMultiplication(Exception):
    """Generic exception class for impossible multiplications."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Unable to multiply"


class InconsistentCoordinates(Exception):
    """Generic exception class for when coordinates are inconsistent."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Coordinates are inconsistent"


class SVOFilterNotFound(Exception):
    """Exception class for when an SVO filter code isn't in the database."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Filter not found!"


class SVOTransmissionHasUnits(Exception):
    """Exception class for when an SVO filter returns units.

    This should not be the case for a transmission curve. For example, where
    the effective area is provided instead of a transmission curve, which is
    a case we currently don't support.
    """

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Filter provides units, but transmission is dimensionless!"


class InconsistentWavelengths(Exception):
    """Exception class for when wavelengths are inconsistent."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Coordinates are inconsistent"


class MissingSpectraType(Exception):
    """Exception class for when a spectra type is missing."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Spectra type not in grid!"


class MissingLines(Exception):
    """Exception class for when a line is missing from a spectrum."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Line not in grid!"


class MissingImage(Exception):
    """Exception class for when an image has not yet been made."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "Image not yet created!"


class MissingModelSettings(Exception):
    """Exception class for when a model is missing settings."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        return "Model is missing settings!"


class MissingIFU(Exception):
    """Exception class for when an IFU has not yet been made."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "IFU not yet created!"


class WavelengthOutOfRange(Exception):
    """Exception class for when a wavelength is out of range."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        else:
            return "The provided wavelength is out of the filter range!"


class SVOInaccessible(Exception):
    """Generic exception class for when SVO is inaccessible."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "SVO database is down!"


class UnrecognisedOption(Exception):
    """Exception class string arguments not being a recognised option."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Unrecognised option."


class MissingAttribute(Exception):
    """Exception class for when a required attribute is missing."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Missing attribute"


class GridError(Exception):
    """Exception class Grid related issues."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Theres an issues with the grid."


class UnmetDependency(Exception):
    """Exception for missing dependencies.

    A generic exception class for anything to do with not having specific
    packages not mentioned in the requirements. This is usually when there
    are added dependency due to including extraneous capabilities.
    """

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "There are unmet package dependencies."


class DownloadError(Exception):
    """Exception for failed downloads.

    A generic exception class for anything to do with not having specific
    packages not mentioned in the requirements. This is usually when there
    are added dependency due to including extraneous capabilities.
    """

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "There was an error downloading the data."


class PipelineNotReady(Exception):
    """Exception class for when a required pipeline step hasn't been run."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Pipeline isn't ready to run current operation."


class BadResult(Exception):
    """Exception class for when a result is not as expected."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Result is not as expected."


class MissingMaskAttribute(Exception):
    """Exception class for when the masking attribute is missing."""

    def __init__(self, *args):
        """Initialise the exception with an optional message.

        Args:
            *args: Optional message to include in the exception.
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The string representation of the exception.
        """
        if self.message:
            return "{0} ".format(self.message)
        return "Mask is missing an attribute."
