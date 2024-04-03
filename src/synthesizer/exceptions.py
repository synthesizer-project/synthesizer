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

from typing import Optional


class MissingArgument(Exception):
    """Generic exception class for when an argument is missing."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the MissingArgument class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Missing argument"


class IncorrectUnits(Exception):
    """Generic exception class for when incorrect units are provided."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the IncorrectUnits class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Incorrect units"


class MissingUnits(Exception):
    """Generic exception class for when expected units aren't provided."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the MissingUnits class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Units are missing"


class InconsistentParameter(Exception):
    """Generic exception class for inconsistent parameters."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the InconsistentParameter class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return (
            self.message if self.message else "Inconsistent parameter choice"
        )


class InconsistentArguments(Exception):
    """Generic exception class for inconsistent combinations of arguments."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the InconsistentArguments class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Inconsistent arguments"


class UnimplementedFunctionality(Exception):
    """Generic exception class for functionality not yet implemented."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the UnimplementedFunctionality class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Unimplemented functionality"


class UnknownImageType(Exception):
    """Generic exception class for unknown image types."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the UnknownImageType class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Unknown image type"


class InconsistentAddition(Exception):
    """Generic exception class for when adding two objects is impossible."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the InconsistentAddition class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Inconsistent addition"


class InconsistentCoordinates(Exception):
    """Generic exception class for when coordinates are inconsistent."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the InconsistentCoordinates class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Inconsistent coordinates"


class SVOFilterNotFound(Exception):
    """Exception class for when an SVO filter code is not in the database."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the SVOFilterNotFound class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "SVO filter not found"


class InconsistentWavelengths(Exception):
    """Exception class for when array dimensions don't match."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the InconsistentWavelengths class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Inconsistent wavelengths"


class MissingSpectraType(Exception):
    """Exception class for when an SPS grid is missing a spectra type."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the MissingSpectraType class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Missing spectra type"


class MissingImage(Exception):
    """Exception class for when an image has not yet been made."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the MissingImage class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Missing image"


class WavelengthOutOfRange(Exception):
    """Exception class for when a wavelength is outside a Filter's range."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the WavelengthOutOfRange class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Wavelength out of range"


class SVOInaccessible(Exception):
    """Generic exception class for when SVO is inaccessible."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the SVOInaccessible class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "SVO database is down"


class UnrecognisedOption(Exception):
    """Generic exception class for when a string argument is not recognised."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the UnrecognisedOption class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Unrecognised option"


class MissingAttribute(Exception):
    """Generic exception class for when a required attribute is missing."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the MissingAttribute class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return self.message if self.message else "Missing attribute"


class GridError(Exception):
    """Generic exception class for issues related to grids."""

    def __init__(self, message: Optional[str] = None) -> None:
        """
        Initialise the GridError class.

        Args:
            message: The message to display.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the message to display."""
        return (
            self.message if self.message else "There's an issue with the grid"
        )
