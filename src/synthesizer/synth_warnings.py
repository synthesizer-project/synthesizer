"""A module containing warnings and deprecation utilities.

This module contains functions and decorators for issuing warnings to the end
user.

Example usage::

    deprecation("x will have to be a unyt_array in future versions.")

    @deprecated()
    def old_function():
        pass

    @deprecated("will be removed in v2.0")
    def old_function():
        pass

    warn("This is a warning message.")

"""

import warnings


def deprecation(message, category=FutureWarning):
    """Issue a deprecation warning to the end user.

    A message must be specified, and a category can be optionally specified.
    FutureWarning will, by default, warn the end user, DeprecationWarning
    will only warn when the user has set the PYTHONWARNINGS environment
    variable, and `PendingDeprecationWarning` can be used for far future
    deprecations.

    Args:
        message (str):
            The message to be displayed to the end user.
        category (Warning):
            The warning category to use. `FutureWarning` by default.

    """
    warnings.warn(message, category=category, stacklevel=3)


def deprecated(message=None, category=FutureWarning):
    """Decorate a function to mark it as deprecated.

    This decorator will issue a warning to the end user when the function is
    called. The message and category can be optionally specified, if not a
    default message will be used and `FutureWarning` will be issued (which will
    by default warn the end user unless explicitly silenced).

    Args:
        message (str):
            The message to be displayed to the end user. If None a default
            message will be used.
        category (Warning):
            The warning category to use. `FutureWarning` by default.

    """

    def _deprecated(func):
        def wrapped(*args, **kwargs):
            # Determine the specific deprecation message
            if message is None:
                _message = (
                    f"{func.__name__} is deprecated and "
                    "will be removed in a future version."
                )
            else:
                _message = f"{func.__name__} {message}"

            # Issue the warning
            deprecation(
                _message,
                category=category,
            )
            return func(*args, **kwargs)

        return wrapped

    return _deprecated


def warn(message, category=RuntimeWarning, stacklevel=3):
    """Issue a warning to the end user.

    A message must be specified, and a category can be optionally specified.
    RuntimeWarning will, by default, warn the end user, and can be silenced by
    setting the PYTHONWARNINGS environment variable.

    This function is a simple wrapper around the `warnings.warn` function but
    with a default stacklevel of 3, removing the need to specify it each time.

    Args:
        message (str):
            The message to be displayed to the end user.
        category (Warning):
            The warning category to use. `RuntimeWarning` by default.
        stacklevel (int):
            The number of stack levels to skip when displaying the warning.

    """
    warnings.warn(message, category=category, stacklevel=stacklevel)
