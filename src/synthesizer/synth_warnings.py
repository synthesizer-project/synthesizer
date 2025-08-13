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

import inspect
import os
import textwrap
import warnings
from pathlib import Path


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


def _find_user_stacklevel():
    """Find the appropriate stack level to point to user code.

    This walks up the call stack and finds the first frame that appears
    to be user code (not part of your library).

    Returns:
        int: The stack level to use for warnings.warn()
    """
    frame = inspect.currentframe()
    try:
        # Start from the caller of warn() (skip current frame and warn())
        level = 2
        frame = frame.f_back.f_back  # Skip _find_user_stacklevel() and warn()

        while frame is not None:
            filename = frame.f_code.co_filename

            # Check if this frame is from user code
            if _is_user_code(filename):
                return level

            level += 1
            frame = frame.f_back

        # Fallback if we can't find user code
        return 2

    finally:
        # Always clean up frame references to avoid memory leaks
        del frame


def _is_user_code(filename):
    """Determine if a filename represents user code vs library code.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if this appears to be user code.
    """
    filename = str(filename).replace("\\", "/")  # Normalize path separators

    # Patterns that indicate library/internal code
    library_patterns = [
        "synthesizer/",  # Your main library
        "site-packages/",  # Installed packages
        "/lib/python",  # Standard library
        "/lib64/python",  # Standard library (64-bit)
        "<frozen",  # Frozen modules
        "<built-in>",  # Built-in functions
        "importlib/",  # Import machinery
        "pkgutil.py",  # Package utilities
    ]

    # Check if this looks like library code
    for pattern in library_patterns:
        if pattern in filename:
            return False

    # Additional checks for common library indicators
    path_parts = Path(filename).parts
    for part in path_parts:
        # Common library directory names
        if part in ("site-packages", "__pycache__", ".tox", "venv", "env"):
            return False

    # If it's not obviously library code, assume it's user code
    return True


def _estimate_warning_prefix(stacklevel, category):
    """Estimate the warning prefix that will be added by warnings.warn().

    Args:
        stacklevel (int): The stack level for the warning.
        category (Warning): The warning category.

    Returns:
        str: Estimated prefix string.
    """
    frame = inspect.currentframe()
    try:
        # Navigate to the target frame
        target_frame = frame
        for _ in range(stacklevel + 1):  # +1 to account for this function
            if target_frame.f_back is not None:
                target_frame = target_frame.f_back

        filename = target_frame.f_code.co_filename
        lineno = target_frame.f_lineno

        # Format: "filename:line: CategoryName: "
        return f"{filename}:{lineno}: {category.__name__}: "

    except (AttributeError, TypeError):
        # Fallback estimate
        return f"<file>:<line>: {category.__name__}: "
    finally:
        del frame


def _wrap_with_prefix(message, stacklevel, category):
    """Wrap text accounting for a prefix on the first line.

    This function properly wraps text when the first line has a prefix
    (like warning headers) and subsequent lines should be indented.

    Args:
        message (str): The message to wrap.
        stacklevel (int): The stack level for the warning.
        category (Warning): The warning category.

    Returns:
        str: The wrapped message text (without the prefix).
    """
    # Get terminal width
    try:
        width = os.get_terminal_size().columns
    except (AttributeError, OSError):
        width = 80

    # Calculate available width for the message
    prefix_length = len(_estimate_warning_prefix(stacklevel, category))
    available_width = max(40, width - prefix_length - 2)  # Leave small margin

    # Wrap the message
    wrapped = textwrap.fill(
        str(message),
        width=available_width,
        break_long_words=False,
        break_on_hyphens=False,
        expand_tabs=False,
    )

    return wrapped


def warn(message, category=RuntimeWarning, stacklevel=None):
    """Issue a warning to the end user with proper text wrapping.

    A message must be specified, and a category can be optionally specified.
    RuntimeWarning will, by default, warn the end user, and can be silenced by
    setting the PYTHONWARNINGS environment variable.

    This function automatically determines the appropriate stack level to point
    to user code rather than library internals. The message will be wrapped to
    fit the terminal width for better readability, accounting for the warning
    prefix that gets added by the warnings system.

    Args:
        message (str):
            The message to be displayed to the end user.
        category (Warning):
            The warning category to use. `RuntimeWarning` by default.
        stacklevel (int, optional):
            The number of stack levels to skip when displaying the warning.
            If None (default), automatically determines the appropriate level
            to point to user code.
    """
    # Auto-determine stacklevel if not provided
    if stacklevel is None:
        stacklevel = _find_user_stacklevel()

    # Estimate the warning prefix and wrap the message accordingly
    wrapped_message = _wrap_with_prefix(message, stacklevel, category)

    warnings.warn(wrapped_message, category=category, stacklevel=stacklevel)
