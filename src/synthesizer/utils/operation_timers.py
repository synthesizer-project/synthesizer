"""Helpers for interacting with accumulated operation timings."""

from contextlib import contextmanager
from functools import wraps

from synthesizer.extensions.timers import (
    get_operation_names,
    get_operation_source,
    get_operation_timings,
    reset_timings,
    tic,
    toc,
)


@contextmanager
def timer(operation_name):
    """Context manager that accumulates timing for a block of code.

    The context manager records timing using the existing ``tic``/``toc``
    machinery and guarantees that the matching ``toc`` call happens even if the
    wrapped block raises an exception.

    Args:
        operation_name (str):
            The operation name to use for the timing entry.

    Returns:
        Iterator[None]:
            A context manager that wraps a code block in the timing
            machinery.
    """
    # Start timing immediately before entering the wrapped block.
    tic(operation_name)
    try:
        # Yield control back to the caller while the timer is active.
        yield
    finally:
        # Always stop the timer, even if the wrapped block raises, so the
        # timing stack remains balanced.
        toc(operation_name)


def timed(operation_name=None):
    """Return a decorator that accumulates timing for a wrapped function.

    The decorator records timing using the existing ``tic``/``toc`` machinery
    and guarantees that the matching ``toc`` call happens even if the wrapped
    function raises an exception.

    Args:
        operation_name (str, optional):
            The operation name to use for the timing entry. If omitted, the
            wrapped function's ``__qualname__`` will be used.

    Returns:
        callable:
            A decorator that wraps a function in the timing machinery.
    """

    # Return the actual decorator so callers can optionally configure the
    # operation name at decoration time.
    def decorator(func):
        """Wrap a function so it contributes to the accumulated timings.

        Args:
            func (callable):
                The function to wrap with timing instrumentation.

        Returns:
            callable:
                A wrapped function with the same interface as ``func``.
        """
        # Resolve the timer label once at decoration time so each invocation
        # uses a stable operation name.
        timer_name = (
            func.__qualname__ if operation_name is None else operation_name
        )

        @wraps(func)
        def wrapped(*args, **kwargs):
            """Execute a wrapped function while accumulating timings.

            Args:
                *args:
                    Positional arguments passed to the wrapped function.
                **kwargs:
                    Keyword arguments passed to the wrapped function.

            Returns:
                The result returned by the wrapped function.
            """
            # Start timing immediately before calling the wrapped function.
            tic(timer_name)
            try:
                # Return the wrapped function result unchanged so the decorator
                # is transparent aside from its timing side effect.
                return func(*args, **kwargs)
            finally:
                # Always stop the timer, even if the wrapped function raises,
                # so the timing stack remains balanced.
                toc(timer_name)

        return wrapped

    return decorator


class OperationTimers:
    """Dictionary-like interface to accumulated operation timings.

    This class provides access to timing data accumulated by C++ and Python
    toc() calls. It behaves like a dictionary where keys are operation names
    and values are tuples of (cumulative_time, call_count, source).

    The underlying timing data is stored in C++ using atomic operations for
    thread-safe accumulation. This class simply provides a Pythonic interface
    to access that data.

    Example:
        >>> timers = OperationTimers()
        >>> timers.reset()
        >>> # ... run some operations that call tic()/toc() ...
        >>> print(timers.keys())
        ['Finding particle grid indices', 'Creating Sed', ...]
        >>> cumulative_time, count, source = timers['Creating Sed']
        >>> print(f"Total: {cumulative_time}s over {count} calls")
        Total: 0.00035s over 3 calls
        >>> print(timers.get_source('Creating Sed'))
        'Python'
    """

    def reset(self):
        """Clear all accumulated timings.

        This should be called before each profiling run to ensure fresh
        timing data is collected.
        """
        reset_timings()

    def keys(self):
        """Return list of operation names.

        Returns:
            list: List of operation names (str) that have accumulated timing
                data.
        """
        return get_operation_names()

    def __getitem__(self, key):
        """Get timing data for an operation.

        Args:
            key (str): Operation name.

        Returns:
            tuple: (cumulative_time, call_count, source) where:
                - cumulative_time (float): Total time in seconds across all
                  calls.
                - call_count (int): Number of times this operation was called.
                - source (str): "C" or "Python" indicating where the timing
                  originated.

        Raises:
            KeyError: If operation name doesn't exist.
        """
        return get_operation_timings(key)

    def __contains__(self, key):
        """Check if operation exists.

        Args:
            key (str): Operation name.

        Returns:
            bool: True if operation has accumulated timing data.
        """
        return key in self.keys()

    def get_source(self, key):
        """Get source ('C' or 'Python') for an operation.

        This is used to determine linestyle in plots (solid for C,
        dashed for Python).

        Args:
            key (str): Operation name.

        Returns:
            str: 'C' or 'Python'.

        Raises:
            KeyError: If operation name doesn't exist.
        """
        return get_operation_source(key)

    def items(self):
        """Iterate over (operation, timing_data) pairs.

        Yields:
            tuple: (operation_name, (cumulative_time, call_count, source)).
        """
        for key in self.keys():
            yield key, self[key]

    def __len__(self):
        """Return number of operations.

        Returns:
            int: Number of operations with accumulated timing data.
        """
        return len(self.keys())

    def __repr__(self):
        """String representation.

        Returns:
            str: Human-readable representation showing number of operations.
        """
        return f"OperationTimers({len(self)} operations)"
