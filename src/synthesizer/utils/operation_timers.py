"""Dict-like interface to C++ accumulated operation timings."""

from synthesizer.extensions.timers import (
    get_operation_names,
    get_operation_source,
    get_operation_timings,
    reset_timings,
)


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
        >>> # ... run some operations that call toc() ...
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
