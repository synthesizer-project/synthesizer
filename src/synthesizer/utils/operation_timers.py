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


def build_timing_analysis_rows(timing_data, total_elapsed):
    """Build sorted timing rows including overhead and total entries.

    Args:
        timing_data (dict):
            A timing dictionary keyed by operation name with ``seconds``,
            ``count``, and ``source`` entries for each operation.
        total_elapsed (float):
            The total elapsed wall-clock time represented by the analysis.

    Returns:
        list:
            A list of row dictionaries sorted by descending operation time,
            with trailing ``Overhead`` and ``Total`` summary rows appended.
    """
    timed_elapsed = sum(data["seconds"] for data in timing_data.values())
    untimed_elapsed = max(total_elapsed - timed_elapsed, 0.0)

    rows = []
    for operation, data in timing_data.items():
        fraction = (
            data["seconds"] / total_elapsed * 100.0
            if total_elapsed > 0.0
            else 0.0
        )
        rows.append(
            {
                "operation": operation,
                "seconds": data["seconds"],
                "fraction_percent": fraction,
                "count": data["count"],
                "source": data["source"],
            }
        )

    rows.sort(key=lambda row: row["seconds"], reverse=True)
    rows.append(
        {
            "operation": "Overhead",
            "seconds": untimed_elapsed,
            "fraction_percent": (
                untimed_elapsed / total_elapsed * 100.0
                if total_elapsed > 0.0
                else 0.0
            ),
            "count": None,
            "source": "N/A",
        }
    )
    rows.append(
        {
            "operation": "Total",
            "seconds": total_elapsed,
            "fraction_percent": 100.0 if total_elapsed > 0.0 else 0.0,
            "count": None,
            "source": "N/A",
        }
    )

    return rows


def print_timing_analysis_table(rows, print_func=print):
    """Print a timing analysis table to stdout.

    Args:
        rows (list):
            The timing rows produced by ``build_timing_analysis_rows``.
        print_func (callable):
            The function used to emit each formatted line.

    Returns:
        None
    """
    filtered_rows = []
    for row in rows:
        if row["operation"] in ("Overhead", "Total"):
            filtered_rows.append(row)
            continue

        if row["fraction_percent"] >= 0.01:
            filtered_rows.append(row)

    formatted_rows = []
    for row in filtered_rows:
        if 0.0 < row["seconds"] < 0.01:
            seconds_str = f"{row['seconds']:.2e}"
        else:
            seconds_str = f"{row['seconds']:.2f}"

        formatted_rows.append(
            {
                "operation": row["operation"],
                "seconds": seconds_str,
                "fraction_percent": f"{row['fraction_percent']:.2f}",
                "count": "-" if row["count"] is None else str(row["count"]),
                "source": row["source"],
            }
        )

    operation_width = max(
        len("Operation"), *(len(r["operation"]) for r in formatted_rows)
    )
    time_width = max(
        len("Time (s)"), *(len(r["seconds"]) for r in formatted_rows)
    )
    frac_width = max(
        len("Fraction (%)"),
        *(len(r["fraction_percent"]) for r in formatted_rows),
    )
    count_width = max(len("Count"), *(len(r["count"]) for r in formatted_rows))
    source_width = max(
        len("Source"), *(len(r["source"]) for r in formatted_rows)
    )

    divider = (
        "+"
        + "-" * (operation_width + 2)
        + "+"
        + "-" * (time_width + 2)
        + "+"
        + "-" * (frac_width + 2)
        + "+"
        + "-" * (count_width + 2)
        + "+"
        + "-" * (source_width + 2)
        + "+"
    )

    print_func(divider)
    print_func(
        "| "
        + f"{'Operation':<{operation_width}}"
        + " | "
        + f"{'Time (s)':>{time_width}}"
        + " | "
        + f"{'Fraction (%)':>{frac_width}}"
        + " | "
        + f"{'Count':>{count_width}}"
        + " | "
        + f"{'Source':<{source_width}}"
        + " |"
    )
    print_func(divider)

    for row in formatted_rows:
        print_func(
            "| "
            + f"{row['operation']:<{operation_width}}"
            + " | "
            + f"{row['seconds']:>{time_width}}"
            + " | "
            + f"{row['fraction_percent']:>{frac_width}}"
            + " | "
            + f"{row['count']:>{count_width}}"
            + " | "
            + f"{row['source']:<{source_width}}"
            + " |"
        )

    print_func(divider)


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
        >>> OperationTimers.print_table()
    """

    @classmethod
    def snapshot(cls):
        """Return the accumulated timings as a plain dictionary.

        Returns:
            dict:
                A dictionary keyed by operation name containing ``seconds``,
                ``count``, and ``source`` entries for each accumulated timing.
        """
        timers = cls()
        timing_data = {}
        for operation in timers.keys():
            cumulative_time, call_count, source = timers[operation]
            timing_data[operation] = {
                "seconds": cumulative_time,
                "count": call_count,
                "source": source,
            }

        return timing_data

    @classmethod
    def build_rows(cls, total_elapsed=None):
        """Build sorted timing rows matching the pipeline timing report.

        Args:
            total_elapsed (float, optional):
                The total elapsed wall-clock time represented by the timings.
                If omitted, the sum of all accumulated timed operations is
                used, which yields a zero-overhead summary.

        Returns:
            list:
                The timing rows produced by
                :func:`build_timing_analysis_rows`.
        """
        timing_data = cls.snapshot()
        if total_elapsed is None:
            total_elapsed = sum(
                data["seconds"] for data in timing_data.values()
            )

        return build_timing_analysis_rows(timing_data, total_elapsed)

    @classmethod
    def print_table(cls, total_elapsed=None, print_func=print):
        """Print a pipeline-style timing breakdown table.

        Args:
            total_elapsed (float, optional):
                The total elapsed wall-clock time represented by the timings.
                If omitted, the sum of all accumulated timed operations is
                used, so the printed table shows no overhead row contribution.
            print_func (callable):
                The function used to emit the table lines. Defaults to
                :func:`print`.

        Returns:
            list:
                The rows that were printed.
        """
        rows = cls.build_rows(total_elapsed=total_elapsed)
        print_timing_analysis_table(rows, print_func=print_func)
        return rows

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
