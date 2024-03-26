"""
This is a helper script to quickly make plots from profiling outputs.

To use this diagnostic script first profile a script redirecting the output to
a file:
python -m profile <script>.py > <script_profile_file>

Then input that file to this script along with a location for the plots:
python profile_plot.py <script_profile_file> /path/to/plot/directory/

This will print an output of operations that took >5% of the time and make bar
charts.

Note that percentages are not exclusive, functions nested in other functions
will exhibit the same runtime.
"""

import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def make_report(
    funcs: NDArray[Any],
    ncalls: NDArray[np.int32],
    tottime: NDArray[np.float64],
    pcent: NDArray[np.float64],
    col_width: int,
    numeric_width: int = 14,
) -> str:
    """
    Generates a formatted string report of profiling data.

    Args:
        funcs: List of function names.
        ncalls: List of call counts for each function.
        tottime: List of total execution times for each function.
        pcent: List of percentages of total runtime for each function.
        col_width: Column width for the function names.
        numeric_width: Width for numeric columns (default is 14).

    Returns:
        A string representing the formatted report.
    """
    # Define string holding the table
    report_string: str = ""

    # Check the initial width is large enough
    if numeric_width < 14:
        numeric_width = 14

    # Make the header
    head: str = "|"
    head += "Function call".ljust(col_width) + "|"
    head += "ncalls".ljust(numeric_width) + "|"
    head += "tottime (s)".ljust(numeric_width) + "|"
    head += "percall (ms)".ljust(numeric_width) + "|"
    head += "percent (%)".ljust(numeric_width) + "|"

    # Include the header
    report_string += "=" * len(head) + "\n"
    report_string += head + "\n"

    # Include a horizontal line
    report_string += "+"
    for n in range(5):
        if n == 0:
            report_string += "=" * col_width + "+"
        else:
            report_string += "=" * numeric_width + "+"
    report_string += "\n"

    # Loop over making each row
    for ind, func in enumerate(funcs):
        # Make this row of the table
        row_string: str = ""
        row_string += "|" + func.strip("\n").ljust(col_width) + "|"
        row_string += str(int(ncalls[ind])).ljust(numeric_width) + "|"
        row_string += f"{tottime[ind]:.4f}".ljust(numeric_width) + "|"
        row_string += (
            f"{tottime[ind] / ncalls[ind] * 1000:.4f}".ljust(numeric_width)
            + "|"
        )
        row_string += f"{pcent[ind]:.2f}".ljust(numeric_width) + "|"
        row_string += "\n"

        report_string += row_string

        # Do we need to start again with a larger column width?
        if len(func) + 2 > col_width:
            return make_report(
                funcs,
                ncalls,
                tottime,
                pcent,
                col_width=len(func.ljust(col_width)) + 1,
                numeric_width=numeric_width,
            )
        elif len(str(int(ncalls[ind]))) > numeric_width:
            return make_report(
                funcs,
                ncalls,
                tottime,
                pcent,
                col_width,
                numeric_width=len(str(int(ncalls[ind]))) + 1,
            )

    # Close off the bottom of the table
    report_string += "=" * len(head)

    return report_string


if __name__ == "__main__":
    profile_file: str = sys.argv[1]
    plot_loc: str = sys.argv[2]

    ncalls_dict: Dict[str, float] = {}
    tottime_dict: Dict[str, float] = {}
    extract_data: bool = False

    with open(profile_file, "r") as file:
        for line in file:
            line_split = [s for s in line.split(" ") if s.strip()]

            if line_split and line_split[-1] == "seconds\n":
                extract_data = True
                total_runtime: float = float(line_split[-2])
                print(f"Total Runtime: {total_runtime:.2f} seconds")

            if (
                extract_data
                and len(line_split) == 6
                and line_split[0].strip() != "ncalls"
            ):
                func_name: str = line_split[-1]
                if "/" in line_split[0]:
                    ncalls_dict[func_name] = max(
                        float(val) for val in line_split[0].split("/")
                    )
                else:
                    ncalls_dict[func_name] = float(line_split[0])
                    tottime_dict[func_name] = float(line_split[1])

    funcs: NDArray[Any] = np.array(list(ncalls_dict.keys()))
    ncalls: NDArray[np.int32] = np.array(list(ncalls_dict.values()))
    tottime: NDArray[np.float64] = np.array(list(tottime_dict.values()))

    # Mask away inconsequential operations
    okinds: NDArray[np.bool_] = tottime > 0.05 * total_runtime
    funcs = funcs[okinds]
    ncalls = ncalls[okinds]
    tottime = tottime[okinds]

    # Compute the percentage of runtime spent doing operations
    pcent = tottime / total_runtime * 100

    # Sort arrays in descending cumaltive time order
    sinds: NDArray[np.int64] = np.argsort(tottime)[::-1]
    funcs = funcs[sinds]
    ncalls = ncalls[sinds]
    tottime = tottime[sinds]
    pcent = pcent[sinds]

    # Remove the script call itself and other uninteresting operations
    okinds = np.ones(funcs.size, dtype=bool)
    for ind, func in enumerate(funcs):
        if "<module>" in func or "exec" in func:
            okinds[ind] = False
    funcs = funcs[okinds]
    ncalls = ncalls[okinds]
    tottime = tottime[okinds]
    pcent = pcent[okinds]

    # Clean up the function labels a bit
    for ind, func in enumerate(funcs):
        # Split the function signature
        func_split = [
            s
            for s1 in func.split(":")
            for s2 in s1.split("(")
            for s in s2.split(")")
        ]

        if len(func_split[0]) > 0:
            funcs[ind] = func_split[0] + ":" + func_split[2]
        else:
            funcs[ind] = func_split[2]

    # Report the results to the user
    col_width: int = 15
    report_string: str = make_report(funcs, ncalls, tottime, pcent, col_width)

    print(report_string)

    # Get the name of this file
    file_name: str = profile_file.split("/")[-1].split(".")[0]

    # And write the table to a file
    with open(plot_loc + file_name + "_report.txt", "w") as text_file:
        text_file.write(report_string)

    # Now make a plot of the percentage time taken
    fig: Figure = plt.figure(figsize=(3.5, 3.5))
    ax: Axes = fig.add_subplot(111)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.scatter(funcs, pcent, marker="+")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    ax.set_title(file_name)
    ax.set_ylabel("Percentage (%)")
    outpng = plot_loc + file_name + "_percent.png"
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    # Now make a plot of the number of calls
    fig1: Figure = plt.figure(figsize=(3.5, 3.5))
    ax1: Axes = fig.add_subplot(111)
    ax1.semilogy()
    ax1.grid(True)
    ax1.set_axisbelow(True)
    ax1.bar(funcs, ncalls, width=1, edgecolor="grey", alpha=0.8)
    ax1.set_xticks(ax.get_xticks())
    ax1.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    ax1.set_title(file_name)
    ax1.set_ylabel("Number of calls")
    outpng = plot_loc + file_name + "_ncalls.png"
    fig1.savefig(outpng, bbox_inches="tight")
    plt.close(fig1)

    # Now make a plot of the number of calls
    fig2: Figure = plt.figure(figsize=(3.5, 3.5))
    ax2: Axes = fig.add_subplot(111)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.scatter(funcs, tottime, marker="+")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    ax2.set_title(file_name)
    ax2.set_ylabel("Runtime (s)")
    outpng = plot_loc + file_name + "_tot_time.png"
    fig2.savefig(outpng, bbox_inches="tight")
    plt.close(fig2)
