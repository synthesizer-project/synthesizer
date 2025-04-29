"""This is a helper script to quickly make plots from profiling outputs.

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

import matplotlib.pyplot as plt
import numpy as np


def make_report(funcs, ncalls, tottime, pcent, col_width, numeric_width=14):
    """Translate the profiling data into a table.

    Args:
        funcs (list): List of function names.
        ncalls (list): List of number of calls to each function.
        tottime (list): List of total time taken by each function.
        pcent (list): List of percentage time taken by each function.
        col_width (int): Width of the first column.
        numeric_width (int): Width of the numeric columns.
    """
    # Define string holding the table
    report_string = ""

    # Check the initial width is large enough
    if numeric_width < 14:
        numeric_width = 14

    # Make the header
    head = "|"
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
        row_string = ""
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
    # Get the commandline inputs
    profile_file = sys.argv[1]
    plot_loc = sys.argv[2]

    # Create some dictionaries
    ncalls = {}
    tottime = {}

    # Create a flag to skip non-profiling information in the output
    extract_data = False

    # Open the profile file
    with open(profile_file, "r") as file:
        for iline, line in enumerate(file):
            line_split = [s for s in line.split(" ") if s != " " and s != ""]

            # Flag that we found the table (this is very primitive)
            if (
                line_split[-1] == "seconds\n"
                and line_split[-3] == "in"
                and line_split[-4] == "calls)"
            ):
                # Flag that we can now extract data
                extract_data = True

                # Get the total runtime
                total_runtime = float(line_split[-2])
                print("Total Runtime: %.2f seconds" % total_runtime)

            # Have we reached the profiling information?
            if extract_data:
                # Are we in the table?
                if len(line_split) == 6 and line_split[0].strip() != "ncalls":
                    # Store the data
                    if "/" in line_split[0]:
                        ncalls[line_split[-1]] = np.max(
                            [float(val) for val in line_split[0].split("/")]
                        )
                    else:
                        ncalls[line_split[-1]] = float(line_split[0])
                        tottime[line_split[-1]] = float(line_split[1])

    # Convert dictionaries to arrays to manipulate, report and plot
    funcs = np.array(list(ncalls.keys()))
    ncalls = np.array(list(ncalls.values()))
    tottime = np.array(list(tottime.values()))

    # Mask away inconsequential operations
    okinds = tottime > 0.05 * total_runtime
    funcs = funcs[okinds]
    ncalls = ncalls[okinds]
    tottime = tottime[okinds]

    # Compute the percentage of runtime spent doing operations
    pcent = tottime / total_runtime * 100

    # Sort arrays in descending cumaltive time order
    sinds = np.argsort(tottime)[::-1]
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
    col_width = 15
    report_string = make_report(funcs, ncalls, tottime, pcent, col_width)

    print(report_string)

    # Get the name of this file
    file_name = profile_file.split("/")[-1].split(".")[0]

    # And write the table to a file
    with open(plot_loc + file_name + "_report.txt", "w") as text_file:
        text_file.write(report_string)

    # Now make a plot of the percentage time taken
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
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
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.bar(funcs, ncalls, width=1, edgecolor="grey", alpha=0.8)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    ax.set_title(file_name)
    ax.set_ylabel("Number of calls")
    outpng = plot_loc + file_name + "_ncalls.png"
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    # Now make a plot of the number of calls
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.scatter(funcs, tottime, marker="+")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    ax.set_title(file_name)
    ax.set_ylabel("Runtime (s)")
    outpng = plot_loc + file_name + "_tot_time.png"
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)
