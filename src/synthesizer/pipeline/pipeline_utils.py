"""A submodule with helpers for writing out Synthesizer pipeline results."""

import copy
import inspect
import sys
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Unit, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.emissions.line import LineCollection
from synthesizer.imaging import Image, SpectralCube
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.instruments import InstrumentCollection
from synthesizer.photometry import PhotometryCollection
from synthesizer.synth_warnings import warn
from synthesizer.units import unit_is_compatible
from synthesizer.utils.operation_timers import (
    timed,
)

# Special model label for operations that are not tied to a specific model.
# This can be used for operations such as SFZH / SFH with no relation to
# an emission model.
NO_MODEL_LABEL = "no_model_label"


@timed("clear_pipeline_outputs")
def clear_pipeline_outputs(gal):
    """Clear additive pipeline outputs from a galaxy and components.

    Args:
        gal:
            The galaxy whose additive pipeline outputs should be reset.

    Returns:
        None
    """
    # Clear the galaxy and any attached components
    for obj in (gal, gal.stars, gal.gas, gal.black_holes):
        if obj is None:
            continue

        # Clear out all the emission containers and caches
        for attr, value in (
            ("spectra", {}),
            ("lines", {}),
            ("photo_lnu", {}),
            ("photo_fnu", {}),
            ("spectroscopy", {}),
            ("images_lnu", {}),
            ("images_fnu", {}),
            ("images_psf_lnu", {}),
            ("images_psf_fnu", {}),
            ("images_noise_lnu", {}),
            ("images_noise_fnu", {}),
            ("particle_spectra", {}),
            ("particle_lines", {}),
            ("particle_photo_lnu", {}),
            ("particle_photo_fnu", {}),
            ("particle_spectroscopy", {}),
            ("data_cubes_lnu", {}),
            ("data_cubes_fnu", {}),
            ("model_param_cache", {}),
            ("_grid_weights", {"cic": {}, "ngp": {}}),
            ("sfh", None),
            ("sfzh", None),
        ):
            if hasattr(obj, attr):
                setattr(obj, attr, value)


@timed("accumulate_pipeline_results_from_child")
def accumulate_pipeline_results_from_child(parent, *children):
    """Accumulate additive pipeline outputs from child galaxies.

    Args:
        parent:
            The parent galaxy receiving accumulated outputs.
        *children:
            Child galaxies whose additive pipeline outputs should be combined
            onto the parent.

    Returns:
        object:
            The parent galaxy after accumulation.
    """

    def combine(current, other):
        # Recursively combine any additive pipeline outputs, preserving nested
        # dictionary structure and using object-specific addition where needed.
        if other is None:
            return current
        if current is None:
            return other

        # Handle the dictionary recursive case
        if isinstance(current, dict):
            combined = copy.deepcopy(current)
            for key, value in other.items():
                combined[key] = combine(combined.get(key), value)
            return combined

        # Use overloaded object addition if possible.
        if isinstance(
            current,
            (
                Sed,
                LineCollection,
                Image,
                ImageCollection,
                SpectralCube,
                PhotometryCollection,
            ),
        ):
            return current + other

        return current + other

    for child in children:
        # First combine any additive outputs stored directly on the galaxy.
        for attr in (
            "spectra",
            "lines",
            "photo_lnu",
            "photo_fnu",
            "spectroscopy",
            "images_lnu",
            "images_fnu",
            "images_psf_lnu",
            "images_psf_fnu",
            "images_noise_lnu",
            "images_noise_fnu",
            "data_cubes_lnu",
            "data_cubes_fnu",
        ):
            if hasattr(child, attr):
                setattr(
                    parent,
                    attr,
                    combine(
                        getattr(parent, attr, None),
                        getattr(child, attr),
                    ),
                )

        # Then combine additive component-level outputs, but skip shared
        # components that were intentionally attached directly to the child.
        for name in ("stars", "gas", "black_holes"):
            parent_component = getattr(parent, name, None)
            child_component = getattr(child, name, None)
            if parent_component is None or child_component is None:
                continue
            if child_component is parent_component:
                continue

            for attr in (
                "spectra",
                "lines",
                "photo_lnu",
                "photo_fnu",
                "spectroscopy",
                "images_lnu",
                "images_fnu",
                "images_psf_lnu",
                "images_psf_fnu",
                "images_noise_lnu",
                "images_noise_fnu",
                "sfh",
                "sfzh",
            ):
                if hasattr(child_component, attr):
                    setattr(
                        parent_component,
                        attr,
                        combine(
                            getattr(parent_component, attr, None),
                            getattr(child_component, attr),
                        ),
                    )

    return parent


def get_atomic_timing_snapshot():
    """Return the current atomic timing snapshot for this process.

    Args:
        None

    Returns:
        dict:
            A dictionary keyed by operation name containing cumulative timing
            information with ``seconds``, ``count``, and ``source`` entries.
    """
    # Import the timer wrapper lazily to avoid unnecessary import cost in code
    # paths that never analyse timings.
    from synthesizer.utils.operation_timers import OperationTimers

    # Convert the OperationTimers interface into a plain dictionary that is
    # easier to gather, merge, serialise, and plot.
    timers = OperationTimers()
    timing_data = {}
    for operation in timers.keys():
        cumulative_time, call_count, source = timers[operation]
        timing_data[operation] = {
            "seconds": cumulative_time,
            "count": call_count,
            "source": source,
        }

    return timing_data


def combine_atomic_timing_snapshots(
    comm, using_mpi, rank, timing_data, total_elapsed
):
    """Combine timing snapshots across ranks when running under MPI.

    Args:
        comm:
            The MPI communicator associated with the Pipeline.
        using_mpi (bool):
            Whether the Pipeline is currently running under MPI.
        rank (int):
            The rank of the current process.
        timing_data (dict):
            The local timing snapshot for this rank.
        total_elapsed (float):
            The wall-clock time elapsed on this rank since Pipeline
            instantiation.

    Returns:
        tuple:
            A pair ``(timing_data, total_elapsed)``. On non-MPI runs this is
            just the input data. On MPI runs rank 0 receives the aggregated
            timings and summed elapsed time, while all other ranks receive
            ``(None, None)``.
    """
    # In the non-MPI case there is nothing to merge, so return the local
    # timing snapshot unchanged.
    if not using_mpi:
        return timing_data, total_elapsed

    # Gather both the detailed timings and the total elapsed time from every
    # rank so rank 0 can produce a combined report.
    gathered_timings = comm.gather(timing_data, root=0)
    gathered_elapsed = comm.gather(total_elapsed, root=0)

    # Non-root ranks do not write timing outputs, so they can return early.
    if rank != 0:
        return None, None

    # Merge operation timings rank-by-rank, summing both cumulative time and
    # invocation count for matching operation names.
    combined = {}
    for rank_timings in gathered_timings:
        for operation, data in rank_timings.items():
            if operation not in combined:
                combined[operation] = {
                    "seconds": 0.0,
                    "count": 0,
                    "source": data["source"],
                }

            combined[operation]["seconds"] += data["seconds"]
            combined[operation]["count"] += data["count"]

            if combined[operation]["source"] != data["source"]:
                combined[operation]["source"] = "Mixed"

    # Sum the per-rank elapsed times so the overhead contribution is measured
    # in the same rank-seconds units as the atomic timing totals.
    return combined, sum(gathered_elapsed)


def write_timing_analysis_summary(rows, outdir):
    """Write the timing analysis summary CSV.

    Args:
        rows (list):
            The timing rows produced by ``build_timing_analysis_rows``.
        outdir (str or Path):
            The directory where the timing summary CSV should be written.

    Returns:
        None
    """
    # Normalise the output path so callers can pass either strings or Path
    # objects.
    outdir = Path(outdir)
    summary_path = outdir / "timing_summary.csv"

    # Write a compact CSV summary that can be inspected manually or reused by
    # other analysis scripts later.
    with open(summary_path, "w") as handle:
        handle.write("operation,seconds,fraction_percent,count,source\n")
        for row in rows:
            count = "" if row["count"] is None else row["count"]
            handle.write(
                f"{row['operation']},{row['seconds']:.12f},"
                f"{row['fraction_percent']:.6f},{count},{row['source']}\n"
            )


def plot_timing_analysis(rows, outdir):
    """Write timing analysis plots to disk.

    Args:
        rows (list):
            The timing rows produced by ``build_timing_analysis_rows``.
        outdir (str or Path):
            The directory where the timing plots should be written.

    Returns:
        None
    """
    # Normalise the output path and drop the synthetic total row, since that
    # row is only useful in the textual summary.
    outdir = Path(outdir)
    plot_rows = [row for row in rows if row["operation"] != "Total"]
    nonzero_rows = [row for row in plot_rows if row["seconds"] > 0.0]

    # Build the pie chart from non-zero rows only so the chart focuses on the
    # operations that actually contributed measurable time.
    if nonzero_rows:
        total_seconds = sum(row["seconds"] for row in nonzero_rows)
        pie_rows = []
        other_seconds = 0.0

        # Group the smallest contributions into a single slice so the pie chart
        # stays readable without a forest of overlapping labels.
        for row in nonzero_rows:
            fraction_percent = (
                row["seconds"] / total_seconds * 100.0
                if total_seconds > 0.0
                else 0.0
            )
            if fraction_percent < 1.0:
                other_seconds += row["seconds"]
            else:
                pie_rows.append(row)

        if other_seconds > 0.0:
            pie_rows.append(
                {
                    "operation": "Other <1%",
                    "seconds": other_seconds,
                    "source": "N/A",
                }
            )

        # Use a categorical palette for the visible wedges while reserving
        # neutral colours for the synthetic overhead and grouped slices.
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(pie_rows), 1)))
        colors = []
        palette_index = 0
        for row in pie_rows:
            if row["operation"] == "Overhead":
                colors.append("#7f7f7f")
            elif row["operation"] == "Other <1%":
                colors.append("#bab0ac")
            else:
                colors.append(palette[palette_index])
                palette_index += 1

        # Save the main pie chart showing the fractional timing breakdown.
        fig, ax = plt.subplots(figsize=(9, 6))
        wedges, _ = ax.pie(
            [row["seconds"] for row in pie_rows],
            labels=None,
            startangle=90,
            counterclock=False,
            colors=colors,
        )

        # Put the operation names in a legend instead of directly on the pie
        # wedges so the figure stays readable when several slices are present.
        legend_labels = []
        for row in pie_rows:
            fraction_percent = (
                row["seconds"] / total_seconds * 100.0
                if total_seconds > 0.0
                else 0.0
            )
            legend_labels.append(
                f"{row['operation']} ({fraction_percent:.1f}%)"
            )
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(
            outdir / "timing_pie.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)
    else:
        # Fall back to a simple placeholder figure when there is no timing data
        # to visualise yet.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No timing data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "timing_pie.png", dpi=200)
        plt.close(fig)

    # Restrict the bar chart to the same >1% contributions used in the pie
    # chart so both plots focus on the materially important timing costs.
    bar_rows = []
    if nonzero_rows:
        total_seconds = sum(row["seconds"] for row in nonzero_rows)
        for row in plot_rows:
            fraction_percent = (
                row["seconds"] / total_seconds * 100.0
                if total_seconds > 0.0
                else 0.0
            )
            if fraction_percent >= 1.0:
                bar_rows.append(row)

    # Build the bar chart from the filtered rows so absolute timings are
    # available alongside the fractional pie chart view.
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * max(len(bar_rows), 1))))
    y_positions = np.arange(len(bar_rows))
    bar_colors = []

    # Colour the bar chart by timing source while keeping overhead visually
    # distinct from both Python and C timings.
    for row in bar_rows:
        if row["operation"] == "Overhead":
            bar_colors.append("#7f7f7f")
        elif row["source"] == "C":
            bar_colors.append("#4c72b0")
        elif row["source"] == "Python":
            bar_colors.append("#dd8452")
        else:
            bar_colors.append("#937860")

    # Save the horizontal bar chart ordered consistently with the input rows.
    ax.barh(
        y_positions,
        [row["seconds"] for row in bar_rows],
        color=bar_colors,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["operation"] for row in bar_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)")
    ax.grid(axis="x", alpha=0.3)

    # Add a legend explaining the meaning of the bar colours so the source of
    # each timing contribution is clear without inspecting the code.
    legend_entries = []
    legend_labels = []
    seen_labels = set()
    for row, color in zip(bar_rows, bar_colors):
        if row["operation"] == "Overhead":
            label = "Overhead"
        elif row["source"] == "C":
            label = "C++"
        elif row["source"] == "Python":
            label = "Python"
        else:
            label = row["source"]

        if label in seen_labels:
            continue

        seen_labels.add(label)
        legend_entries.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        )
        legend_labels.append(label)

    if legend_entries:
        ax.legend(legend_entries, legend_labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "timing_bar.png", dpi=200)
    plt.close(fig)


def discover_attr_paths_recursive(obj, prefix="", output_set=None):
    """Recursively discover all outputs attached to an object.

    This function will collate all paths to attributes at any level within
    the input object.

    If the object is a dictionary, we will loop over all keys and values
    recursing where appropriate.

    If the object is a class instance (e.g. Galaxy, Stars,
    ImageCollection, etc.), we will loop over all attributes and
    recurse where appropriate.

    If the object is a "value" (i.e. an array or a scalar), we will append
    the full path to the output list.

    NOTE: this function is currently unused but is kept for debugging purposes
    since it is extremely useful to see the nesting of attributes on objects.

    Args:
        obj (dict):
            The dictionary to search.
        prefix (str):
            A prefix to add to the keys of the arrays.
        output_set (set):
            A set to store the output paths in.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(obj, dict):
        for k, v in obj.items():
            output_set = discover_attr_paths_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # If it's a class instance and not a leaf type
    elif hasattr(obj, "__class__") and not isinstance(
        obj, (unyt_array, unyt_quantity, np.ndarray, str, bool, int, float)
    ):
        members = inspect.getmembers(
            obj.__class__, lambda a: isinstance(a, property)
        )
        prop_names = {name for name, _ in members}

        # Collect public instance attributes if the object has a __dict__
        # attribute
        if hasattr(obj, "__dict__"):
            keys = set(vars(obj).keys())
        else:
            # Otherwise, just collect the property names
            keys = set()
        keys.update(prop_names)

        for k in keys:
            # Handle Quantity objects
            if hasattr(obj, k[1:]):
                k = k[1:]

            # Skip private attributes
            if k.startswith("_"):
                continue

            try:
                v = getattr(obj, k)
            except Exception:
                continue  # Skip properties that raise errors

            # Skip if None
            if v is None:
                continue

            discover_attr_paths_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # Nothing to do if its an unset optional value
    elif obj is None:
        return output_set

    # Skip undesirable types
    elif isinstance(obj, (str, bool)):
        return output_set

    # Otherwise, we have something we need to write out so add the path to
    # the set
    else:
        output_set.add(prefix.replace(" ", "_"))

    return output_set


def discover_dict_recursive(data, prefix="", output_set=None):
    """Recursively discover all leaves in a dictionary.

    Args:
        data (dict): The dictionary to search.
        prefix (str): A prefix to add to the keys of the arrays.
        output_set (set): A set to store the output paths in.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(data, dict):
        for k, v in data.items():
            output_set = discover_dict_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # Otherwise, we have something we need to write out so add the path to
    # the set
    else:
        output_set.add(prefix[1:].replace(" ", "_"))

    return output_set


def discover_dict_structure(data):
    """Recursively discover the structure of a dictionary.

    Args:
        data (dict):
            The dictionary to search.

    Returns:
        dict:
            A dictionary of all the paths in the input dictionary.
    """
    # Set up the set to hold the global output paths
    output_set = set()

    # Loop over the galaxies and recursively discover the outputs
    output_set = discover_dict_recursive(data, output_set=output_set)

    return output_set


def count_and_check_dict_recursive(data, prefix=""):
    """Recursively count the number of leaves in a dictionary.

    Args:
        data (dict): The dictionary to search.
        prefix (str): A prefix to add to the keys of the arrays.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    count = 0

    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(data, dict):
        for k, v in data.items():
            count += count_and_check_dict_recursive(
                v,
                prefix=f"{prefix}/{k}",
            )
        return count

    # Otherwise, we are at a leaf with some data to account for. Check the
    # result makes sense.The count is always the first element of the
    # shape tuple
    if data is None:
        raise exceptions.BadResult(
            f"Found a NoneType object at {prefix}. "
            "All results should be numeric with associated units."
        )

    if not hasattr(data, "units") and isinstance(data, np.ndarray):
        raise exceptions.BadResult(
            f"Found an array object without units at {prefix}. "
            "All results should be numeric with associated units. "
            f"Data: {data}"
        )

    if not hasattr(data, "shape"):
        raise exceptions.BadResult(
            f"Found a non-array object at {prefix}. "
            "All results should be numeric with associated units."
        )

    # If we have a Sed then we have a count of 1
    if isinstance(data, Sed):
        return 1
    return data.shape[0]


@lru_cache(maxsize=500)
def cached_split(split_key):
    """Split a key into a list of keys.

    This is a cached version of the split function to avoid repeated
    splitting of the same key.

    Args:
        split_key (str):
            The key to split in "key1/key2/.../keyN" format.

    Returns:
        list:
            A list of the split keys.
    """
    return split_key.split("/")


def combine_list_of_dicts(dicts):
    """Combine a list of dictionaries into a single dictionary.

    Args:
        dicts (list):
            A list of dictionaries to combine.

    Returns:
        dict:
            The combined dictionary.
    """

    def combine_values(values):
        # Combine values into a unyt_array
        return unyt_array(values)

    def recursive_merge(dict_list):
        if len(dict_list) == 0:
            return {}
        if not isinstance(dict_list[0], dict):
            # Base case: combine non-dict leaves
            return combine_values(dict_list)

        # Recursive case: merge dictionaries
        merged = {}
        keys = dict_list[0].keys()
        for key in keys:
            # Ensure all dictionaries have the same keys
            if not all(key in d for d in dict_list):
                raise ValueError(
                    f"Key '{key}' is missing in some dictionaries."
                )
            # Recurse for each key
            merged[key] = recursive_merge([d[key] for d in dict_list])
        return merged

    return recursive_merge(dicts)


def sum_dicts_recursive(dicts):
    """Sum a list of nested dictionaries with additive leaves.

    Args:
        dicts (list):
            A list of dictionaries or additive leaf values.

    Returns:
        dict or object:
            The recursively summed dictionary or leaf value.
    """
    values = [value for value in dicts if value is not None]
    if len(values) == 0:
        return {}

    if not isinstance(values[0], dict):
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    summed = {}
    keys = set()
    for value in values:
        keys.update(value.keys())

    for key in keys:
        summed[key] = sum_dicts_recursive(
            [value[key] for value in values if key in value]
        )

    return summed


def sanitise_hdf5_key_part(value):
    """Return a HDF5-safe string fragment for generated labels."""
    return (
        str(value).replace(".", "p").replace("/", "_per_").replace("\\", "_")
    )


def divide_dicts_recursive(data, divisors):
    """Divide nested dictionary leaves by matching nested divisors."""
    if isinstance(data, dict):
        return {
            key: divide_dicts_recursive(data[key], divisors[key])
            for key in data
            if key in divisors
        }
    return data / divisors


def unify_dict_structure_across_ranks(data, comm, root=0):
    """Recursively unify the structure of a dictionary across all ranks.

    This function will ensure that all ranks have the same structure in their
    dictionaries. This is necessary for writing out the data in parallel.

    Args:
        data (dict): The data to unify.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
    """
    # If we don't have a dict, just return the data straight away, theres no
    # need to check the structure
    if not isinstance(data, dict):
        return data

    # Ok, we have a dict. Before we get to the meat, lets make sure we have
    # the same structure on all ranks
    my_out_paths = discover_dict_structure(data)
    gathered_out_paths = comm.gather(my_out_paths, root=root)
    if comm.rank == root:
        unique_out_paths = set.union(*gathered_out_paths)
    else:
        unique_out_paths = None
    out_paths = comm.bcast(unique_out_paths, root=root)

    # Warn the user if the structure is different
    if len(out_paths) != len(my_out_paths):
        warn(
            "The structure of the data is different on different ranks. "
            "We'll unify the structure but something has gone awry."
        )

        # Ensure all ranks have the same structure
        for path in out_paths:
            d = data
            for k in path.split("/")[:-1]:
                d = d.setdefault(k, {})
            d.setdefault(path.split("/")[-1], unyt_array([], "dimensionless"))

    return data


def get_dataset_properties(data, comm, root=0):
    """Return the shapes, dtypes and units of all data arrays in a dictionary.

    Args:
        data (dict): The data to get the shapes of.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.

    Returns:
        dict: A dictionary of the shapes of all data arrays.
        dict: A dictionary of the dtypes of all data arrays.
        dict: A dictionary of the units of all data arrays.
    """
    # If we don't have a dict, just return the data straight away, theres no
    # need to check the structure
    if not isinstance(data, dict):
        return {"": data.shape}, {"": data.dtype}

    # Ok, we have a dict. Before we get to the meat, lets make sure we have
    # the same structure on all ranks
    my_out_paths = discover_dict_structure(data)
    gathered_out_paths = comm.gather(my_out_paths, root=root)
    if comm.rank == root:
        unique_out_paths = set.union(*gathered_out_paths)
    else:
        unique_out_paths = None
    out_paths = comm.bcast(unique_out_paths, root=root)

    # Create a dictionary to store the shapes and dtypes
    shapes = {}
    dtypes = {}
    units = {}

    # Loop over the paths and get the shapes
    for path in out_paths:
        d = data
        for k in path.split("/"):
            d = d[k]
        shapes[path] = d.shape
        dtypes[path] = d.dtype
        units[path] = str(d.units)

    return shapes, dtypes, units, out_paths


def get_full_memory(obj, seen=None):
    """Estimate memory usage of a Python object, including NumPy arrays.

    Args:
        obj: The object to inspect.
        seen: Set of seen object ids to avoid double-counting.

    Returns:
        int: Approximate size in bytes.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = 0

    # NumPy arrays — very important to check early
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)

    # Built-in container types
    elif isinstance(obj, Mapping):
        size += sys.getsizeof(obj)
        for k, v in obj.items():
            size += get_full_memory(k, seen)
            size += get_full_memory(v, seen)

    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sys.getsizeof(obj)
        for item in obj:
            size += get_full_memory(item, seen)

    # Objects with __dict__
    elif hasattr(obj, "__dict__"):
        size += sys.getsizeof(obj)
        size += get_full_memory(vars(obj), seen)

    # Objects with __slots__
    elif hasattr(obj, "__slots__"):
        size += sys.getsizeof(obj)
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += get_full_memory(getattr(obj, slot), seen)

    else:
        # Fallback: include basic object size
        size += sys.getsizeof(obj)

    return size


def validate_noise_unit_compatibility(instruments, expected_unit):
    """Validate that noise attributes have compatible units.

    This function checks that instruments with noise capabilities have
    depth and noise_maps attributes with units compatible with the expected
    unit for the image type (luminosity or flux).

    Note: depth can be specified as:
        - Plain float/dict of floats: apparent magnitudes (dimensionless,
          valid for both luminosity and flux images)
        - unyt_quantity/dict of unyt_quantity: flux/luminosity with units
          (must match image type)

    Args:
        instruments (list):
            A list of Instrument objects to validate.
        expected_unit (unyt.Unit):
            The expected unit for the image type (e.g., "erg/s/Hz" for
            luminosity images or "nJy" for flux images).

    Raises:
        InconsistentArguments:
            If an instrument has depth or noise_maps with incompatible units.
    """
    # Ensure expected_unit is a Unit object
    if not isinstance(expected_unit, Unit):
        expected_unit = Unit(expected_unit)

    for inst in instruments:
        if inst.can_do_noisy_imaging:
            # Check depth units if using SNR-based noise
            if inst.depth is not None:
                if isinstance(inst.depth, dict):
                    for filt, depth_val in inst.depth.items():
                        # Skip plain floats/ints (apparent magnitudes)
                        if isinstance(depth_val, (float, int)):
                            continue
                        # Validate unyt quantities
                        if isinstance(depth_val, unyt_quantity):
                            if not unit_is_compatible(
                                depth_val, expected_unit
                            ):
                                raise exceptions.InconsistentArguments(
                                    f"Depth units must be compatible with "
                                    f"{expected_unit}. Got {depth_val.units} "
                                    f"for filter {filt} in instrument "
                                    f"{inst.label}. Are you using a "
                                    "rest-frame or observed-frame instrument "
                                    "with the wrong image type?"
                                )
                        else:
                            raise exceptions.InconsistentArguments(
                                f"Depth must be a float (apparent magnitude) "
                                f"or unyt_quantity with units. Got "
                                f"{type(depth_val)} for filter {filt} in "
                                f"instrument {inst.label}."
                            )
                # Skip plain floats/ints (apparent magnitudes)
                elif isinstance(inst.depth, (float, int)):
                    pass  # Apparent magnitudes are valid for both types
                # Validate unyt quantities
                elif isinstance(inst.depth, unyt_quantity):
                    if not unit_is_compatible(inst.depth, expected_unit):
                        raise exceptions.InconsistentArguments(
                            f"Depth units must be compatible with "
                            f"{expected_unit}. Got {inst.depth.units} "
                            f"in instrument {inst.label}. Are you using a "
                            "rest-frame or observed-frame instrument with "
                            "the wrong image type?"
                        )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Depth must be a float (apparent magnitude) or "
                        f"unyt_quantity with units. Got {type(inst.depth)} "
                        f"in instrument {inst.label}."
                    )

            # Check noise_maps units if using noise maps
            if inst.noise_maps is not None:
                if isinstance(inst.noise_maps, dict):
                    for filt, noise_map in inst.noise_maps.items():
                        if isinstance(noise_map, unyt_array):
                            if not unit_is_compatible(
                                noise_map, expected_unit
                            ):
                                raise exceptions.InconsistentArguments(
                                    f"Noise map units must be compatible "
                                    f"with {expected_unit}. Got "
                                    f"{noise_map.units} for filter {filt} "
                                    f"in instrument {inst.label}. Are you "
                                    "using a rest-frame or observed-frame "
                                    "instrument with the wrong image type?"
                                )
                        else:
                            raise exceptions.InconsistentArguments(
                                f"Noise map must be a unyt_array with units. "
                                f"Got {type(noise_map)} for filter {filt} in "
                                f"instrument {inst.label}."
                            )
                elif isinstance(inst.noise_maps, unyt_array):
                    if not unit_is_compatible(inst.noise_maps, expected_unit):
                        raise exceptions.InconsistentArguments(
                            f"Noise map units must be compatible with "
                            f"{expected_unit}. Got "
                            f"{inst.noise_maps.units} in instrument "
                            f"{inst.label}. Are you using a rest-frame or "
                            "observed-frame instrument with the wrong image "
                            "type?"
                        )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Noise map must be a unyt_array with units. Got "
                        f"{type(inst.noise_maps)} in instrument {inst.label}."
                    )


class OperationKwargs:
    """A container class holding the kwargs needed by any pipeline operation.

    Attributes:
        _kwargs : dict
            The original kwargs dict used to build this object.
            (Values are not copied; we just hold the references.)
    """

    __slots__ = ("_kwargs", "_hash_key")

    def __init__(self, **kwargs):
        """Initialise the kwargs."""
        # Store the kwargs dict (no deep copies).
        self._kwargs = kwargs

        # Lazy cache of the structural key used for hashing/equality.
        self._hash_key = None

        # Convert any 'instruments' list to InstrumentCollection.
        self._convert_instruments_list()

    def _convert_instruments_list(self):
        """Convert any 'instruments' list to a InstrumentCollection."""
        # If we don't have instruments, nothing to do.
        if "instruments" not in self._kwargs:
            return

        # Convert list to InstrumentCollection if needed.
        inst_val = self._kwargs["instruments"]
        if isinstance(inst_val, list):
            self._kwargs["instruments"] = InstrumentCollection()
            self._kwargs["instruments"].add_instruments(*inst_val)
        elif not isinstance(inst_val, InstrumentCollection):
            raise exceptions.InconsistentArguments(
                "'instruments' kwarg must be a list of Instrument objects "
                "or an InstrumentCollection."
            )

    def __getitem__(self, key):
        """Dict-like access: obj['fov'] -> kwargs['fov']."""
        return self._kwargs[key]

    def __getattr__(self, name):
        """Attribute-style access: obj.fov -> kwargs['fov'].

        Called only if normal attribute lookup fails.
        """
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from None

    def get(self, key, default=None):
        """Dict-like get method: obj.get('fov', default) -> kwargs.get()."""
        return self._kwargs.get(key, default)

    @property
    def kwargs(self):
        """Return the underlying kwargs dict."""
        return self._kwargs

    def _build_hash_key(self):
        """Build a hashable structural key based on kwarg names and values.

        Rules
        -----
        - For each kwarg name:
            - If value is hashable, use ("val", value).
            - If value is unhashable (lists, arrays, etc.),
              use ("id", id(value)).

        This:
        - avoids any deep conversion or inspection of big objects,
        - deduplicates when all *references* are the same and hashables
          are equal.
        """
        items = []
        for name, value in self._kwargs.items():
            try:
                hash(value)
            except TypeError:
                # Unhashable => treat by identity only.
                items.append((name, ("id", id(value))))
            else:
                # Hashable => treat by value.
                items.append((name, ("val", value)))

        # Sort by kwarg name to make the key order-independent.
        items.sort(key=lambda kv: kv[0])
        return tuple(items)

    def get_hash(self):
        """Get the hash representation of the kwargs for caching purposes."""
        if self._hash_key is None:
            self._hash_key = self._build_hash_key()
        return hash(self._hash_key)

    def __hash__(self):
        """Return the hash of the kwargs for caching purposes."""
        return self.get_hash()

    def __eq__(self, other):
        """Check equality of two OperationKwargs based on their structure."""
        if not isinstance(other, OperationKwargs):
            return NotImplemented

        if self._hash_key is None:
            self._hash_key = self._build_hash_key()
        if other._hash_key is None:
            other._hash_key = other._build_hash_key()

        return self._hash_key == other._hash_key

    def __repr__(self):
        """Return a string representation of the OperationKwargs."""
        return f"{type(self).__name__}(kwargs={self._kwargs!r})"


class OperationKwargsHandler:
    """Container for Pipeline operation kwargs.

    This handler enables running pipeline operations multiple times
    with different parameters for different models in a clean,
    expandable and organized manner.

    Internally it stores unique OperationKwargs objects per operation
    (func_name) and associates them with one or more model labels and
    their instruments:

        self._func_map[func_name][OperationKwargs][label] -> list[instruments]

    This avoids duplicating identical kwargs sets across labels and
    provides a clean interface to loop over:

        - all (label, OperationKwargs) for a given operation, or
        - all OperationKwargs for a given (label, operation), or
        - groups of labels that share the same OperationKwargs.
    """

    def __init__(self, model_labels):
        """Initialise the OperationKwargsHandler.

        Args:
            model_labels (list or set of str):
                All the labels associated with the EmissionModel we
                are working on.
        """
        # Convert the input model_labels to a set for efficient lookup.
        self._allowed_models = set(model_labels)

        # Mapping:
        #   func_name -> {OperationKwargs -> list[model_label]}
        self._func_map = defaultdict(dict)
        #   func_name -> list[model_label]
        self._label_map = defaultdict(set)
        #   func_name -> OperationKwargs (for single-kwarg operations)
        # Note: Plain dict (not defaultdict) so missing keys return None
        # via .get()
        self._unique_func_map = {}

    def _check_model_label(self, model_label):
        """Validate that the provided model_label is allowed.

        Make sure the model_label exists in the allowed models for this
        handler, i.e. the label exists in the Pipeline's EmissionModel.

        Args:
            model_label (str):
                The model label to check.

        Raises:
            InconsistentArguments:
                If the model_label is not found in the allowed models.
        """
        if model_label in self._allowed_models:
            return

        raise exceptions.InconsistentArguments(
            f"Model label {model_label} not found in the Pipeline's "
            "EmissionModel."
        )

    def _normalize_labels(self, model_label):
        """Return a set of labels from the model_label argument.

        This helper exists to handle the various possible input types for
        model_label: None, str, or iterable of str.

        Args:
            model_label (str or iterable of str or None):
                Emission model label(s) or None for NO_MODEL_LABEL.

        Returns:
            set:
                A set of model labels.
        """
        if model_label is None or model_label == NO_MODEL_LABEL:
            return self._allowed_models
        if isinstance(model_label, str):
            return {model_label}
        # list / tuple / set
        return set(model_label)

    def add(self, model_label, func_name, **kwargs):
        """Add a kwargs set for a given func_name and one or more labels.

        This wraps the kwargs in an OperationKwargs and deduplicates them
        based on its hashing / equality semantics.

        Args:
            model_label (str or iterable of str or None):
                Emission model label(s) or None for NO_MODEL_LABEL.
            func_name (str):
                Operation / method name, e.g. "get_images_luminosity".
            **kwargs:
                Arbitrary keyword arguments to store for this func.

        Returns:
            OperationKwargs:
                The OperationKwargs instance representing this kwargs set.
        """
        labels = self._normalize_labels(model_label)
        for lab in labels:
            self._check_model_label(lab)

        # Create the kwargs object
        op_kwargs = OperationKwargs(**kwargs)

        # Link the operation kwargs into the internal mapping.
        # Get the per-function mapping (defaultdict creates empty dict if
        # missing)
        func_kwargs_map = self._func_map[func_name]

        # Get the existing label list for this op_kwargs (or create empty list)
        label_map = func_kwargs_map.get(op_kwargs, [])
        label_map.extend(labels)
        func_kwargs_map[op_kwargs] = label_map

        # Update the label map for this function.
        self._label_map[func_name].update(labels)

        return op_kwargs

    def add_unique(self, func_name, **kwargs):
        """Add a single unique kwargs set for a given func_name.

        This is used for operations that should only have one configuration
        per pipeline run (e.g., get_sfzh, get_sfh, get_observed_spectra).

        Args:
            func_name (str):
                Operation / method name, e.g. "get_sfzh".
            **kwargs:
                Arbitrary keyword arguments to store for this func.

        Returns:
            OperationKwargs:
                The OperationKwargs instance representing this kwargs set.
        """
        # Just exit if we already have an entry for this function.
        # We can get here multiple times if we recurse, so we just
        # return the existing one. If theres an issue with conflicting
        # kwargs, that should be caught elsewhere.
        if func_name in self._unique_func_map:
            return self._unique_func_map[func_name]

        # Create the kwargs object
        op_kwargs = OperationKwargs(**kwargs)

        # Store it in the unique func map.
        self._unique_func_map[func_name] = op_kwargs

        return op_kwargs

    def has(self, func_name, model_label=None):
        """Return True if any kwargs are stored for the given operation.

        Args:
            func_name (str):
                Operation / method name.
            model_label (str, optional):
                If provided, restrict the check to this model.
                If omitted, all models are searched.

        Returns:
            bool:
                True if at least one OperationKwargs exists matching the query.
        """
        # Do we actually have an entry for this function name?
        func_entries = self._func_map.get(
            func_name, self._unique_func_map.get(func_name, None)
        )
        if func_entries is None:
            return False

        # If no model_label is provided, any entry counts as a match by here
        if model_label is None:
            return True

        # Check for the specific model label.
        self._check_model_label(model_label)

        # Check in the label map first (most efficient)
        if func_name in self._label_map:
            return model_label in self._label_map[func_name]

        # func_entries for unique is just OperationKwargs object (not dict).
        if isinstance(func_entries, OperationKwargs):
            return True

        # Fallback to checking values in func_entries (which is a dict)
        for labels in func_entries.values():
            if model_label in labels:
                return True

        return False

    def __contains__(self, func_name):
        """Return True if any kwargs are stored for this operation.

        Allows syntax like:

            if "get_images_flux" in handler:
                ...
        """
        return self.has(func_name)

    def iter_all(self, func_name):
        """Iterate over (model_label, OperationKwargs) pairs for an operation.

        This is the main entry point for Pipeline methods that want to
        process all configs for a given operation, regardless of model.

        Non-consuming: internal state is left unchanged.

        Args:
            func_name (str):
                Operation / method name.

        Yields:
            (model_label, OperationKwargs):
                Tuples of model label and OperationKwargs object.
        """
        func_entries = self._func_map.get(func_name, {})
        for op_kwargs, label_map in func_entries.items():
            if not isinstance(label_map, (list, set)):
                label_map = [label_map]
            yield label_map, op_kwargs

    def __getitem__(self, func_name):
        """Return an iterator over (model_label, OperationKwargs).

        This allows syntax like:

            for model_label, op_kwargs in handler["get_images_flux"]:
                ...

        which is non-consuming.
        """
        return self.iter_all(func_name)

    def get_unique_kwargs(self, func_name):
        """Return the unique OperationKwargs for a given func_name.

        This is only applicable for operations added via add_unique() and
        can never have multiple variations.

        Args:
            func_name (str):
                Operation / method name.

        Returns:
            OperationKwargs:
                The unique OperationKwargs for this operation.
        """
        return self._unique_func_map.get(func_name, None)
