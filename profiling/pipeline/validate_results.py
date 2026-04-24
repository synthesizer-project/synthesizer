"""Validate and compare numerical precision across Pipeline HDF5 outputs.

This script compares datasets from multiple Pipeline.write() outputs to
validate numerical precision. It uses the structured HDF5 layout that
Pipeline actually produces (e.g. Galaxies/Spectra/..., Galaxies/Stars/...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_h5_dataset(filepath: Path, dataset_path: str) -> np.ndarray | None:
    """Load dataset from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        dataset_path: Full path to dataset within file

    Returns:
        Array if found, None otherwise
    """
    try:
        with h5py.File(filepath, "r") as f:
            if dataset_path in f:
                return np.array(f[dataset_path][()])
    except Exception:
        pass
    return None


def discover_datasets(filepath: Path) -> list[str]:
    """Discover all dataset paths in an HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        List of full dataset paths
    """
    dataset_paths = []

    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_paths.append(name)

    try:
        with h5py.File(filepath, "r") as f:
            f.visititems(visit_func)
    except Exception:
        pass

    return dataset_paths


def main() -> None:
    """Main entry point for the precision validation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare precision across N Pipeline HDF5 outputs"
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input HDF5 files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help="Labels for each run (default: use filenames)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Output directory for summary (default: current directory)",
    )
    parser.add_argument(
        "--tolerance",
        choices=["default", "loose", "tight"],
        default="default",
        help="Tolerance level for comparisons",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help=(
            "Specific dataset paths to compare "
            "(default: auto-discover from first file)"
        ),
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set tolerances
    tolerances = {
        "default": (
            1e-5,
            1e-7,
        ),  # (rtol, atol) - relaxed for typical numerical differences
        "loose": (1e-4, 1e-6),
        "tight": (1e-7, 1e-9),
    }
    rtol, atol = tolerances[args.tolerance]

    # Setup labels
    labels = []
    for i, input_file in enumerate(args.inputs):
        if args.labels and i < len(args.labels):
            labels.append(args.labels[i])
        else:
            labels.append(input_file.stem)

    print(
        f"Comparing {len(args.inputs)} outputs with {args.tolerance} tolerance"
    )
    print(f"Labels: {labels}\n")

    # Discover dataset paths if not provided
    if args.datasets:
        dataset_paths = args.datasets
    else:
        print(f"Auto-discovering datasets from {args.inputs[0]}...")
        dataset_paths = discover_datasets(args.inputs[0])
        # Filter to likely numerical datasets (exclude metadata)
        dataset_paths = [
            p
            for p in dataset_paths
            if not any(
                excl in p.lower()
                for excl in ["metadata", "units", "labels", "ids", "names"]
            )
        ]
        print(f"Found {len(dataset_paths)} datasets to compare\n")

    overall_pass = True
    compared_count = 0
    skipped_non_numeric = 0
    skipped_missing = 0
    pair_pass_count = 0
    pair_fail_count = 0
    shape_fail_count = 0
    failures: list[dict[str, object]] = []

    # Compare each dataset
    for dataset_path in dataset_paths:
        # Load data from all files
        data_dict = {}
        for filepath, label in zip(args.inputs, labels):
            data = load_h5_dataset(filepath, dataset_path)
            if data is not None:
                data_dict[label] = data

        if len(data_dict) < 2:
            print(
                f"⊗ Skipping {dataset_path} "
                f"(found in {len(data_dict)}/{len(labels)} files)"
            )
            skipped_missing += 1
            continue

        print(f"Comparing {dataset_path}...")
        compared_count += 1

        # Compare all pairs
        first_label = list(data_dict.keys())[0]
        for label in list(data_dict.keys())[1:]:
            ref = data_dict[first_label]
            comp = data_dict[label]

            # Skip non-numeric datasets (e.g. strings/labels/metadata).
            if not (
                np.issubdtype(ref.dtype, np.number)
                and np.issubdtype(comp.dtype, np.number)
            ):
                print(
                    f"  ⊗ Skipping {first_label} vs {label}: "
                    f"non-numeric dtype ({ref.dtype} vs {comp.dtype})"
                )
                skipped_non_numeric += 1
                continue

            # Ensure shapes match
            if ref.shape != comp.shape:
                print(
                    f"  ✗ FAIL {first_label} vs {label}: "
                    f"shape mismatch ({ref.shape} vs {comp.shape})"
                )
                overall_pass = False
                shape_fail_count += 1
                pair_fail_count += 1
                failures.append(
                    {
                        "dataset": dataset_path,
                        "pair": f"{first_label} vs {label}",
                        "reason": "shape mismatch",
                        "shape_ref": ref.shape,
                        "shape_comp": comp.shape,
                    }
                )
                continue

            max_diff = np.max(np.abs(ref - comp))
            mean_diff = np.mean(np.abs(ref - comp))

            tolerance_ok = np.allclose(ref, comp, rtol=rtol, atol=atol)

            if tolerance_ok:
                print(
                    f"  ✓ PASS {first_label} vs {label}: "
                    f"max={max_diff:.2e}, mean={mean_diff:.2e}"
                )
                pair_pass_count += 1
            else:
                print(
                    f"  ✗ FAIL {first_label} vs {label}: "
                    f"max={max_diff:.2e}, mean={mean_diff:.2e}"
                )
                overall_pass = False
                pair_fail_count += 1
                rel_denom = np.maximum(np.abs(comp), atol)
                max_rel_diff = float(np.max(np.abs(ref - comp) / rel_denom))
                failures.append(
                    {
                        "dataset": dataset_path,
                        "pair": f"{first_label} vs {label}",
                        "reason": "value mismatch",
                        "max_diff": float(max_diff),
                        "mean_diff": float(mean_diff),
                        "max_rel_diff": max_rel_diff,
                    }
                )

    # Summary
    print("\n" + "=" * 60)
    print(f"Compared {compared_count} datasets")
    print(f"Pair comparisons passed: {pair_pass_count}")
    print(f"Pair comparisons failed: {pair_fail_count}")
    print(f"Shape mismatches: {shape_fail_count}")
    print(f"Skipped missing datasets: {skipped_missing}")
    print(f"Skipped non-numeric comparisons: {skipped_non_numeric}")
    if overall_pass:
        print("✅ ALL COMPARISONS PASSED")
    else:
        print("❌ SOME COMPARISONS FAILED")
    print("=" * 60)

    summary_file = args.output_dir / "validation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Validation Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Inputs: {', '.join(str(p) for p in args.inputs)}\n")
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(
            "Tolerance preset: "
            f"{args.tolerance} (rtol={rtol}, atol={atol})\n\n"
        )
        f.write(f"Compared datasets: {compared_count}\n")
        f.write(f"Pair comparisons passed: {pair_pass_count}\n")
        f.write(f"Pair comparisons failed: {pair_fail_count}\n")
        f.write(f"Shape mismatches: {shape_fail_count}\n")
        f.write(f"Skipped missing datasets: {skipped_missing}\n")
        f.write(f"Skipped non-numeric comparisons: {skipped_non_numeric}\n")
        f.write(f"Overall status: {'PASS' if overall_pass else 'FAIL'}\n\n")

        if failures:
            failures_sorted = sorted(
                failures,
                key=lambda x: float(x.get("max_rel_diff", 0.0)),
                reverse=True,
            )
            f.write("Top failures by max relative difference:\n")
            for item in failures_sorted[:20]:
                if item["reason"] == "shape mismatch":
                    f.write(
                        f"- {item['dataset']} [{item['pair']}]: "
                        f"shape mismatch ({item['shape_ref']} vs "
                        f"{item['shape_comp']})\n"
                    )
                else:
                    f.write(
                        f"- {item['dataset']} [{item['pair']}]: "
                        f"max_diff={item['max_diff']:.3e}, "
                        f"mean_diff={item['mean_diff']:.3e}, "
                        f"max_rel_diff={item['max_rel_diff']:.3e}\n"
                    )

    print(f"✓ Saved: {summary_file}")


if __name__ == "__main__":
    main()
