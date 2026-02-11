"""Validate and compare precision across N Pipeline HDF5 outputs.

This script compares datasets from multiple Pipeline.write() outputs to
validate numerical precision. It uses the structured HDF5 layout that
Pipeline actually produces (e.g. Galaxies/Spectra/..., Galaxies/Stars/...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
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
    """Main entry point."""
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
        default=Path("profiling/outputs/precision_validation"),
        help="Output directory for plots",
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
        "default": (1e-6, 1e-8),  # (rtol, atol)
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
            continue

        print(f"Comparing {dataset_path}...")
        compared_count += 1

        # Compare all pairs
        first_label = list(data_dict.keys())[0]
        for label in list(data_dict.keys())[1:]:
            ref = data_dict[first_label]
            comp = data_dict[label]

            # Ensure shapes match
            if ref.shape != comp.shape:
                print(
                    f"  ✗ FAIL {first_label} vs {label}: "
                    f"shape mismatch ({ref.shape} vs {comp.shape})"
                )
                overall_pass = False
                continue

            max_diff = np.max(np.abs(ref - comp))
            mean_diff = np.mean(np.abs(ref - comp))

            tolerance_ok = np.allclose(ref, comp, rtol=rtol, atol=atol)

            if tolerance_ok:
                print(
                    f"  ✓ PASS {first_label} vs {label}: "
                    f"max={max_diff:.2e}, mean={mean_diff:.2e}"
                )
            else:
                print(
                    f"  ✗ FAIL {first_label} vs {label}: "
                    f"max={max_diff:.2e}, mean={mean_diff:.2e}"
                )
                overall_pass = False

        # Create plot (limit to smaller datasets for clarity)
        if ref.size <= 10000:
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, data in data_dict.items():
                ax.plot(data.flat, label=label, alpha=0.7)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            # Use dataset path as title (shorten if needed)
            title = dataset_path.replace("/", " / ")
            if len(title) > 60:
                title = "..." + title[-57:]
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()

            # Safe filename from dataset path
            safe_name = dataset_path.replace("/", "_").replace(" ", "_")
            plot_file = args.output_dir / f"{safe_name}.png"
            fig.savefig(plot_file, dpi=150)
            plt.close(fig)
            print(f"  ✓ Saved: {plot_file}")
        else:
            print(f"  (Skipped plot: dataset too large, {ref.size} elements)")

    # Summary
    print("\n" + "=" * 60)
    print(f"Compared {compared_count} datasets")
    if overall_pass:
        print("✅ ALL COMPARISONS PASSED")
    else:
        print("❌ SOME COMPARISONS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
