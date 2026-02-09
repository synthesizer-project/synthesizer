"""Validate and compare precision across N pipeline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_h5_dataset(filepath: Path, dataset_path: str) -> np.ndarray | None:
    """Load dataset from HDF5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            if dataset_path in f:
                return np.array(f[dataset_path][()])
    except Exception:
        pass
    return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate and compare precision across N pipeline outputs"
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

    # Dataset paths to compare
    datasets = [
        ("spectra_stellar", "spectra"),
        ("photometry_total", "photometry"),
        ("imaging_total", "imaging"),
    ]

    overall_pass = True

    # Create comparison plots
    for dataset_name, plot_label in datasets:
        print(f"Comparing {dataset_name}...")

        # Load data from all files
        data_dict = {}
        for filepath, label in zip(args.inputs, labels):
            data = load_h5_dataset(filepath, dataset_name)
            if data is not None:
                data_dict[label] = data
            else:
                print(f"  Warning: {dataset_name} not found in {filepath}")

        if len(data_dict) < 2:
            print(f"  Skipping {dataset_name} (insufficient data)")
            continue

        # Compare all pairs
        first_label = list(data_dict.keys())[0]
        for label in list(data_dict.keys())[1:]:
            ref = data_dict[first_label]
            comp = data_dict[label]

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

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, data in data_dict.items():
            ax.plot(data.flat, label=label, alpha=0.7)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"{plot_label.capitalize()} Comparison")
        ax.legend()
        fig.tight_layout()

        plot_file = args.output_dir / f"{plot_label}_comparison.png"
        fig.savefig(plot_file, dpi=150)
        print(f"  ✓ Saved: {plot_file}")

    # Summary
    print("\n" + "=" * 60)
    if overall_pass:
        print("✅ ALL COMPARISONS PASSED")
    else:
        print("❌ SOME COMPARISONS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
