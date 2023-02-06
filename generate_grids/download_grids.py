import os
import gdown
import argparse
import synthesizer

grid_url = {}

grid_url[
    "bpass-v2.2.1-bin_chab-300"
] = "https://drive.google.com/file/d/135XvWyz06DgU0_uMzczB4_Azy4DC7bOH/view?usp=share_link"
grid_url[
    "bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2"
] = "https://drive.google.com/file/d/1iJTi0ciskqsV6kL5ObbRV4orQ0xnKqvI/view?usp=share_link"

filepath = os.path.abspath(synthesizer.__file__)

default_grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")


def print_available_grids():
    for grid in grid_url.keys():
        print(grid)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Install pre-computed Synthesizer grids."
    )

    parser.add_argument(
        "-grid",
        type=str,
        help="grid code (use -list_grids to find available grids)",
        default=None,
    )
    parser.add_argument(
        "-dir",
        "--directory",
        type=str,
        default=default_grid_dir,
        help="grid directory location, defaults to synthesizer install directory",
    )
    parser.add_argument("-list_grids", "--list_grids", action="store_true")
    args = parser.parse_args()

    if args.list_grids:
        print("-" * 50)
        print("AVAILABLE GRIDS:")
        print_available_grids()

    else:
        if args.grid:
            gdown.download(
                grid_url[args.grid],
                output=f"{args.directory}/{args.grid}.h5",
                quiet=False,
                fuzzy=True,
            )
        else:
            print("ERROR! No grid provided. Available grids:")
            print_available_grids()
