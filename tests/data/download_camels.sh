#!/bin/bash
dir=./
# $1
snap=031
# $2
lh=1
# $3

wget "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/LH/LH_${lh}/snap_$snap.hdf5" --output-document "${dir}/LH_${lh}_snap_${snap}.hdf5"

wget "https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/LH/LH_${lh}/fof_subhalo_tab_$snap.hdf5" --output-document "$dir/LH_${lh}_fof_subhalo_tab_$snap.hdf5"
