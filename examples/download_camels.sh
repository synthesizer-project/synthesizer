#!/bin/bash

dir=$1
snap=$2
lh=$3

wget "https://users.flatironinstitute.org/~camels/Sims/SIMBA/LH_${lh}/snap_$snap.hdf5" --output-document "${dir}/LH_${lh}_snap_${snap}.hdf5"

wget "https://users.flatironinstitute.org/~camels/Sims/SIMBA/LH_${lh}/fof_subhalo_tab_$snap.hdf5" --output-document "$dir/LH_${lh}_fof_subhalo_tab_$snap.hdf5"

