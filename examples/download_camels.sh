#!/bin/bash

dir=$1
snap=$2

wget "https://users.flatironinstitute.org/~camels/Sims/SIMBA/LH_0/snap_$snap.hdf5" --output-document "${dir}/snap_${snap}.hdf5"

wget "https://users.flatironinstitute.org/~camels/Sims/SIMBA/LH_0/fof_subhalo_tab_$snap.hdf5" --output-document "$dir/fof_subhalo_tab_$snap.hdf5"

