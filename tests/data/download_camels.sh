#!/bin/bash
dir=./
# $1
snap=086
# $2
cv=0
# $3

wget "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_${cv}/snapshot_$snap.hdf5" --output-document "${dir}/CV_${cv}_snap_${snap}.hdf5"

wget "https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/CV/CV_${cv}/groups_$snap.hdf5" --output-document "$dir/CV_${cv}_fof_subhalo_tab_$snap.hdf5"
