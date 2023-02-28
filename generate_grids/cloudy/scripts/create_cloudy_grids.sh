#!/bin/bash

# create a cloudy grid using the default assumptions

# synthesizer_dir="/Users/stephenwilkins/Dropbox/Research/data/synthesizer/"
synthesizer_dir="/research/astrodata/highz/synthesizer/" # apollo
machine="apollo"
c=$CLOUDY17

cd ..
while IFS="" read -r p || [ -n "$p" ]
do
  arrIN=(${p// / })
  sps=${arrIN[0]}
  params=${arrIN[1]}
  printf '%s\n' "$sps"
  printf '%s\n' "$params"
  python3 make_cloudy_input_grid.py -dir $synthesizer_dir -m $machine -sps $sps  -p $params.yaml  -c $c
done < scripts/spsparams.txt
