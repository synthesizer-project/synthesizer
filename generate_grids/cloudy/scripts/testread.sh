let index=2

# access line at index from input_names file
p=$(sed "${index}q;d" spsparams.dat)
arrIN=(${p// / })
sps=${arrIN[0]}
params=${arrIN[1]}
echo $sps
echo $params
