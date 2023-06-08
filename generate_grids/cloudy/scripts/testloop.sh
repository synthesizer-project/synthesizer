

# # access line at index from input_names file
# p=$(sed "${index}q;d" spsparams.dat)
# arrIN=(${p// / })
# sps=${arrIN[0]}
# params=${arrIN[1]}
# echo $sps
# echo $params


while IFS="" read -r p || [ -n "$p" ]
do
  arrIN=(${p// / })
  sps=${arrIN[0]}
  params=${arrIN[1]}
  printf '%s\n' "$sps"
  printf '%s\n' "$params"
done < spsparams.txt
