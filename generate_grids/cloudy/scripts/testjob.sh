

index=5

# access line at index from input_names file
grid=$(sed "${index}q;d" grids.txt)

echo $grid
