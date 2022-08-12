

synthesizer_data=/Users/stephenwilkins/Dropbox/Research/data/synthesizer
mkdir $synthesizer_data/input_files
mkdir $synthesizer_data/grids

## bc03 original
wget http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_2000_chabrier_imf.tar.gz
tar zxvf bc03.models.padova_2000_chabrier_imf.tar.gz -C $synthesizer_data/input_files
rm bc03.models.padova_2000_chabrier_imf.tar.gz

parent=$synthesizer_data/input_files/bc03/models/Padova2000/chabrier

for f in $parent/bc2003_hr_m*_chab_ssp.ised_ASCII.gz; do
  STEM=$(basename "${f}" .gz)
  # echo $f $parent/$STEM
  gunzip -c "${f}" > "$parent/${STEM}"
done

python3 grid_bc03.py $synthesizer_data
