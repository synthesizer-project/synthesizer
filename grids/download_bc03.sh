stelib=stelib
imf=chabrier

## bc03-2016
# wget http://www.bruzual.org/bc03/Updated_version_2016/BC03_miles_chabrier.tgz
# wget http://www.bruzual.org/bc03/Updated_version_2016/BC03_${stelib}_${imf}.tgz
# tar zxvf BC03_${stelib}_${imf}.tgz -C input_files/bc03
# rm BC03_${stelib}_${imf}.tgz

## bc03 original
wget http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_2000_chabrier_imf.tar.gz
tar zxvf bc03.models.padova_2000_chabrier_imf.tar.gz -C input_files/bc03
rm bc03.models.padova_2000_chabrier_imf.tar.gz
# for f in input_files/bc03/bc03/models/Padova2000/chabrier/;
#     do gzip -d f;
# done
cd input_files/bc03/bc03/models/Padova2000/chabrier
for f in bc2003_hr_m*_chab_ssp.ised_ASCII.gz; do
  STEM=$(basename "${f}" .gz)
  gunzip -c "${f}" > ../"${STEM}"
done
