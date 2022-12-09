#!/bin/sh

declare -a dataset_list=("IMDB")

source /localhome/pnaddaf/anaconda3/etc/profile.d/conda.sh
conda activate env

for i in "${dataset_list[@]}"
do
  python pn2_main.py --dataSet "$i" >> "results/importance sampling basic/IS_single_$i"
done
