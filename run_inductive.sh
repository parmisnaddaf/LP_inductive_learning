#!/bin/sh

declare -a dataset_list=("IMDB", "cora")

for i in "${dataset_list[@]}"
do
  python pn2_main.py --dataSet "$i" 
done

