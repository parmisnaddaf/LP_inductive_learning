
#!/bin/bash


for i in "single" "multi" 
do
for j in "cora" "ACM" "IMDB" "citeseer"
do
for k in "monte" 
do
python pn2_main.py --method "$i" --dataSet "$j" --sampling_method "$k"
done
done
done