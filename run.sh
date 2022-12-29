
#!/bin/bash




for j in "cora" "ACM" "IMDB" "citeseer"
do
for k in "monte" 
do
python pn2_main.py --dataSet "$j" --sampling_method "$k"
done
done
