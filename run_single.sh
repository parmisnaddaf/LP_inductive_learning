
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
for i in "cora" "ACM" "IMDB" "citeseer" "photos" "computers"
do
for j in "True" "False"
do
python temp_main.py --dataSet "$i" --disjoint_transductive_inductive "$j"
done
done