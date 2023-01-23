
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
for i in "cora" "ACM" "IMDB" "citeseer" "photos" "computers"
do
for j in "True" "False"
do
python -u pn2_main.py --dataSet "$i" --sampling_method="importance_sampling" --method="single" --disjoint_transductive_inductive "$j" > log.txt &
done
done
