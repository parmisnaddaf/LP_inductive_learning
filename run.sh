
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
for i in "ACM"
do
for j in "True" "False"
do
python pn2_main.py --dataSet "$i" --disjoint_transductive_inductive "$j"
done
done
