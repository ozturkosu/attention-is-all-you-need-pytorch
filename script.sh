#!/bin/bash
#SBATCH --time=08:59:00
#SBATCH --ntask-per-node=1
#SBATCH --partition=batch-bdw-v100


cd /home/ozturk.27/ModelEfficiency/VirtualEnv
. eminEnv/bin/activate
cd /home/ozturk.27/ModelEfficiency/attention-is-all-you-need-pytorch

echo "==========================================="
echo "            Running                        "
echo "==========================================="




python  train.py -data_pkl m30k_deen_shr.pkl  -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b $1 -warmup $2 -epoch $3 -lr_mul $4 -factorized_k $5  >> $6.txt
