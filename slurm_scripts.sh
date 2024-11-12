#!/bin/bash
#SBATCH --job-name="150M_16gpus"
#SBATCH --constraint="gpu"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4   #   using 4 cores each. 
#SBATCH --time=24:00:00
#SBATCH -o /tmp/150M_%A_%a.out

conda activate TokenFormer

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12856
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Your hostfile creation script from above
bash ./write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE, you can customize any path. 
export DLTS_HOSTFILE=/tmp/hosts_$SLURM_JOBID

python3 deepy.py train.py ./configs/tokenformer/150M_train_pile.yml