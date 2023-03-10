#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=25G
#SBATCH -c 10
#SBATCH --time=24:00:00

python d4rl_train.py --env=HalfCheetah_default --seed 100 --desired-level 2500
