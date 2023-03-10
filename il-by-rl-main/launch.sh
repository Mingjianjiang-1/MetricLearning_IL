#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=25G
#SBATCH -c 10
#SBATCH --time=24:00:00

python train.py --env=walker --method=gaifo --ep 8 --seed 100
#python train.py --env=$1 --method=$2 --ep 8 --seed $3 --suffix halfgrav
#python train.py --env=$1 --method=$2 --ep 8 --seed $3 --suffix tentimesdensity
