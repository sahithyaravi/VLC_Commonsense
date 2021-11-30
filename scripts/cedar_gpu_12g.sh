#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100l:1
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=10:03:00
#SBATCH --account=def-vshwartz
#SBATCH --output=captions.out
./caption_inference.sh

