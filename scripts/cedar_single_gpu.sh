#!/bin/bash
#SBATCH --gres=gpu:1             
#SBATCH --ntasks-per-node=6
#SBATCH --mem=120G
#SBATCH --time=10:03:00
#SBATCH --account=def-vshwartz
#SBATCH --output=captions.out
./caption_inference.sh

