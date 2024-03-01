#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40-00:00:42
#SBATCH --job-name=jupyter
#SBATCH -o outfile_jupyter.out
#SBATCH -e errfile_jupyter.out
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=64G

source ~/venvs/getty_images_venv/bin/activate

jupyter lab --port=5656