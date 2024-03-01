#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --time=40-00:00:00
#SBATCH --job-name=getty_1
#SBATCH -o outfile_test_1.out
#SBATCH -e errfile_test_1.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

source ~/venvs/two_step_regression_venv/bin/activate

python ~/projects/two_step_regression/main.py
