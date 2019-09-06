#!/bin/bash
#SBATCH --account=def-hindle
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --cpus-per-task 8
#SBATCH --mem=124G
#SBATCH --time=2:00:00
#SBATCH --output=%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=awwong1@ualberta.ca

module load gcc cuda cudnn opencv/3.4.3
nvidia-smi
source venv/bin/activate

python3 main.py configs/cifar100/classification_vgg19_bn.json
