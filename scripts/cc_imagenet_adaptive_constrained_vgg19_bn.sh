#!/bin/bash
#SBATCH --account=def-hindle
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:8
#SBATCH --exclusive
#SBATCH --cpus-per-task 28
#SBATCH --mem=0
#SBATCH --time=1-00:00
#SBATCH --output=%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=awwong1@ualberta.ca

module load arch/avx512 StdEnv/2018.3
nvidia-smi
source venv/bin/activate

curdir=$PWD
dataset_dir=$SLURM_TMPDIR/imagenet/

echo "dataset directory: ${dataset_dir}".

# prepare the data
mkdir -p $dataset_dir
cd $dataset_dir
echo "Copying imagenet folders..."
time tar xf /scratch/awwong1/datasets/imagenet/imagenet_raw_train.tar .
time tar xf /scratch/awwong1/datasets/imagenet/imagenet_raw_val.tar .

# change back to workspace directory
cd $curdir

OVERRIDE="{\"gpu_ids\":[0],\"train_data\":{\"args\":[\"$dataset_dir\"]},\"eval_data\":{\"args\":[\"$dataset_dir\"]}}"
python3 main.py configs/imagenet/adaptive_constrained_vgg19_bn.json --override ${OVERRIDE}
