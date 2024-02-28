#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user an963984@ucf.edu

#SBATCH --output=pytorch-BLIP2_Eval-slurm-%J.out
#SBATCH --job-name=BLIP2_Eval

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load gasp/gasp-miniconda-4.3.30
#module load cuda/cuda-12.1

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of PyTorch
# Modify PATH to use the Python interpreter from the lavis_dev Conda environment
export PATH=/home/aelkommos/.conda/envs/lavis_dev/bin:$PATH

# Check which Python will be used
which python
python --version


# Run PyTorch Training
echo "BLIP2 Eval Start:"


echo "Running BLIP2 Retrieval on 6.7B model"
time python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval.yaml


echo "Running BLIP2 Coco Caption on 6.7b model"
time python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_opt6.7b_eval.yaml


echo "Running BLIP2 VQAv2 on 6.7B model"
time python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_opt_eval.yaml"

# You're done!
echo "Ending script..."
date