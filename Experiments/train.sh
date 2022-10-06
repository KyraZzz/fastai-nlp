#!/bin/bash
#SBATCH -A MULLINS-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

module load python/3.7 cuda/11.2 cudnn/8.1_cuda-11.2
source /home/yz709/fastainlp/venv/bin/activate

cd /home/yz709/fastainlp/fastai-nlp
python ./text_transfer_learning.py