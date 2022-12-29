#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10000M
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/RNN_queue/rnn-lstm-gru/train_GtGt1.py

