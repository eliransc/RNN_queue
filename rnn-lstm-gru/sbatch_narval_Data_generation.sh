#!/bin/bash
#SBATCH -t 0-10:58
#SBATCH -A def-dkrass
#SBATCH --mem 30000
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/RNN_queue/rnn-lstm-gru/RNN_data_generating.py


