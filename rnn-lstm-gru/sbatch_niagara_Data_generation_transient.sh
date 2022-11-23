#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /home/d/dkrass/eliransc/RNN_queue/rnn-lstm-gru/time_dependent.py


