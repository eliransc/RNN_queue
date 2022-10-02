#!/bin/bash
#SBATCH -t 0-10:58
#SBATCH -A def-dkrass
#SBATCH --mem 30000
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/Lindley_GG1.py


