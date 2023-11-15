import pickle as pkl
import numpy as np
import os
from tqdm import tqdm


path = '/scratch/eliransc/CSV4_experiment'


files = os.listdir(path)


for file in files:
    try:
        a, b = pkl.load(open(os.path.join(path, file)))
    except:
        print(file)
        os.remove(os.path.join(path, file))