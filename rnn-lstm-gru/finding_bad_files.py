import pickle as pkl
import numpy as np
import os
from tqdm import tqdm


path = '/scratch/eliransc/CSV4_experiment'


files = os.listdir(path)


for file in tqdm(files):



    try:
        a, b = pkl.load(open(os.path.join(path, file)))
        print(os.path.join(path, file))
    except:
        pass
        #os.remove(os.path.join(path, file))