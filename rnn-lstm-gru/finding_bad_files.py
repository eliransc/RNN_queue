import pickle as pkl
import numpy as np
import os
from tqdm import tqdm


path = '/scratch/eliransc/new_special_test'


files = os.listdir(path)


for file in tqdm(files):

    try:
        a, b = pkl.load(open(os.path.join(path, file),'rb'))
        # print(os.path.join(path, file))

    except:
        print('bad')
        os.remove(os.path.join(path, file))