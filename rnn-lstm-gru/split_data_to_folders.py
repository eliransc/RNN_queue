import numpy as np
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastai.vision.all import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
from fastai.vision.all import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import shutil



def main():


    batch_size = 3200
    path = '/scratch/eliransc/new_gt_g_1_trans1'
    files = os.listdir(path)
    num_batches = int(len(files) / batch_size)
    counter = 1
    for curr_batch in tqdm(range(num_batches)):
        new_dst_folder = '/scratch/eliransc/all_new_g_g_1_trans1/folder_' + str(counter)
        if not os.path.exists(new_dst_folder):
            os.mkdir(new_dst_folder)
        for ind in range(curr_batch * batch_size, (curr_batch + 1) * batch_size):
            src = os.path.join(path, files[ind])
            dst = os.path.join(new_dst_folder, files[ind])
            shutil.move(src, dst)
            # print(src, dst)
        counter += 1



if __name__ == '__main__':
    main()