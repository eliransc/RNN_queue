import os
import pickle as pkl
from tqdm import tqdm

path = '/scratch/eliransc/ph_random/large_ph'

files = os.listdir(path)

new_path = '/scratch/eliransc/ph_random/large_ph_one_in_pkl'

for file in tqdm(files):
    dd = pkl.load(open(os.path.join(path, file), 'rb'))
    for ind, dist in enumerate(dd):
        pkl.dump(dist, open(os.path.join(new_path, file[:-4] + '_' + str(ind)+'.pkl'), 'wb'))

