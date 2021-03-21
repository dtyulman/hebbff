import argparse
import numpy as np
import torch
import joblib

from networks import HebbAugRecog
from data import generate_aug_recog_data_batch

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--R', type=int)
parser.add_argument('--T', type=int)
parser.add_argument('--batchSize', type=int)
parser.add_argument('--dataSize', type=int)
parser.add_argument('--iters', type=int)
parser.add_argument('--c', type=float)
parser.add_argument('--useBCE', action='store_const', const=True)
args = parser.parse_args()

#data params
R = 1 if args.R is None else args.R #repeat interval
P = 0.5  #repeat probability
T = 500 if args.T is None else args.T #timepoints per sample
d = 25 #length of item
k = 1 #length of augment
B = 1 if args.batchSize is None else args.batchSize #samples per batch
intlv = True #TODO: need to implement in generate_*_batch if want False
cacheSize = 64
useBCE = True if args.useBCE is None else args.useBCE
gen_data = lambda: generate_aug_recog_data_batch(T=T, batchSize=cacheSize, d=d, k=k, R=R, P=P, 
                                                 interleave=intlv, zeroOneVal=useBCE)

#network params
Nh = 50 #number of hidden units
dims = [d+k, Nh, 1+k]
if useBCE:
    c = None
else:
    c = 0.5 if args.c is None else args.c

   
#%% 
net = HebbAugRecog(dims, c=c)

#%%
fname = '{name}_R={R}_c={c}.pkl'.format(name=net.name, c=c, R=R)
folder = 'HebbAugRecog_09-05'
iters = 2**20 if args.iters is None else args.iters
net.fit('infinite', gen_data, iters=iters, batchSize=B, filename=fname, folder=folder)
     

