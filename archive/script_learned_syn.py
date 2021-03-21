import argparse
import numpy as np

from dt_utils import Timer
from hebbian import LearnedSynHebb
from data_utils import generate_recognition_data

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--R', type=int)
parser.add_argument('--T', type=int)
parser.add_argument('--batchsize', type=int)
parser.add_argument('--train')
parser.add_argument('--epochs', type=int)
parser.add_argument('--Ns', type=int)
args = parser.parse_args()

#data params
R = 10 if args.R is None else args.R #repeat interval
T = 2000 if args.T is None else args.T #samples per trial
P = 0.5  #repeat probability
d = 25 #length of item
k = 1 #length of augment
intlv = True 

#training params
train = 'inf1' if args.train is None else args.train #training method

#network params
Nh = 50 #number of hidden units
dims = [d, Nh, 1]
Ns = 5 if args.Ns is None else args.Ns #number of hidden units in synapse MLP

#%% 
net = LearnedSynHebb(dims, lam=0.9, Ns=Ns)

#%%
epochs = 300000 if args.epochs is None else args.epochs

fname = '{}_R={}_Ns={}.pkl'.format(net.name, R, Ns) 
lastSave = -1 if any([vars(args)[i] for i in vars(args).keys()]) else np.inf
with Timer(fname) as timer: 
    if train.lower().startswith('inf'):
        perBatch = int(train[3:])
        for _ in range(epochs/perBatch):
            trainData = generate_recognition_data(T=T, d=d, R=R, P=P, interleave=intlv)
            net.fit(trainData, epochs=perBatch)
            fname, lastSave = net._autosave(fname, lastSave)
    else:
        raise ValueError("Training must be 'infN' or 'std'")

net.hist['time'] = timer.elapsed
net.save( fname )






