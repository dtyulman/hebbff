import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F

import sys; sys.path.append('/Users/danil/My')
from dt_utils import Timer

from data_utils import generate_recognition_data, data_to_tensor
from neural_net_utils import Sigmoid, CrossEntropy
from fastweights import FeedforwardNet, FastWeights
from fastweights_numpy import FeedforwardNet as FeedforwardNumpy
from fastweights_numpy import FastWeights as FastWeightsNumpy

#torch.set_default_tensor_type(torch.cuda.FloatTensor)


#%% Generate data
d = 10     #length of input vector
R = 1      #repeat interval
P = .5     #probability of repeat

print('Creating data, d={}, R={}, P={}'.format(d, R, P))
trainData = generate_recognition_data(T=10, d=d, R=R, P=P)
trainDataTensor = data_to_tensor(trainData)

trainChance = 1-np.sum([xy[1] for xy in trainData], dtype=np.float)/len(trainData)
print('Train_chance:{}'.format(trainChance))

#%%
dims = [d, 10, 1]                    #dimensions of the network layers
epochs = 1

npNet = FeedforwardNumpy(dims, f=Sigmoid, Loss=CrossEntropy)
initW, initB = npNet.W, npNet.B

#%%
fw = FastWeights(initW, initB, eta=0.5)
with Timer('FastWeights Torch'):
    fw.fit(trainDataTensor, epochs=10)

#%%
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#    print('Using device:', device)
#    print(torch.cuda.get_device_name(0))
#    print('Memory Usage:')
#    print('   Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#    print('   Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
#    
#    net = FeedforwardNet(initW, initB, f=nn.Sigmoid(), fOut=nn.Sigmoid())
#    net.to(device)
#    [t.to(device) for t in trainDataTensor.tensors]
#    
#    with Timer('Feedforward GPU'):
#        net.fit(trainDataTensor, epochs=epochs)
#
##%%
#net = FeedforwardNet(initW, initB, f=nn.Sigmoid(), fOut=nn.Sigmoid())
#with Timer('Feedforward Torch'):
#    net.fit(trainDataTensor, epochs=epochs)
#
##%%
#with Timer('Feedforward Numpy'):
#    npNet.adam(trainData, epochs=epochs)
#    
##%%
#npFw = FastWeightsNumpy(initW, initB, eta=0.5)
#with Timer('FastWeights Numpy'):
#    npFw.adam(trainData, epochs=epochs)
#   
##%%
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#    print('Using device:', device)
#    print(torch.cuda.get_device_name(0))
#    print('Memory Usage:')
#    print('   Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#    print('   Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
#    
#    fw = FastWeights(initW, initB, eta=0.5)
##    fw.to(device)
##    trainDataTensor.to('cuda')
#    with Timer('FastWeights GPU'):
#        fw.fit(trainDataTensor, epochs=epochs)    

    
    