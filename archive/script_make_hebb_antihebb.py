import sys
import numpy as np
import torch
import torch.nn as nn
import joblib

from dt_utils import Timer

from data_utils import generate_recognition_data
from plastic import PlasticNet
from neural_net_utils import random_weight_init

d = 50           #length of input vector
R = int(sys.argv[1])            #repeat interval
P = .5           #probability of repeat

h = int(sys.argv[3])

dims = [d, h, 1] #dimensions of the network layers

np.random.seed(12345) #for reproducibility
initW, initB = random_weight_init(dims, bias=True)

#%%
netType = sys.argv[2]
if netType == 'hebb':
    etaInit = 0.1
elif netType == 'anti':
    etaInit = -0.1
else:
    raise Exception()
#%%
filename = 'PlasticNet_R={}_{}_h={}.pkl'.format(R, netType, h)  
net = PlasticNet(initW, initB, eta=etaInit, lam=0.9, positiveEta=False)   

with Timer(filename):
    for epoch in range(200000):
        trainData = generate_recognition_data(T=2000, d=d, R=R, P=P, interleave=True, astensor=True)
        net.fit(trainData, epochs=1)
        if sum(net.hist['train_acc'][-5:]) >= 4.99:
            break #early stop
joblib.dump(net, filename)






