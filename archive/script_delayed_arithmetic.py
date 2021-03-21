import sys
import torch.nn.functional as F
import numpy as np
import joblib
import matplotlib.pyplot as plt

from dt_utils import Timer
from plastic import PlasticRecall
from data_utils import generate_delayed_recall_data, recall_chance, generate_arithmetic_data_onehot
from neural_net_utils import random_weight_init, plot_train_perf

#%%
R = int(sys.argv[1])
training = sys.argv[2] #'standard' or 'infdat###' (where ### is 3-digit(!) number of epochs per dataset) 

#%%
np.random.seed(98765) #for reproducibility
d = 25
dims = [d+2, 50, d]
initW, initB = random_weight_init(dims, bias=True)
net = PlasticRecall(initW, initB, eta=.5, lam=.5,
                    name='PlasticArithmetic_{}_R={}'.format(training,R)) #UPDATE

#%%
with Timer(net.name):
    if training.startswith('infdat'): #Infinite data training regime (new dataset for each epoch)
        epochsPer = int(training[-3:])
        for epoch in range(200000/epochsPer):
            trainData = generate_arithmetic_data_onehot(T=2000, d=d, R=R, intertrial=0)
            net.fit(trainData, epochs=epochsPer, earlyStop=True)
    elif training == 'standard': #Standard training regime (train on fixed dataset)
        trainData = generate_arithmetic_data_onehot(T=2000, d=d, R=R, intertrial=0)
        net.fit(trainData, epochs=200000, earlyStop=True)              
        #%%
        joblib.dump(trainData, net.name+'_trainData.pkl')
    #%%
    else:
        raise ValueError('Invalid entry for training method')
#%%
joblib.dump(net, net.name+'.pkl')

