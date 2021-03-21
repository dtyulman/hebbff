import sys
import torch.nn.functional as F
import numpy as np
import joblib

from dt_utils import Timer
from plastic import PlasticRecall
from data_utils import generate_delayed_recall_data, recall_chance
from neural_net_utils import random_weight_init, plot_train_perf

#%%
if len(sys.argv)>=4:
    R = int(sys.argv[1]) 
    training = sys.argv[2] #'standard' or 'infdat###' (where ### is 3-digit(!) number of epochs per dataset) 
    useNull = True if sys.argv[3].lower()=='true' else False
    print '[Script] Parsed vars:', R, training, useNull
else:
    R = 1 #UPDATE
    training = 'standard' #UPDATE
    useNull=False #UPDATE

#%%
np.random.seed(98765) #for reproducibility
d = 25
dims = [d+2, 50, d]
initW, initB = random_weight_init(dims, bias=True)

if len(sys.argv)>=5:
    print('[Script] Loading pre-trained net {} ...'.format(sys.argv[4]))
    net = joblib.load(sys.argv[4])
else:    
    net = PlasticRecall(initW, initB, eta=.5, lam=.5,
                    name='PlasticRecall_{}_R={}{}'.format(training,R,'_useNull' if useNull else '')) #UPDATE

#%%
T=2000
maxEpochs = 200000

with Timer(net.name):
    if training.startswith('infdat'): #Infinite data training regime (new dataset for each epoch)
        epochsPer = int(training[-3:])
        for epoch in range(maxEpochs/epochsPer):
            trainData = generate_delayed_recall_data(T=T, d=d, R=R, intertrial=0, useNull=useNull)
            net.fit(trainData, epochs=epochsPer, earlyStop=True) #TODO: FIX: this doesn't actually stop early bc of outer loop...
    elif training == 'standard': #Standard training regime (train on fixed dataset)
        if len(sys.argv)>=5:
            print('[Script] Loading data {} ...'.format(sys.argv[5]))
            trainData = joblib.load(sys.argv[5])
        else:
            trainData = generate_delayed_recall_data(T=T, d=d, R=R, intertrial=0, useNull=useNull)    
        net.fit(trainData, epochs=maxEpochs, earlyStop=True)              
        #%%
        joblib.dump(trainData, net.name+'_trainData.pkl')
    #%%
    else:
        raise ValueError('Invalid entry for training method')
#%%
joblib.dump(net, net.name+'.pkl')

#%%
#import torchviz
#x,y = trainData[0]
#loss = F.cross_entropy(net(x), y.unsqueeze(0))
#dot = torchviz.make_dot(net(x))
#dot.render('compgraph20')
