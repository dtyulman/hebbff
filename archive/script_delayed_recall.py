"""Recreate the 'memory capacity' experiement from Collins..Sussillo2016 w/ alternative network architectures"""
import sys
import joblib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

from dt_utils import Timer
from hebbian import HebbRecall
from plastic_gated import GSP1Recall
from recurrent_nets import VanillaRNNRecall, LSTMRecall
from data_utils import generate_delayed_recall_data
from neural_net_utils import plot_train_perf, plot_train_perf_from_log, infinite_data_training, nan_mse_loss


#%% data params
R = 2
T = 2000
interleave = False
#%% netwk params
net = 'RNN'
d = 25
Nh = 50 #should be able to go down to Nh=d in a recurrent net with perfect reconstruction
dims = [d, Nh, d]

#%% training params
trainingMethod = 'inf1'
maxEpochs=10

#%% parse command line args (if they exist) to override defaults given above
# script_delayed_recall.py net R trainingMethod interleave
if len(sys.argv)>=2:
    net = sys.argv[1]
    maxEpochs=500000
    if len(sys.argv)>=3:
        R = int(sys.argv[2])
        if len(sys.argv)>=4:
            trainingMethod = sys.argv[3]
            if len(sys.argv)>=5:
                interleave = True if sys.argv[4].lower()=='true' else False


#%% run
if net == '3GS':
    raise NotImplementedError
    # net = ThreeGateSynapseRecall(dims)
elif net == 'GSP1':
    net = GSP1Recall(dims, lossfn=nan_mse_loss)
elif net == 'RNN':
    net = VanillaRNNRecall(dims, lossfn=nan_mse_loss)
elif net == 'Hebb':
    net = HebbRecall(dims, lossfn=nan_mse_loss, lam=0.5, eta=-0.5, )
elif net == 'LSTM':
    net = LSTMRecall(dims, lossfn=nan_mse_loss)
elif net.endswith('.pkl'):
    net = joblib.load(net)

#%%
with Timer('{}, training {}'.format(net.name, trainingMethod)): 
    if trainingMethod.startswith('inf'):
        epochsPerBatch = int(trainingMethod[3:])
        infinite_data_training(net, generate_delayed_recall_data, 
                               maxEpochs=maxEpochs, epochsPerBatch=epochsPerBatch,
                               T=T, d=d, R=R, interleave=interleave)        
    elif trainingMethod == 'std':
        trainData = generate_delayed_recall_data(T=T, d=d, R=R, interleave=interleave)
        net.fit(trainData, epochs=maxEpochs)
    else: 
        raise ValueError('Invalid training method')
        
        
#%%
net.save('{}_Nh={}_R={}_{}_intrlv={}.pkl'.format(net.name,Nh,R,trainingMethod,interleave))











