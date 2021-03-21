import types
import random

import numpy as np
import matplotlib.pyplot as plt
import joblib

from fastweights_numpy import FastWeights, FeedforwardNet
from dt_utils import Timer
from data_utils import generate_recognition_data, augment_data
from neural_net_utils import Sigmoid, Relu, Quadratic, CrossEntropy, plot_W, plot_B

#%% Generate data
d = 50     #length of input vector
R = 1      #repeat interval
P = .5     #probability of repeat

print 'Creating data, d={}, R={}, P={}'.format(d, R, P)
trainData = generate_recognition_data(T=20000, R=R, d=d, P=P)
testData = generate_recognition_data(T=5000, R=R, d=d, P=P)
testData2 = random.sample(trainData, len(trainData))
testData3 = random.sample(trainData, len(trainData))

trainChance = 1-np.sum([xy[1] for xy in trainData], dtype=np.float)/len(trainData)
testChance = 1-np.sum([xy[1] for xy in testData ], dtype=np.float)/len(testData)  

#%% Define network hyperparameters
dims = [d, 100, 1]                    #dimensions of the network layers
Nonlin = Sigmoid# Relu#              #nonlinearity of hidden units
Loss = CrossEntropy#  Quadratic#        #loss function

if Loss is CrossEntropy:
    gam = 0.03                       #backprop learning rate
elif Loss is Quadratic:
    gam = 0.2
    
lam = 0                               #fast weight decay multiplier
eta = .5                              #fast weight learning rate

epochs = 3

bias = True

#Convenience functions
def make_freezeMask(W, maskType):
    mask = [np.ones_like(w) for w in W]
    r1,c1 = W[0].shape
    r2,c2 = W[1].shape
    if 'h1in' == maskType:
        mask[0][:] = 0
    elif 'h1in_top' == maskType:
        mask[0][:r1/2] = 0
    elif 'h1out_top' == maskType:
        mask[1][:,:c2/2] = 0
    elif 'h1in+h1out_top' == maskType:
        mask[0][:] = 0
        mask[1][:,:c2/2] = 0
    elif 'h1in_top+h1out_top' == maskType:
        mask[0][:r1/2] = 0
        mask[1][:,:c2/2] = 0
    elif 'h1weights' == maskType:
        mask[0][:-1,:-1] = 0
#        mask[1][-1:] = 0
#        mask[1][:,-1:] = 0
    else:
        ValueError()
    return mask


def run(net, optimizer):
    '''Run optimizer on the network'''
    with Timer('{}, {}'.format(net.name, optimizer)):        
        if 'sgd' in optimizer.lower():
            hist = net.sgd(trainData, testData=testData, gam=gam, epochs=epochs)
        elif 'adam' in optimizer.lower():
            hist = net.adam(trainData, testData=testData, epochs=epochs)
        else:
            raise ValueError()
    return hist


def make_net(initW, initB, netType, freezeW=None, freezeB=None):
    if 'feedforward' in netType.lower():
         net = FeedforwardNet(initW, initB, f=Nonlin, Loss=Loss)
    elif 'fastweights' in netType.lower():
         net = FastWeights(initW, initB, f=Nonlin, Loss=Loss, lam=lam, eta=eta)
    elif 'fastrandom' in netType.lower():
         net = make_fwRand(initW,initB)
    elif 'fasthalf'in netType.lower():
        net = make_fwHalf(initW,initB)
    else: 
         raise ValueError()
     
    if freezeW:    
        net.freezeW = freezeW
    if freezeB:
        net.freezeB = freezeB
    return net
         

def make_fwRand(initW, initB):
    '''Make a network with random noise instead of fast weights'''
    def random_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            hhT = np.outer(h[l+1], h[l])
            #random normal matrix with mean and std equal to that of h*h^T
            R = np.std(hhT)*np.random.randn(h[l+1].size, h[l].size) + np.mean(hhT)
            self.A[l] = self.lam*self.A[l] + self.eta*R
    
    #replace update_A method with random        
    fwRand = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastRandom', lam=lam, eta=eta)
    fwRand.update_A = types.MethodType(random_A, fwRand)  
    return fwRand


def make_fwHalf(initW, initB):
    '''Make a network with fast weights only in the top half of hidden layer'''
    def half_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            topHalf = np.ones_like(h[l+1])
            topHalf[len(h[l+1])/2:] = 0
            topHalf = np.outer(topHalf, h[l])
            self.A[l] = self.lam*self.A[l] + self.eta*topHalf
    
    #replace update_A method        
    fwHalf = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastHalf', lam=lam, eta=eta)
    fwHalf.update_A = types.MethodType(half_A, fwHalf)  
    return fwHalf


#%%
#fixed random initial W for comparison across networks
net = FeedforwardNet(dims, B=bias, f=Nonlin, Loss=Loss)
initW, initB = net.W, net.B
 
#%%
freezeW, freezeB = None, None #make_freezeMask(initW, 'h1weights')
#%% Run the experiment
print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)

netList = ['FastHalf'] #['FastWeights', 'FastHalf', 'Feedforward', 'FastRandom']
nets = {net:None for net in netList}   
for netType in netList:
    nets[netType] = make_net(initW, initB if bias else None, netType, freezeW, freezeB)
    run(nets[netType], 'Adam')


#%%
plotsList = ['test_acc', 'test_loss', 'train_acc', 'train_loss'] #keys in net.hist dict
plt.subplots(2,2)
for s,k in enumerate(plotsList, start=1):
    plt.subplot(2,2,s)
    for netType in netList:            
        #Plot
        hist = nets[netType].hist
        epochsRange = list(range(hist['epoch']+1))
        plt.plot(epochsRange, hist[k], label=netType)
    if k == 'test_acc':
        plt.plot(epochsRange, testChance*np.ones(len(hist[k])), 'k--', label='chance') 
    if k == 'train_acc':
        plt.plot(epochsRange, trainChance*np.ones(len(hist[k])), 'k--', label='chance') 
    plt.title(k)
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()

#%%
#for netType in netList:
#    for net, title in zip([nets_both, nets_last, nets_last_bot, nets_both_bot],
#                          ['both', 'output', 'output_bottom', 'both_bottom']):
#        fig, _ = plot_W(net[netType].W)
#        fig.set_size_inches(4.3, 6.6)
#        fig.axes[1].images[-1].colorbar.set_label(title)
#        fig.savefig('results/2019-01-15/{}_W_{}'.format(netType, title))
#
#fig,_ = plot_W(initW)
#fig.set_size_inches(4.3, 6.6)
#fig.axes[1].images[-1].colorbar.set_label('initialization')

#%%

for netType in netList:
    fig, _ = plot_B(nets[netType].B)
    fig.set_size_inches(1.8, 6.6) 
#    fig.savefig('results/2019-01-29/FastHalf_nobias_both_b_epoch3'.format(netType, 'both'))
    
    fig, _ = plot_W(nets[netType].W)
    fig.set_size_inches(4.3, 6.6) 
#    fig.savefig('results/2019-01-29/FastHalf_nobias_both_W_epoch20'.format(netType, 'both'))



#%%
#TODO: why is fwRand so slow?
#TODO: regularization: L2? Dropout? Early-stop (validation set)?
#TODO: switch to pytorch
#TODO: merge optimizers


