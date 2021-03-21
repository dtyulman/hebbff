import types
import random

import numpy as np
import matplotlib.pyplot as plt

from fastweights_numpy import FastWeights, FeedforwardNet
from dt_utils import Timer
from data_utils import generate_recognition_data
from neural_net_utils import Sigmoid, Relu, Quadratic, CrossEntropy 

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

epochs = 100


#Convenience functions
def run(net, optimizer):
    '''Run optimizer on the network'''
    with Timer('{}, {}'.format(net.name, optimizer)):        
        if 'sgd' in optimizer.lower():
            hist = net.sgd(trainData, testData=testData, gam=gam, epochs=epochs)
        elif 'adam' in optimizer.lower():
            hist = net.adam(trainData, testData=testData, epochs=epochs)
        else:
            raise ValueError()
    print 
    return hist


def make_net(initW, initB, netType):
    if 'feedforward' == netType.lower():
         return FeedforwardNet(initW, initB, f=Nonlin, Loss=Loss)
    elif 'fastweights' == netType.lower():
         return FastWeights(initW, initB, f=Nonlin, Loss=Loss, lam=lam, eta=eta)
    elif 'fastrandom' == netType.lower():
         return make_fwRand(initW, initB)
    elif 'fasthalf' == netType.lower():
        return make_fwHalf(initW, initB)
    elif 'fasthalf_correct' == netType.lower():
        return make_fwHalf_correct(initW, initB)
    elif 'fastweights_ones' == netType.lower():
        return make_fw_ones(initW, initB)
    else: 
         raise ValueError()
         

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
    '''Make a network with random noise instead of fast weights'''
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


def make_fwHalf_correct(initW, initB):
    def half_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            topHalf = h[l+1][:] #<-- THIS LINE IS CORRECTED FROM OTHER VERSION
            topHalf[len(h[l+1])/2:] = 0
            topHalf = np.outer(topHalf, h[l])
            self.A[l] = self.lam*self.A[l] + self.eta*topHalf
    
    #replace update_A method        
    fwHalf = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastHalf_correct', lam=lam, eta=eta)
    fwHalf.update_A = types.MethodType(half_A, fwHalf)  
    return fwHalf    


def make_fw_ones(initW, initB):
    def update_A_ones(self, h):
        """Use hidden layer activation to all ones instead of what it should be"""
        for l in range(self.L-2): #leave A[-1] at 0
            self.A[l] = self.lam*self.A[l] + self.eta*np.outer(np.ones_like(h[l+1]), h[l]) 
    
    #replace update_A method        
    fw = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastWeights_Ones', lam=lam, eta=eta)
    fw.update_A = types.MethodType(update_A_ones, fw)  
    return fw   
 
#%%
#fixed random initial W for comparison across networks
initW = FeedforwardNet(dims, f=Nonlin, Loss=Loss).W 
initB = FeedforwardNet(dims, B=True, f=Nonlin, Loss=Loss).B

#%% Run the experiment
netList = ['FastWeights']#, 'FastWeights_ones', 'FastHalf', 'FastHalf_correct']#'FastWeights', 'FastWeights_ones', 'FastHalf',   'FastRandom', 'Feedforward', ]
optList = ['Adam'] #, 'SGD']

print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)
nets = {net:{opt:None for opt in optList} for net in netList}   
for netType in netList:
    for optimizer in optList: 
        nets[netType][optimizer] = make_net(initW, initB, netType)
        run(nets[netType][optimizer] , optimizer)

#%% Plot the results
for optimizer in optList:
    plt.subplots(2,2)
    s=1
    for k in ['test_acc', 'test_loss', 'train_acc', 'train_loss']:
        plt.subplot(2,2,s)
        for netType in netList:
            hist = nets[netType][optimizer].hist
            epochsRange = list(range(len(hist[k])))
            plt.plot(epochsRange, hist[k], label=netType)
        if k == 'test_acc':
            plt.plot(epochsRange, testChance*np.ones(len(hist[k])), 'k--', label='chance') 
        if k == 'train_acc':
            plt.plot(epochsRange, trainChance*np.ones(len(hist[k])), 'k--', label='chance') 
        plt.title(k)
        s += 1
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()

#%%
#TODO: add bias
#TODO: determine convergence via validation set
#TODO: why is fwRand so slow?
#TODO: regularization:  Train only output layer (and/or L2? Dropout? Early-stop?)
#TODO: switch to pytorch


