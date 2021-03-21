import types

import numpy as np
import matplotlib.pyplot as plt

from fastweights_numpy import FastWeights, FeedforwardNet
from dt_utils import Timer
from data_utils import generate_recognition_data
from neural_net_utils import Sigmoid, CrossEntropy, plot_W, plot_B, maxabs

#%% Generate data
d = 50     #length of input vector
R = 1     #repeat interval
P = .5     #probability of repeat

print 'Creating data, d={}, R={}, P={}'.format(d, R, P)
trainData = generate_recognition_data(T=20000, R=R, d=d, P=P)
testData = generate_recognition_data(T=5000, R=R, d=d, P=P)

trainChance = 1-np.sum([xy[1] for xy in trainData], dtype=np.float)/len(trainData)
testChance  = 1-np.sum([xy[1] for xy in testData ], dtype=np.float)/len(testData)  

print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)

#%% Define network hyperparameters
dims = [d, 1, 1]                    #dimensions of the network layers
Nonlin = Sigmoid              #nonlinearity of hidden units
Loss = CrossEntropy           #loss function
   
lam = 0                      #fast weight decay multiplier
eta = .5                      #fast weight learning rate

epochs = 3

def make_fwHalf(initW, initB):
    '''Make a network with fast weights only in the top half of hidden layer'''
    def half_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            topHalf = h[l+1][:]
            topHalf[len(h[l+1])/2:] = 0
            topHalf = np.outer(topHalf, h[l])
            self.A[l] = self.lam*self.A[l] + self.eta*topHalf
    
    #replace update_A method        
    fwHalf = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastHalf', lam=lam, eta=eta)
    fwHalf.update_A = types.MethodType(half_A, fwHalf)  
    return fwHalf

def make_fwOnes(initW, initB):
    def ones_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            self.A[l] = self.lam*self.A[l] + self.eta*np.outer(np.ones_like(h[l+1]), h[l])   
    
    #replace update_A method        
    fw = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastOnes', lam=lam, eta=eta)
    fw.update_A = types.MethodType(ones_A, fw)  
    return fw    

#%%
#fixed random initial W for comparison across networks
net = FeedforwardNet(dims, B=True)
initW, initB = net.W, net.B

initW[0] = np.zeros_like(initW[0])

#%% Run the experiment
print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)
  
#net = make_fwHalf(initW, initB)
#net = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastWeights', lam=lam, eta=eta)
net = make_fwOnes(initW, initB)
#net.freeze_weights([1], ['in'])
#net.freeze_bias([1,2])

with Timer(net.name):
    hist = net.adam(trainData, testData=testData, epochs=3)

#%%
plot_train_perf(net, trainChance, testChance)

#%%
smallTest = generate_recognition_data(T=10, R=R, d=d, P=P)
plot_h_seq(net, smallTest, 'x')
plot_h_seq(net, smallTest, 'a')
plot_h_seq(net, smallTest, 'h')
plot_W_seq(net, smallTest)


#%%
#fig, _ = plot_B(net.B)
#fig.set_size_inches(1.8, 6.6) 
##    fig.savefig('results/2019-01-29/FastHalf_nobias_both_b_epoch3'.format(netType, 'both'))
#
#fig, _ = plot_W(net.W)
#fig.set_size_inches(4.3, 6.6) 
##    fig.savefig('results/2019-01-29/FastHalf_nobias_both_W_epoch20'.format(netType, 'both'))
#
#
##%%
#
#fig, ax = plt.subplots(2,2)
#ax[0,0].hist([net.W[0][:50].flatten(), 
#              initW[0][:50].flatten()], bins=50)
#ax[0,0].set_title('$W_1^{top}$')
#
#ax[0,1].hist([net.W[0][50:].flatten(), 
#              initW[0][50:].flatten()], bins=50)
#ax[0,1].set_title('$W_1^{bot}$')
#
#ax[1,0].hist([net.W[1][:,:50].flatten(), 
#              initW[1][:,:50].flatten()], bins=20)
#ax[1,0].set_title('$W_2^{top}$')
#
#ax[1,1].hist([net.W[1][:,50:].flatten(), 
#              initW[1][:,50:].flatten()], bins=20, label=['trained', 'init'])
#ax[1,1].set_title('$W_2^{bot}$')
#
#ax[1,1].legend()
#
#
#
##%%
##TODO: regularization: L2? Dropout? Early-stop (validation set)?
##TODO: switch to pytorch
##TODO: merge optimizers
#
#
