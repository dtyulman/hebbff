import types

import numpy as np
import matplotlib.pyplot as plt

from fastweights_numpy import FastWeights, FeedforwardNet
from dt_utils import Timer
from data_utils import generate_recognition_data, chance
from neural_net_utils import plot_W, plot_B, plot_W_seq, plot_h_seq, plot_train_perf


def make_fwRandPlast(initW, initB):
    def randplast_A(self, h):
        """Only the first layer has fastweights cf. in others "all but the last" have 
        fastweights (which is the same for L=2)"""               
        nRows = self.A[0].shape[0]
        randRow = np.random.randint(nRows)
        self.A[0][randRow,:] = self.eta * h[0]
                       
    fw = FastWeights(initW, initB, name='RandomPlasticity')
    fw.update_A = types.MethodType(randplast_A, fw)      
    return fw

#%% Generate data
d = 50     #length of input vector
R = 1      #repeat interval
P = .7     #probability of repeat
 
print 'Creating data, d={}, R={}, P={}'.format(d, R, P)
trainData = generate_recognition_data(T=20000, R=R, d=d, P=P, interleave=True)    
print 'Train_chance:{}'.format(chance(trainData))

testDataList = []
testChance = []
for R in range(1,20):
    testDataList.append( generate_recognition_data(T=20000, R=R, d=d, P=P, interleave=True) )
    testChance.append( chance(testDataList[-1]) )

#%% Run all of them
nets = []
for Nh in [1,2,3,4,10,20]:
    dims = [d, Nh, 1] #dimensions of the network layers
    tmp = FeedforwardNet(dims, B=True)
    initW, initB = tmp.W, tmp.B    
    initW[0] = np.zeros_like(initW[0])
    
    net = make_fwRandPlast(initW, initB)
    net.freeze_weights([1], ['in'])    
    with Timer('{} (Nh={})'.format(net.name, Nh)):
        net.adam(trainData, epochs=10000, earlyStop=True)
    nets.append(net) #save for posterity

#%% Test and plot all of them
netsPerf = []
for net in nets:
    netsPerf.append( [] )
    with Timer('Testing'): 
        for testData in testDataList:
            netsPerf[-1].append( net.accuracy(testData) )

plt.figure()
for i in range(len(netsPerf)):
    plt.plot(netsPerf[i], label='Nh={}'.format(nets[i].W[0].shape[0]))   
plt.plot(testChance, 'k--', label='Chance')
plt.xlabel('Repeat interval')
plt.ylabel('Test accuracy')
plt.legend()

#%% Run just one
Nh=40

dims = [d, Nh, 1] #dimensions of the network layers
tmp = FeedforwardNet(dims, B=True)
initW, initB = tmp.W, tmp.B    
initW[0] = np.zeros_like(initW[0])

net = make_fwRandPlast(initW, initB)
net.B[0]=np.tile(nets[-1].B[0],2)
net.W[1]=np.tile(nets[-1].W[1],2)
net.B[1]=nets[-1].B[1]

net.freeze_weights([1], ['in'])    
with Timer('{} (Nh={})'.format(net.name, Nh)):
    net.adam(trainData, epochs=10000, earlyStop=True)

#%% Test and plot just one
perform = []
for testData in testDataList:
    perform.append( net.accuracy(testData) )

plt.figure()
plt.plot(perform,'--', label='Nh={}'.format(net.W[0].shape[0]))
    

#%%
#TODO: regularization: L2? Dropout? Early-stop (validation set)?
#TODO: merge optimizers














