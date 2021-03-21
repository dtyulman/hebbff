import numpy as np
import torch 

import sys; sys.path.append('/Users/danil/My')
from dt_utils import Timer

from data_utils import generate_recognition_data, data_to_tensor, chance
from plastic import PlasticNet, LearnedHebb
from torch_net_utils import plot_hebb
from fastweights_numpy import FastWeights as FastWeightsNumpy


#%% Generate data
d = 50     #length of input vector
R = 6      #repeat interval
P = .5     #probability of repeat

print('Creating data, d={}, R={}, P={}'.format(d, R, P))
trainData = generate_recognition_data(T=2000, d=d, R=R, P=P)
trainDataTensor = data_to_tensor(trainData)

testData = generate_recognition_data(T=5000, d=d, R=R, P=P)
testDataTensor = data_to_tensor(testData)

trainChance = chance(trainData)
testChance = chance(testData)

print('Train_chance:{}, test_chance:{}'.format(trainChance, testChance))

#%%
dims = [d, 100, 1] #dimensions of the network layers
npNet = FastWeightsNumpy(dims)
initW, initB = npNet.W, npNet.B

#%% Infinite data training
net = PlasticNet(initW,initB, eta=0.5, lam=0.5, positiveEta=False)   
with Timer('PlasticNet Torch (train_chance={})'.format(trainChance)):
    for epoch in range(5000):
        trainDataTensor = generate_recognition_data(T=200, d=d, R=R, P=P, astensor=True)
        net.fit(trainDataTensor, epochs=1)
        print net.eta.item()
#        if net.eta < -0.05:
#            break
#%%
#learnHebb=True
#forcePositiveHebb=False
#eta = 0.9 #0.1354
#lam = 0.1 #-0.5107 
#
#net = PlasticNet(initW,initB, forcePositiveHebb=forcePositiveHebb, learnHebb=learnHebb, eta=eta, lam=lam)   
#with Timer('PlasticNet Torch (train_chance={})'.format(trainChance)):
#    net.fit(trainDataTensor, epochs=5)


net2 = LearnedHebb(initW,initB)
        
           
#%%
#import joblib
#joblib.dump(net.hist, filename)
#filename = 'plastnet_train=NP_learnH={}_posH={}_eta={}_lam={}_R={}_chance={}.pkl'.format(
#            learnHebb, forcePositiveHebb, eta, lam, R, trainChance)
#print(filename)

    
#%%
import torchviz
x = trainDataTensor[0][0]
x.requires_grad_()
dot = torchviz.make_dot(net(x))
dot.render('compgraph_syn')




#%%

    



#%% Test the pure feedforward network on binary MNIST
###################################################################################################
#from fastweights_numpy import FeedforwardNet as FeedforwardNumpy    
#    
#import tensorflow.keras.datasets.mnist as mnist
#def preproc_mnist(x,y):
#    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) #take each image from 28x28 to 1x784
#    x = x/np.float(np.max(x))*2-1
#    y = (y>=4).astype(np.float64) #for binary classifier
#    return zip(x, y)
#
#(x_train, y_train), _ = mnist.load_data()
#trainData = preproc_mnist(x_train[0:5000], y_train[0:5000])
#trainDataTensor = data_to_tensor(trainData)
#
##%%
#dims = [784, 100, 1] #dimensions of the network layers
#epochs = 5
#npNet = FeedforwardNumpy(dims)
#initW, initB = npNet.W, npNet.B
#
#npNet = FeedforwardNumpy(initW,initB)
#with Timer('Feedforward Numpy'):
#    npNet.adam(trainData, epochs=epochs)
#
##NEED TO COMMENT OUT ALL MENTIONS OF self.A IN PlasticNet FOR THIS TEST! 
#net = PlasticNet(initW,initB)   
#with Timer('PlasticNet Torch'):
#    net.fit(trainDataTensor, epochs=epochs)    
#
## Note: PlasticNet and FeedforwadNumpy don't train the same way, 
##       possibly due to different Adam implementations?
############################################################################################# 