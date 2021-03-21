"""
Notes:
    - Fast Weights under/overflows if using Relu with eta or lam > 0
    
"""
import numpy as np
import tensorflow.keras.datasets.mnist as mnist

from fastweights import FeedforwardNet, FastWeights, weights_to_vector, vector_to_weights
from dt_utils import Timer, numerical_gradient
from neural_net_utils import Sigmoid, Relu, CrossEntropy, Quadratic


#%%
def preproc_mnist(x,y):
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) #take each image from 28x28 to 1x784
    x = x/np.float(np.max(x))*2-1
    y = (y>=4).astype(np.float64) #for binary classifier
    return zip(x, y)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
trainData = preproc_mnist(x_train, y_train)#[0:5000]
testData = preproc_mnist(x_test, y_test)[0:2000]

dims = [784, 100, 1]                  #dimensions of the network layers
#Nonlin = Sigmoid# Relu#              #nonlinearity of hidden units
#Loss = CrossEntropy# Quadratic#      #loss function

#%% Validate backprop gradient calculation against numerical gradient
for Nonlin in [Sigmoid, Relu]:
    for Loss in [Quadratic, CrossEntropy]:
        print '\n{},{}'.format(Nonlin, Loss)
        def E(W_vector):
            """Loss as a function of the weights. Used in numerical gradient approx"""
            W = vector_to_weights(W_vector, dims)
            net = FeedforwardNet(W, f=Nonlin, Loss=Loss)
            return net.average_loss([trainData[0]]) #TODO: this may be off by constant factor because average loss instead of total
        
        ff = FeedforwardNet(dims, f=Nonlin, Loss=Loss)
        fw = FastWeights(ff.W, f=Nonlin, Loss=Loss)
        
        with Timer('Backprop'):
            dEdW_bp = ff.backprop(trainData[0][0], trainData[0][1])  
        with Timer('FW Backprop'):
            dEdW_bp_fw = fw.backprop(trainData[0][0], trainData[0][1])
        with Timer('Numerical gradient'):   
            dEdW_num = vector_to_weights( numerical_gradient(E, weights_to_vector(ff.W)), dims )     
            
        for l in range(len(dEdW_bp)):
            err = np.sum(np.abs(dEdW_num[l] - dEdW_bp[l]))
            if err > 1e-5: #numerical gradient must equal backprop
                print 'layer:{}, err:{}, num_norm:{}, bp_norm:{}'.format(
                        l, err, 
                        np.linalg.norm(weights_to_vector(dEdW_num[l])),
                        np.linalg.norm(weights_to_vector(dEdW_bp[l]))
                        )
                raise AssertionError('NUM != BP')
            
            err = np.sum(np.abs(dEdW_bp_fw[l] - dEdW_bp[l]))    
            if err != 0: #fast weight backprop must be equal to feedforward backprop on first datapoint
                print 'layer:{}, err:{}, fw_norm:{}, bp_norm:{}'.format(
                        l, err, 
                        np.linalg.norm(weights_to_vector(dEdW_bp_fw[l])),
                        np.linalg.norm(weights_to_vector(dEdW_bp[l]))
                        )
                raise AssertionError('FW BP != FF BP')
        
#%% Train and test the network as a binary classifier  
lam = 0
eta = 0
for Nonlin in [Sigmoid, Relu]:
    for Loss in [CrossEntropy, Quadratic]:  
        print '\n{},{}'.format(Nonlin, Loss)
        if Loss is CrossEntropy:
            gam = 0.005
        elif Loss is Quadratic:
            gam = 0.2
        
        epochs = 5
        ff = FeedforwardNet(dims, f=Nonlin, Loss=Loss)
        initW = ff.W
        ff_loss = ff.sgd(trainData, epochs=epochs, gam=gam)
        print 'MNIST Feedforward: Train accuracy: {}, Test accuracy: {}'.format(ff.accuracy(trainData), ff.accuracy(testData))
        
        fw = FastWeights(initW, f=Nonlin, Loss=Loss, lam=lam, eta=eta)
        fw_loss = fw.sgd(trainData, epochs=epochs, gam=gam)
        print 'MNIST Fast Weights: Train accuracy: {}, Test accuracy: {}'.format(fw.accuracy(trainData), fw.accuracy(testData))
        
        for epoch in range(epochs):
            assert( fw_loss[epoch] == ff_loss[epoch] ) 
#%% Train and test the network as a binary classifier using Adam optimizer  
lam = 0
eta = 0
f = Sigmoid
Loss = CrossEntropy 

epochs = 5
ff = FeedforwardNet(dims, f=Nonlin, Loss=Loss)
initW = ff.W
ff_loss = ff.adam(trainData, epochs=20)
print 'MNIST Feedforward, Adam: Train accuracy: {}, Test accuracy: {}'.format(ff.accuracy(trainData), ff.accuracy(testData))

#%%
fw = FastWeights(initW, f=Nonlin, Loss=Loss, lam=lam, eta=eta)
fw_loss = fw.adam(trainData, epochs=epochs)
print 'MNIST Fast Weights, Adam: Train accuracy: {}, Test accuracy: {}'.format(fw.accuracy(trainData), fw.accuracy(testData))

for epoch in range(epochs):
    assert( fw_loss[epoch] == ff_loss[epoch] )





#%%   











