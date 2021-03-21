import types

import numpy as np
import matplotlib.pyplot as plt

from fastweights_numpy import FastWeights, FeedforwardNet
from data_utils import generate_recognition_data
from neural_net_utils import Sigmoid, Relu, Quadratic, CrossEntropy, plot_W, plot_B

#%% Generate data
d = 50     #length of input vector
R = 1      #repeat interval
P = .5     #probability of repeat

print 'Creating data, d={}, R={}, P={}'.format(d, R, P)
trainData = generate_recognition_data(T=20000, R=R, d=d, P=P)
testData = generate_recognition_data(T=5000, R=R, d=d, P=P)

trainChance = 1-np.sum([xy[1] for xy in trainData], dtype=np.float)/len(trainData)
testChance  = 1-np.sum([xy[1] for xy in testData ], dtype=np.float)/len(testData)  

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


def maxabs_in_arraylist(L):
    v = -np.inf 
    for w in L:
        m = np.max(np.abs(w))
        if m > v: v = m
    return v  
 
#%%
#fixed random initial W for comparison across networks
net = FeedforwardNet(dims, B=True, f=Nonlin, Loss=Loss)
initW, initB = net.W, net.B

#%%
initW_topEqBot = initW[:]
initW_topEqBot[0][50:] = initW[0][:50]
#%% Run the experiment
print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)
  
net = make_fwHalf(initW, initB)
#net = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='FastHalf', lam=lam, eta=eta)
net.freeze_weights([1], ['in'])
hist = net.adam(trainData, testData=testData, epochs=epochs)


#%%
epochsRange = list(range(hist['epoch']+1))
plotsList = ['test_acc', 'test_loss', 'train_acc', 'train_loss'] #keys in net.hist dict
plt.subplots(2,1)
for k in plotsList:
    if 'acc' in k:
        plt.subplot(2,1,1)
        plt.title('FastHalf accuracy')
    elif 'loss' in k:
        plt.subplot(2,1,2)
        plt.title('FastHalf loss')
        
    plt.plot(epochsRange, hist[k], 'b-' if 'test' in k else 'b--', label=k)
    
    if k == 'test_acc':
        plt.plot(epochsRange, testChance*np.ones(len(hist[k])), 'k-', label='test_chance') 
    if k == 'train_acc':
        plt.plot(epochsRange, trainChance*np.ones(len(hist[k])), 'k--', label='train_chance') 
    
    plt.legend()

plt.xlabel('Epochs')
plt.gcf().set_size_inches(3,5)
plt.tight_layout()

#%%
T = 10
testData = generate_recognition_data(T=T, R=R, d=d, P=P)
#%%
net.init_A()

WA_t = [net.W[0]]
A_t = []
h_t = []

for i,(x,y) in enumerate(testData):
    a,h = net.feedforward(x)
    h_t.append( h[2].reshape(-1,1) )
    A_t.append( net.A[0] )
    WA_t.append( net.W[0]+net.A[0] )

hv = maxabs_in_arraylist(h_t)
Av = maxabs_in_arraylist(A_t)
WAv = maxabs_in_arraylist(WA_t)

#%%
fig, axs = plt.subplots(1,len(testData))
axs = np.expand_dims(axs,0)
for i in range(len(testData)):
    imH = axs[0,i].matshow(h_t[i], cmap='Reds', vmin=0, vmax=hv)
    axs[0,i].set_title('y={}'.format(testData[i][1]))

[ax.axis('off') for ax in axs.flatten()]
plt.tight_layout()
fig.subplots_adjust(right=0.95)
cax = fig.add_axes([0.95, .04+.33, 0.01, 0.25])  
fig.colorbar(imH, cax=cax)

#%%
fig, axs = plt.subplots(2,len(testData)+1)
for i in range(len(testData)):     
    imA = axs[0,i+1].matshow(A_t[i], cmap='RdBu_r', vmin=-Av, vmax=Av)
    axs[0,i+1].set_title('y={}'.format(testData[i][1]))
    imWA = axs[1,i+1].matshow(WA_t[i+1], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 
imWA = axs[1,0].matshow(WA_t[0], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 

[ax.axis('off') for ax in axs.flatten()]
plt.tight_layout()
fig.subplots_adjust(right=0.95)
cax = fig.add_axes([0.95, .05+.5, 0.01, 0.4])  
fig.colorbar(imA, cax=cax)
cax = fig.add_axes([0.95, .05, 0.01, 0.4])  
fig.colorbar(imWA, cax=cax)


#%%



    
#%%
fig, _ = plot_B(net.B)
fig.set_size_inches(1.8, 6.6) 
#    fig.savefig('results/2019-01-29/FastHalf_nobias_both_b_epoch3'.format(netType, 'both'))

fig, _ = plot_W(net.W)
fig.set_size_inches(4.3, 6.6) 
#    fig.savefig('results/2019-01-29/FastHalf_nobias_both_W_epoch20'.format(netType, 'both'))


#%%

fig, ax = plt.subplots(2,2)
ax[0,0].hist([net.W[0][:50].flatten(), 
              initW[0][:50].flatten()], bins=50)
ax[0,0].set_title('$W_1^{top}$')

ax[0,1].hist([net.W[0][50:].flatten(), 
              initW[0][50:].flatten()], bins=50)
ax[0,1].set_title('$W_1^{bot}$')

ax[1,0].hist([net.W[1][:,:50].flatten(), 
              initW[1][:,:50].flatten()], bins=20)
ax[1,0].set_title('$W_2^{top}$')

ax[1,1].hist([net.W[1][:,50:].flatten(), 
              initW[1][:,50:].flatten()], bins=20, label=['trained', 'init'])
ax[1,1].set_title('$W_2^{bot}$')

ax[1,1].legend()



#%%
#TODO: why is fwRand so slow?
#TODO: regularization: L2? Dropout? Early-stop (validation set)?
#TODO: switch to pytorch
#TODO: merge optimizers


