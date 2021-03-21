import types

import numpy as np
import matplotlib.pyplot as plt

from fastweights_numpy import FastWeights, FeedforwardNet
from dt_utils import Timer
from data_utils import generate_recognition_data
from neural_net_utils import Sigmoid, CrossEntropy, plot_W, plot_B, maxabs

#%% Generate data
d = 50     #length of input vector
R = [1,2]     #repeat interval
P = .5     #probability of repeat

print 'Creating data, d={}, R={}, P={}'.format(d, R, P)
trainData = generate_recognition_data(T=20000, Rlist=R, d=d, P=P)
testData = generate_recognition_data(T=5000, Rlist=R, d=d, P=P)

trainChance = 1-np.sum([xy[1] for xy in trainData], dtype=np.float)/len(trainData)
testChance  = 1-np.sum([xy[1] for xy in testData ], dtype=np.float)/len(testData)  

print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)

#%% Define network hyperparameters
dims = [d, 120, 1]                    #dimensions of the network layers
Nonlin = Sigmoid              #nonlinearity of hidden units
Loss = CrossEntropy           #loss function
   
lams = [.7, 0, 0]                      #fast weight decay multiplier
etas = [.5, 5, 0]                      #fast weight learning rate


def make_fwMulticpt(initW, initB, lams, etas):
    assert(len(lams)==len(etas))

    def multicpt_A(self, h):
        """Only the first layer has fastweights cf. in others "all but the last" have 
        fastweights (which is the same for L=2)"""               
        pre = h[0]
#        post = h[1]
        post = np.ones_like(h[1])
        self.A[0] = self.lam*self.A[0] + self.eta*np.outer(post, pre)
                   
    nC = len(lams)
    r,_ = initW[0].shape
    if r % nC != 0:
        raise ValueError('Modify to deal with unequal compartment sizes')
    f = r/nC   
    lam = np.zeros((r,1))
    eta = np.zeros((r,1))    
    for C in range(nC):
        lam[C*f:(C+1)*f] = lams[C]
        eta[C*f:(C+1)*f] = etas[C]
    
    fw = FastWeights(initW, initB, f=Nonlin, Loss=Loss, name='Multicompartment')
    fw.update_A = types.MethodType(multicpt_A, fw)  
    fw.eta = eta
    fw.lam = lam
    
    return fw

#%%
#fixed random initial W for comparison across networks
net = FeedforwardNet(dims, B=False)
initW, initB = net.W, net.B

#initW[0] = np.zeros_like(initW[0])

#%% Run the experiment
print '##### Train_chance:{}, Test_chance:{} #####'.format(trainChance, testChance)
  
net = make_fwMulticpt(initW, initB, lams, etas)
#net.freeze_weights([1], ['in'])
#net.freeze_bias([1,2])

with Timer(net.name):
    hist = net.adam(trainData, testData=testData, epochs=5)

#%%
epochsRange = list(range(hist['epoch']+1))
plotsList = ['test_acc', 'test_loss', 'train_acc', 'train_loss'] #keys in net.hist dict
plt.subplots(2,1)
for k in plotsList:
    if 'acc' in k:
        plt.subplot(2,1,1)
        plt.title('{} accuracy'.format(net.name))
    elif 'loss' in k:
        plt.subplot(2,1,2)
        plt.title('{} loss'.format(net.name))
        
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
    h_t.append( h[1].reshape(-1,1) )
    A_t.append( net.A[0] )
    WA_t.append( net.W[0]+net.A[0] )

hv = maxabs(h_t)
Av = maxabs(A_t)
WAv = maxabs(WA_t)

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

#axs[1,0].set_title('$W_{1,init}$')

plt.tight_layout()
fig.subplots_adjust(right=0.95)
cax = fig.add_axes([0.95, .05+.5, 0.01, 0.4])  
fig.colorbar(imA, cax=cax)
cax = fig.add_axes([0.95, .05, 0.01, 0.4])  
fig.colorbar(imWA, cax=cax)


#%%

#
##%%
##TODO: regularization: L2? Dropout? Early-stop (validation set)?
##TODO: merge optimizers
#
#
