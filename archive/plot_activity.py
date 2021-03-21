import math

import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from torch.utils.tensorboard import SummaryWriter
import torch

from data import generate_recog_data
from net_utils import load_from_file
from plotting import maxval, maxabs, plot_W, plot_B, ticks_off, plot_train_perf, plot_recog_generalization

torch.set_printoptions(sci_mode=False)
#%%
d = Nh = 25
fname = 'HebbNet_Nh=25_w1=posneg_eye-train_w2=randn-train.pkl'
net = load_from_file(fname, dims=[d, Nh, 1])
net.requires_grad_(False)

T = 20
data = generate_recog_data(T=T, d=d, R=3, P=1, interleave=False, multiRep=False)

h = torch.empty(d, T)
out = torch.empty(1, T)
dA_mean = torch.empty(d, T)
A_mean =  torch.empty(d, T)
for t,(x,y) in enumerate(data):
    A = net.A.clone()
    A_mean[:,t] = A.abs().mean(dim=1)

    a1, h[:,t], a2, out[:,t] = net(x, debug=True)
    dA = A - net.A.clone()

    dA_mean[:,t] = dA.abs().mean(dim=1)
    
#%%
#plt.plot(torch.nonzero((data.tensors[1]==1))[:,0], torch.zeros(data.tensors[1].sum().int().item()), '*')    
#plt.plot(h.mean(dim=0))

fig,ax = plt.subplots(13,2)
for i in range(Nh):
    ax.flatten()[i].plot(dA_mean[i,:])
    ax.flatten()[i].plot(torch.nonzero((data.tensors[1]==1))[:,0], torch.zeros(data.tensors[1].sum().int().item()), '*')    

    ax2 = ax.flatten()[i].twinx()
    ax2.plot(A_mean[i,:], color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    

ax[12,0].set_xlabel('t')
##ax[12,0].set_ylabel('$<|\Delta A_{ij}|>_j$ ')
ax[12,1].axis('off')
#ax[0,0].set_title('$<|\Delta A_{ij}|>_j$')

#plt.savefig('dA_mean.png', )

#%%

def PCA(X, k=2):
    # preprocess the data
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
    
    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

hPC = PCA(h.t())

colors = cm.gray(np.linspace(0, 1, T))
for t,c in enumerate(colors):
    if data[t][1] == 1:
        c = 'red'
    plt.plot(hPC[t,0], hPC[t,1], color=c)
    
#%%
    
net.requires_grad_(False)    
    
d=25
T=50
Nh=50
data = generate_recog_data(T=T, d=d, R=T+1, P=0.5, interleave=True, multiRep=True)
x0 = data[0][0] #keep track of the network's representation of this item throughout the epoch

#data.tensors[0][10] = x0 #show the item again halfway through

h_x0 = torch.empty(Nh, T)
out_x0 = torch.empty_like(data.tensors[1])

net.reset_state()
for t,(x,y) in enumerate(data):
    net.plastic=False
    a1, h_x0[:,t], a2, out_x0[t] = net(x0, debug=True) #write down network's representation of x0 without activating plasticity

    net.plastic=True
    net(x) #show the next item to the network. Will change representatin of x0.
    

h_x0_PCA = PCA(h_x0.t()).detach()

plt.figure()
color = cm.coolwarm
for t,(x,y) in enumerate(data):
    if t>=1:
        plt.arrow(h_x0_PCA[t-1,0], h_x0_PCA[t-1,1], h_x0_PCA[t,0]-h_x0_PCA[t-1,0], h_x0_PCA[t,1]-h_x0_PCA[t-1,1], 
                  length_includes_head=True, width=0.000001, head_width=0.004, color='k')
    plt.plot(h_x0_PCA[t,0], h_x0_PCA[t,1], marker='.', color=color(out_x0[t].item()))

plt.axis('equal')
plt.title('[Hidden layer repr of $x_0$](t)')
plt.xlabel('PC1 of hidden layer')
plt.ylabel('PC2 of hidden layer')
plt.tight_layout()