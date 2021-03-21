import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec

from dt_utils import Timer
from data_utils import generate_recognition_data
from torch_net_utils import maxabs, plot_W_seq, plot_xahy_seq, run_net, plot_corr, list2tensor
from neural_net_utils import plot_W, plot_B

def burn_in(net, data):
    """Run the network on a dataset without resetting to get self.A into a steady-state"""
    for x in data.tensors[0]:
        net(x)

#%%
anti = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-05-20/PlasticNet_R=6_anti.pkl')
hebb = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-05-20/PlasticNet_R=6_hebb.pkl')

d = 50  #length of input vector
P = .5  #probability of repeat
R = 4   #repeat interval
    
burn = generate_recognition_data(T=1000, d=d, R=R, P=P, interleave=True, astensor=True)
burn_in(anti, burn)
burn_in(hebb, burn)


#%%
plot_W([anti.w1.detach().numpy(), anti.w2.detach().numpy()])                 
plot_B([anti.b1.detach().numpy(), anti.b2.detach().numpy()])
                 
plot_W([hebb.w1.detach().numpy(), hebb.w2.detach().numpy()])                 
plot_B([hebb.b1.detach().numpy(), hebb.b2.detach().numpy()]) 

#%%
data = generate_recognition_data(T=100, d=d, R=R, P=P, interleave=True, astensor=True)

#%%
#plot_W_seq(anti, data, resetHebb=False)
#plot_W_seq(hebb, data, resetHebb=False)

#%%
plot_xahy_seq(anti, data, resetHebb=False)     
plot_xahy_seq(hebb, data, resetHebb=False)     
        

#%%
plot_h_distr(anti, data)
plot_h_distr(hebb, data)


#%%
fname = 'PlasticNet_R=10_w2only_w1sparse'
net = joblib.load(fname+'.pkl')

hist = run_net(net, data)
h_hist = list2tensor(hist['h'])
R = plot_corr(h_hist, fname)




















