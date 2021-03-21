import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec

from dt_utils import Timer
from data_utils import generate_recognition_data
from torch_net_utils import maxabs, plot_W_seq, plot_xhy_seq, run_net, plot_corr, list2tensor
from neural_net_utils import plot_W, plot_B

#%%
import sys; 
sys.path.insert(0, '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-04-22')
from plastic import PlasticNet #import the old version of PlasticNet to load the nets properly

#%%
anti = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-04-11/nets/PlasticNet_R=7.pkl')
hebb = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-04-22/R=0-10_T=2000/PlasticNet_R=7_T=2000.pkl')

d = 50  #length of input vector
P = .5  #probability of repeat
R = 4   #repeat interval

def burn_in(net, data):
    """Run the network on a dataset without resetting to get self.A into a steady-state"""
    for x in data.tensors[0]:
        net(x)
        
burn = generate_recognition_data(T=1000, d=d, R=R, P=P, interleave=True, astensor=True)
burn_in(anti, burn)
burn_in(hebb, burn)

#%%
data = generate_recognition_data(T=100, d=d, R=R, P=P, interleave=True, astensor=True)

#%%
A_hist, WA_hist = plot_W_seq(anti, data, resetHebb=False)
#plot_W_seq(hebb, data, resetHebb=False)

#%%
y_hist,h_hist = plot_xhy_seq(anti, data, resetHebb=False)     
#y_hist,h_hist = plot_xhy_seq(hebb, data, resetHebb=False)     
 
#%%                 
plot_W([anti.w1.detach().numpy(), anti.w2.detach().numpy()])                 
plot_B([anti.b1.detach().numpy(), anti.b2.detach().numpy()])
                 
#plot_W([hebb.w1.detach().numpy(), hebb.w2.detach().numpy()])                 
#plot_B([hebb.b1.detach().numpy(), hebb.b2.detach().numpy()])         

#%%   
#%%
anti = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-04-11/nets/PlasticNet_R=7.pkl')
hebb = joblib.load('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-04-22/R=0-10_T=2000/PlasticNet_R=7_T=2000.pkl')

d = 50  #length of input vector
P = .5  #probability of repeat
R = 4   #repeat interval

def burn_in(net, data):
    """Run the network on a dataset without resetting to get self.A into a steady-state"""
    for x in data.tensors[0]:
        net(x)
        
burn_in(anti, burn)
burn_in(hebb, burn)

               
net = hebb
zeroW1 = False
zeroA = True

if zeroW1 and zeroA:
    raise Exception()
elif zeroW1:
    suff = 'Aonly'
elif zeroA:
    suff = 'Wonly'
else:
    suff = 'W+A'
    
if net == anti:
    pref = 'anti'
elif net == hebb:
    pref = 'hebb'
else:
    raise Exception()

_, h_hist, _, _ = run_net(net, data, zeroW1=zeroW1, zeroA=zeroA)
h_hist = list2tensor(h_hist)
R = plot_corr(h_hist,'{}_{}'.format(pref, suff))

#Remove repeats      
Nh = h_hist[0].shape[0] 
y = data.tensors[1]
h_hist_norep = h_hist[(y!=torch.ones(1)).repeat(1,Nh)].reshape(-1,Nh)
Rnr = plot_corr(h_hist_norep, '{}_{}_norep'.format(pref, suff))


                  
            
                  
                  
                  
                  
                  
#%%
#  def plot_xhy_seq(net, data):
#    hist = np.empty((d, len(data.tensors[0])))
#    
#    
#    fig, ax = plt.subplots()
#    hIm = ax.imshow(np.flipud(kur.distrK), cmap='GnBu', aspect='auto', 
#            extent=[np.min(kur.t), np.max(kur.t), np.min(kur.distrKbins), np.max(kur.distrKbins)],
#            norm=colors.LogNorm()) 

#    pos = ax.get_position().bounds
#    cax = fig.add_axes((pos[0]+pos[2]-0.1, pos[1]+.05, 0.01, pos[3]-.1))
#    fig.colorbar(hIm, cax=cax)
#
#%%
def plot_avg_plastic_magnitude(net, data):
    net.A = torch.zeros_like(net.w1) 
    avgmag = []
    minmag = []
    maxmag = []
    for i,x in enumerate(data.tensors[0]):
        net(x)
        avgmag.append( net.A.abs().mean().item() )
        minmag.append( net.A.abs().min().item() )
        maxmag.append( net.A.abs().max().item() )
    x = np.arange(len(data.tensors[1]))
    
    line = plt.plot(x, avgmag)
    plt.fill_between(x, minmag, maxmag, color=line[0].get_color(), alpha=0.2)
    

#longdata = generate_recognition_data(T=1000, d=d, R=R, P=P, interleave=True, astensor=True)
plt.figure()
plot_avg_plastic_magnitude(anti, longdata)
plot_avg_plastic_magnitude(hebb, longdata)
plt.legend(['anti', 'hebb'])
plt.title('steady-state of plastic weights')
plt.ylabel('max(|A|)')
plt.xlabel('time (input #)')



