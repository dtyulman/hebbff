import types, os

import matplotlib, torch, joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from net_utils import load_from_file
from networks import HebbRecogDecoupledManual, HebbRecogDecoupledManualSequential
from data import generate_recog_data, recog_chance
from plotting import plot_recog_generalization
from bogacz import BogaczAntiHebb

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 7}
matplotlib.rc('font', **font)


os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-10-31')
#%%
#d=50
#Nh=100
#Rs = np.arange(1,14)
#eta0s = [0.5, 0.1, -0.1, -0.5]
#
##d=25
##Nh=50
##Rs = np.arange(1,15)
##eta0s = [1, 0.5, 0.1, -0.1, -0.5, -1]
#
#notConverged = []
#_,ax1 = plt.subplots()
#_,ax2 = plt.subplots()
#for i,eta0 in enumerate(eta0s):
#    eta = np.full(len(Rs), np.nan)
#    acc = np.full(len(Rs), np.nan)
#    for j,R in enumerate(Rs):
#        fname = 'HebbNet[{},{},1]_train=inf{}_eta0={:.1f}.pkl'.format(d,Nh,R,eta0) 
#        print fname
#        try:
#            net = load_from_file(fname)
#        except:
#            raw_input('ERROR')
#
#                
#        data = generate_recog_data(T=5000, d=d, R=R, P=0.5)        
#        if net.accuracy(data.tensors) < 0.98:
#            color = cm.cool((eta0+1.1)/2.2)
#            ax1.scatter(R, net.eta.item(), marker='x', color=color)
#            notConverged.append(fname)
#        eta[j] = net.eta.item()
#        acc[j] = net.accuracy(data.tensors)
#
#        
#    color = cm.cool((eta0+1.1)/2.2)
#    ax1.plot(Rs, eta, color=color, label='$\eta_0$={}'.format(eta0))
#    ax2.plot(Rs, acc, color=color, label='$\eta_0$={}'.format(eta0))
#
#
#ax1.legend()
#ax1.set_xlabel('$R_{train}$')
#ax1.set_ylabel('$\eta_{final}$')          
#        
#
#ax2.legend()
#ax2.set_xlabel('$R_{train}$')
#ax2.set_ylabel('acc')     
 

#%% Fig 2. Hebbian vs antiHebbian
ax = None
gen_data = lambda R: generate_recog_data(T=R*100, d=50, R=R, P=0.5, interleave=True, multiRep=True)

for R in [5]:#[2,5,10]:
    for eta0 in [0.1, -0.1, 0]:
        if eta0 == 0:
            os.chdir('RNNs')
            net = load_from_file('VanillaRNN[50,100,1]_train=inf{}.pkl'.format(R))            
            os.chdir('..')
        else:
            net = load_from_file('HebbNet[50,100,1]_train=inf{}_eta0={}.pkl'.format(R,eta0))
#        if R==2:
#            c = 'r'
#        elif R==5:
#            c = 'b'
#        elif R==10:
#            c = 'g'
        
        if eta0 == 0:
            ls = '-'
            label = 'RNN'.format(R)
        elif net.eta>0: 
            ls = '-'
            label = 'HebbFF ($\lambda=${:.2f}, $\eta=${:.2f})'.format(net.lam, net.eta)
        elif net.eta<0:
            ls = '-'
            label = 'HebbFF ($\lambda=${:.2f}, $\eta=${:.2f})'.format(net.lam, net.eta)
        
        fmt = ls #c+ls
        if eta0==0:
            testR = []
            testAcc = []                
            Rmax = 15
            R = 1
            while R < Rmax:
                testData = gen_data(R)
                acc = net.accuracy(testData.tensors).item()
                chance = recog_chance(testData)
                testAcc.append( acc )
                testR.append( R )
                print 'R={}, acc={:.3f}, (chance={})'.format(R, acc, chance)
                R = int(np.ceil(R*1.1))
            ax.semilogx(testR, testAcc, fmt, label=label)
        else:        
            ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, fmt=fmt, label=label)

ax.legend()
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('Accuracy')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

#labels = ax.get_xticklabels()
#labels[1].set_text('$R_{train}$ = 5')
#ax.set_xticklabels(labels)

ax.get_figure().set_size_inches(3.75, 2)
ax.get_figure().tight_layout()

#%% Fig 3. Continual learning
fig, ax = plt.subplots()

Nh = 100
net = load_from_file('HebbNet[100,100,1]_train=inf5_eta0=-0.2.pkl')
bog = BogaczAntiHebb(torch.randn(Nh,Nh), eta=0.6)

Ts = np.logspace(6,12,20,base=2,dtype=int)
Rs = [5, 20]
for R in Rs:
    hebbAcc = []
    bogAcc = []
    chance = []
    for T in Ts:
        if R >= T:
            hebbAcc.append( np.nan )
            bogAcc.append( np.nan )
            continue
        data = generate_recog_data(T=T, R=R, d=100, P=0.5, interleave=True, multiRep=True)
        chance.append( recog_chance(data) )
        
        hebbAcc.append( net.accuracy(data.tensors) )
        
        bog.reset()
        out = bog.forward(data.tensors[0])
        tgt = data.tensors[1].bool().flatten()
        acc, tp, fp, tn, fn = bog.accuracy(out, tgt)
        bogAcc.append( acc )
        
        print 'T={}, R={}, acc={:.2f}, chance={}'.format(T, R, acc, chance[-1])
   
    if R == 20:
        ls = '--'
    else:
        ls = '-'
    ax.semilogx(Ts, hebbAcc, color='blue', linestyle=ls, label='HebbFF, $R_{{test}}$={}'.format(R))
    ax.semilogx(Ts, bogAcc,  color='red',  linestyle=ls, label='From [6], $R_{{test}}$={}'.format(R))
#    ax.semilogx(Ts, chance, 'k--', label='Chance')
    

ax.legend()
ax.set_xlabel('Timepoints in dataset (T)')
ax.set_ylabel('Accuracy')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

labels = ax.get_xticklabels()
labels[2].set_text('$P* \approx 10^2$')
ax.set_xticklabels(labels)

ax.get_figure().set_size_inches(3.75, 2)
ax.get_figure().tight_layout()  

    
    
    
    
    
    
#%% Fig 4. Comparison across architectures and human performance
d = 25
Nh = 250
dims = [d,Nh,1]
gen_data = lambda R: generate_recog_data(T=R*20, d=d, R=R, P=0.5, interleave=True, multiRep=True)


# Human performance (from Brady et al. 2008)
R = [1,2,4,8,16,32,64,128,256,512,1024]
acc = [1, .99, .99, .99, .98, .96, .96, .92, .88, .83, .79]
fig,ax = plt.subplots()
ax.semilogx(R,acc, label='Human (from [8])')


net = load_from_file('HebbNet[{},{},1]_train=cur5_incr=plus1.pkl'.format(d,Nh))
#net = load_from_file('HebbNet[{},{},1]_train=cur2_incr=plus1_eta0=-0.2.pkl'.format(d,Nh))
ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label='HebbFF')


net = HebbRecogDecoupledManualSequential(dims)
def update_hebb(self, pre, post):
    if self.plastic:
        self.A[post==1] = 0
        self.A = self.lam*self.A + self.eta*torch.ger(post,pre)
net.update_hebb = types.MethodType(update_hebb, net) 
net._alpha.data = torch.tensor(-float('inf'))
net.lam.data = torch.tensor(1.)
net.eta.data = torch.tensor(1.)
net.w1.data = torch.zeros_like(net.w1)
net.b1.data = (-d+1)*torch.ones_like(net.b1)
net.w2.data = 10*torch.ones_like(net.w2)
net.b2.data = torch.tensor([-5.])
ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label='Seq. one-hot')


net = HebbRecogDecoupledManual(dims)
def update_hebb(self, pre, post):
    if self.plastic:
        self.A[post==1] = 0
        self.A = self.lam*self.A + self.eta*torch.ger(post,pre)
net.update_hebb = types.MethodType(update_hebb, net) 
net._alpha.data = torch.tensor(-float('inf'))
net.lam.data = torch.tensor(1.)
net.eta.data = torch.tensor(1.)
net.w1.data = torch.zeros_like(net.w1)
net.b1.data = (-d+1)*torch.ones_like(net.b1)
net.w2.data = 10*torch.ones_like(net.w2)
net.b2.data = torch.tensor([-5.])
ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label='Rand. one-hot')


ax.set_xlabel('$R_{test}$')
ax.set_ylabel('Accuracy')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.set_size_inches(3.75, 2)
fig.tight_layout()


