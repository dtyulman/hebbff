import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from net_utils import load_from_file
from plotting import get_all_pkl, plot_recog_generalization, plot_recog_positive_rates
from data import generate_recog_data

def plot_fit(X, Y, ax=None, color=None, marker='.', linestyle='--', label='', xlabel='', title=''):
    if ax is None:
        _,ax = plt.subplots()
    
    if color is None: 
        color = ax._get_lines.prop_cycler.next()['color']
    
    X = X[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]
    
    if len(Y) <= 1:
        return ax
    
    if marker is not None:
        ax.loglog(X, Y, color=color, linestyle='', marker=marker) #TODO: if X[i] is a list, take max
    
    k,c,_,_,_=stats.linregress(np.log(X), np.log(Y))
    x = np.linspace(min(X),max(X))
    ax.loglog(x, np.exp(c)*np.power(x,k), linestyle=linestyle, color=color, label=label + ' (k={:.2f}, c={:.2f})'.format(k,c))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$R_{max}$')    
    ax.set_title(title)
    ax.legend()        
    ax.get_figure().tight_layout()    
    
    return ax


#%%
os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-02-15')
fnames = get_all_pkl(exclude='HebbNet[10,')

Nh = [1, 2, 4, 8, 10, 16, 25, 32, 50, 64, 100, 128, 200, 256, 512, 1024, 2048, 4086, 8172] #Should be 4096 but it's too late now
#d = [10, 25, 50, 100, 200]
fnames = [f for f in fnames if f.find('[100,')>=0 or f.find('[200,')>=0]        
d = [100, 200]

Rmax = pd.DataFrame(index=d, columns=Nh, dtype=float)
etas = pd.DataFrame(index=d, columns=Nh, dtype=float)
lams = pd.DataFrame(index=d, columns=Nh, dtype=float)
for fname in fnames:
    net = load_from_file(fname)
    if net.lam < 0 or net.eta > 0:
        continue
    Nh,d = net.w1.shape
    #TODO: if col d or row Nh doesn't exist, insert it, filled with NaN
    Rmax.loc[d][Nh] = net.hist['increment_R'][-1][1] #TODO: append to list  
    etas.loc[d][Nh] = net.eta.item() 
    lams.loc[d][Nh] = net.lam.item() 

#%%
    
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
ls = '-'
mkr = 'o' 
#fig,axs=plt.subplots(1,2)

#% Plot Rmax(Nh) for each value of d
ax=axs[0]
for i,d in enumerate(Rmax.index):
    ax = plot_fit(Rmax.columns.values, Rmax.loc[d].values, ax=ax, color=cycle[i%len(cycle)], marker=mkr, linestyle=ls, label='$d$={}'.format(d), xlabel='$N_h$')    
ax.set_xlabel('Number of hidden units ($N_h$)')
ax.get_figure().tight_layout()  

##% Plot Rmax(d) for each value of Nh
#ax=axs[1]
#for i,Nh in enumerate(Rmax.columns): #.drop(columns=[1,2,8,32])
#    ax = plot_fit(Rmax.index.values, Rmax[Nh].values, ax=ax, color=cycle[i%len(cycle)], marker=mkr, linestyle=ls, label='$N_h$={}'.format(Nh), xlabel='$d$')
#ax.set_xlabel('Number of input units ($d$)')
#ax.get_figure().tight_layout()  

#% Plot Rmax(Nh*d)
R = []
Nsyn = []
for i,Nh in enumerate(Rmax.columns): #.drop(columns=[1,2,8,32])
    for j,d in enumerate(Rmax.index): #.drop(index=[25,50])
        Nsyn.append( d*Nh )
        R.append( Rmax[Nh][d] )
idx = np.argsort(Nsyn)
Nsyn = np.array(Nsyn)[idx]
R = np.array(R)[idx]
ax = plot_fit(Nsyn, R, xlabel='$N_h \cdot d$', color='k', linestyle=ls, marker=None, ax=axs[1])

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  
for i,d in enumerate(Rmax.index):
    ax.loglog(d*Rmax.columns.values, Rmax.loc[d].values, color=cycle[i%len(cycle)], linestyle='', marker=mkr) #TODO: if X[i] is a list, take max

ax.set_xlabel('Number of synapses ($N_h \cdot d$)')
ax.get_figure().tight_layout() 


#%%
#for ax in axs:
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.get_figure().set_size_inches(3.75, 2)

#%% Plot eta as fn of d
_,ax = plt.subplots()   
for d in sorted(etas.index):
    etas_d = etas.loc[d].values[~np.isnan(etas.loc[d].values)]
    ax.plot(d*np.ones(len(etas_d)), d*etas_d, linestyle='', marker='.')
ax.set_xlabel('$d$')
ax.set_ylabel('$\eta \cdot d$')

#%% Plot lambda as fn of d
_,ax = plt.subplots()   
for d in sorted(lams.index):
    lams_d = lams.loc[d].values[~np.isnan(lams.loc[d].values)]
    ax.plot(d*np.ones(len(lams_d)), lams_d, linestyle='', marker='.')
ax.set_xlabel('$d$')
ax.set_ylabel('$\lambda$')


#%% PLot parameters as fn of network size
ds = np.array([25, 50, 100, 200, 400, 800])
Ns = np.logspace(1,9, 9,base=2, dtype=int)
w1 = np.full((len(ds), len(Ns)), np.nan)
w2 = np.full((len(ds), len(Ns)), np.nan)
b1 = np.full((len(ds), len(Ns)), np.nan)
b2 = np.full((len(ds), len(Ns)), np.nan)
eta = np.full((len(ds), len(Ns)), np.nan)
lam = np.full((len(ds), len(Ns)), np.nan)
for i,d in enumerate(ds):
    for j,N in enumerate(Ns):
        fname = 'capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N)
        try:
            net = load_from_file(fname)
            w1[i,j] = net.w1.abs().max()
            w2[i,j] = net.w2.item()
            b1[i,j] = net.b1.item()     
            b2[i,j] = net.b2.item() 
            eta[i,j] = net.eta.item()
            lam[i,j] = net.lam.item()
        except IOError:
            print fname
            
np.set_printoptions(precision=3, linewidth=1000)
print lam
print eta
print w1
print w2
print b1
print b2    

d_mesh =  np.tile(ds, (9,1)).T
N_mesh = np.tile(Ns, (6,1))
Nd_mesh = d_mesh*N_mesh

params = [('lam',lam,  1.       ),
           ('w1', w1,  1./np.log(N_mesh)**.5 ), 
           ('w2', w2,  1./np.log(d_mesh)  ), 
           ('eta',eta, d_mesh**.95            ), 
           ('b1', b1,  1./np.log(N_mesh)**1.1  ), 
           ('b2', b2,  1./lam  )]

fig, ax = plt.subplots(2,3,sharex=False)
ax = ax.flatten()
for i,(lab,mat,norm) in enumerate(params):  
    mat = mat-norm     
    if i == 0:
        ax[i].plot(Ns, mat.T)
    else:
        ax[i].semilogx(Ns, mat.T)
    ax[i].set_ylabel(lab)
    ax[i].set_xlabel('N')    
#ax[-1].legend(ds[:-1])
fig.tight_layout()

#ax[0].plot(Ns, 0.99*np.ones(len(Ns)), 'k--')
#ax[1].plot(Ns, 4.5*np.log(Ns)**.5, 'k--')
#ax[4].plot(Ns, -5*np.log(Ns)**1.1, 'k--')
#ax[5].plot(Ns, 5*np.log(Ns)**0.6, 'k--')


#fig, ax = plt.subplots(2,3,sharex=True)
#ax = ax.flatten()
#for i,(lab,mat,norm) in enumerate(params): 
##    mat = mat*norm                   
#    line = ax[i].semilogx(ds, mat)
#    ax[i].set_ylabel(lab)
#    ax[i].set_xlabel('d')    
##ax[-1].legend(Ns[:-1])
#fig.tight_layout()
#
#ax[2].plot(ds, -4*np.log(ds), 'k--')
#ax[3].plot(ds, -5.75/ds**.95, 'k--')

fig, ax = plt.subplots(2,3,sharex=True,sharey=True)
ax = ax.flatten()
for i,(lab,mat,norm) in enumerate([('lam',lam, 1./Nd_mesh),
                                   ('eta',eta, 1./d_mesh), 
                                   ('w1', w1,  1), 
                                   ('b1', b1,  1), 
                                   ('w2', w2,  1), 
                                   ('b2', b2,  1)]):        
    
    pcm = ax[i].pcolormesh(Ns, ds, mat*norm)
    ax[i].set_title(lab)
    fig.colorbar(pcm, ax=ax[i])
ax[-1].set_xscale('log')
ax[-1].set_yscale('log')
ax[3].set_xlabel('N')
ax[3].set_ylabel('d')




#%% PLot generalization for every single network

fnames = [
 'HebbNet[25,1,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[25,2,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[25,4,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[25,8,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[25,10,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[25,16,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[25,25,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[25,32,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[25,50,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[25,64,1]_train=cur3_incr=plus1.pkl',
 'HebbNet[25,100,1]_train=cur3_incr=plus1.pkl',
 'HebbNet[25,128,1]_train=cur5_incr=plus1.pkl',
 'HebbNet[25,200,1]_train=cur5_incr=plus1.pkl',
 'HebbNet[25,256,1]_train=cur5_incr=plus1.pkl',
 
 'HebbNet[50,1,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[50,2,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[50,4,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[50,8,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[50,10,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[50,16,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[50,25,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[50,32,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[50,50,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[50,64,1]_train=cur3_incr=plus1.pkl',
 'HebbNet[50,100,1]_train=cur3_incr=plus1.pkl',
 'HebbNet[50,128,1]_train=cur5_incr=plus1.pkl',
 'HebbNet[50,200,1]_train=cur5_incr=plus1.pkl',
 
 'HebbNet[100,1,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[100,2,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[100,4,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[100,8,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[100,10,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[100,16,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[100,25,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[100,32,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[100,50,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[100,64,1]_train=cur3_incr=plus1.pkl',
 'HebbNet[100,100,1]_train=cur3_incr=plus1.pkl',
 
 'HebbNet[200,1,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[200,2,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[200,4,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[200,8,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[200,10,1]_train=cur1_incr=plus1.pkl',
 'HebbNet[200,16,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[200,25,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[200,32,1]_train=cur2_incr=plus1.pkl',
 'HebbNet[200,50,1]_train=cur2_incr=plus1.pkl', 
 ]

gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, interleave=True, multiRep=True)
upToR= -float('inf')
axes = {}
for fname in fnames:
    net = load_from_file(fname)
    if net.lam < 0 or net.eta > 0:
        continue
    label = fname[fname.find('['):fname.find(']')+1].replace(',1]','').replace('[','').replace(',','x')
    Nh,d = net.w1.shape
    if d not in axes:
        axes[d] = [None,None]
    axes[d][0],Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, upToR=upToR, ax=axes[d][0], label=label)
    axes[d][1],Rs,acc = plot_recog_generalization(net, gen_data, ax=axes[d][1], testR=Rs, testAcc=acc, label=label)
    axes[d][0].legend_.remove()
    
