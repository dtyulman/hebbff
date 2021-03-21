import numpy as np
import matplotlib.pyplot as plt

from data import prob_repeat_to_frac_novel, generate_recog_data
from networks import DetHebb, DetHebbNoSplit


#%% Get generalization performance 
Rtest = np.unique(np.logspace(0, 3, 20, dtype=int))
multiRep=False
P = 0.5 
f = prob_repeat_to_frac_novel(P, multiRep=multiRep)

n = 9
D = 500
noSplit=False
if noSplit:
    d = D
    D = np.nan
    net = DetHebbNoSplit(d, n, f=f, Ptp=0.99, Pfp = 0.01)
else:
    d = D+n
    net = DetHebb(D, n, f=f, Ptp=0.99, Pfp = 0.01)


Ptp = np.zeros(len(Rtest))
Pfp = np.zeros(len(Rtest))
net.evaluate(generate_recog_data(T=5000, d=d, R=5000).tensors[0]) #burn-in A to get to steady-state
for i,R in enumerate(Rtest):    
    X,Y = [xy.numpy() for xy in generate_recog_data(T=max(1000, R*50), d=d, R=R, P=P, multiRep=multiRep).tensors]
    f = (1 - sum(Y)/len(Y))[0]    
      
    Yhat = net.evaluate(X)
    Ptp[i], Pfp[i] = net.true_false_pos(Y, Yhat)
    acc = net.accuracy(Y, Yhat)
    acc2 = (1-f)*Ptp[i] + f*(1-Pfp[i]) #sanity check   
    print('R={}, f={:.3f}, Ptp={:.3f}, Pfp={:.3f}, acc={:.4f}={:.4f}'.format(R, f, Ptp[i], Pfp[i], acc, acc2))
#    print('R={}, acc={:.4f}'.format(R, acc))
    
#print('Calculating analytic...')
#PtpAna, PfpAna = net.true_false_pos_analytic(Rtest, corrected=True)

#%% Plot generalization perf
#fig,ax = plt.subplots()
line1 = ax.semilogx(Rtest, Ptp, marker='o', ls='', label='true pos (model)')[0]
line2 = ax.semilogx(Rtest, Pfp, marker='o', ls='', label='false pos (model)')[0]


#line1 = ax.semilogx(Rtest, Ptp, label='TP'+'(no split)' if noSplit else '')[0]
#line2 = ax.semilogx(Rtest, Pfp, color=line1.get_color(), ls='--', label='FP'+'(no split)' if noSplit else '')[0]

fig,ax = plt.subplots()
ax.semilogx(Rtest, (1-f)*Ptp + f*(1-Pfp), label='NoSplit')


#ax.semilogx(Rtest, PtpAna, color=line1.get_color(), label='true pos (analytic)')
#ax.semilogx(Rtest, PfpAna, color=line2.get_color(), label='false pos (analytic)')

ax.set_title('d={}, N={}, f={:.2f}, multiRep={}\n$P_{{fp}}$={:.2f}, $P_{{tp}}$={:.2f}, $\\alpha$={:.2f}, $\gamma$={:.2f}, b={:.2f},'.format(net.d, net.N, net.f, multiRep, net.Pfp, net.Ptp, net.a, net.gam, net.b))
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('True/false positive rate')
ax.legend()


#%% Empirically calculate capacity
noSplit=False

multiRep=False
P = 0.5 
f = prob_repeat_to_frac_novel(P, multiRep=multiRep)

ds = [100, 200]
ns = range(0,8)
Ns = 2**np.array(ns)
Rmax = np.zeros((len(ds),len(Ns)))*np.nan
for i,d in enumerate(ds):
    for j,n in enumerate(ns):
        N = 2**n
        if d*N > 2*10**5:
            continue             
        if noSplit:
            D = np.nan
            d = D
            net = DetHebbNoSplit(d, n, f=f, Ptp=0.99, Pfp = 0.01)
        else:
            D = d-n #d = D+n
            net = DetHebb(D, n, f=f, Ptp=0.995, Pfp = 0.005)     
        print('\n--- d={}, D={}, N={} ---'.format(d, D, N))
    
        #binary search for Rmax
        Rlo = 1
        Rhi = 2*int(1 + 0.0085*N*D/f)               
        while Rhi-Rlo > 1:
            R = (Rlo+Rhi)/2                       
            X,Y = [xy.numpy() for xy in generate_recog_data(T=max(10000, R*100), d=d, R=R, P=P, multiRep=multiRep).tensors]
            Yhat = net.evaluate(X)
            acc = net.accuracy(Y, Yhat)
            print('Rlo={}, Rhi={}, R={}, acc={}'.format(Rlo, Rhi, R, acc))

            if acc >= 0.985:
                Rlo = R
            else:
                Rhi = R
                
            if R==1:
                R = 0
                break
        Rmax[i,j] = R


#%% Correlation bw Ax and Wx vs R
from plotting import ticks_off

corrOrCov = 'corr'# 'cov'#

multiRep=False
P = 0.5 
f = prob_repeat_to_frac_novel(P)

n = 3  
noSplit=False
if noSplit:
    D = np.nan
    d = 200
    net = DetHebbNoSplit(d, n, f=f, Ptp=0.99, Pfp = 0.01)
else:
    D = 200
    d = D+n
    net = DetHebb(D, n, f=f, Ptp=0.99, Pfp = 0.01)     
Nh = 2**n

Rs = [1,2,5,10,20,50,100,200,500,1000,2000]

fig,axs = plt.subplots(2,1)
for a,novOrFam in enumerate(['novel', 'familiar']):
    corrAvg = np.zeros(len(Rs))
    corrStd = np.zeros(len(Rs))
    for i,R in enumerate(Rs):   
        print('R={}'.format(R))
        T=max(R*40,4000)
        data = [xy.numpy() for xy in generate_recog_data(T=T, d=d, R=R, P=P, multiRep=multiRep).tensors]
        
        a1 = np.empty((T,Nh))
        h = np.empty((T,Nh))
        Wxb = np.empty((T,Nh))
        Ax = np.empty((T,Nh))
        out = np.empty_like(data[1])
        for t,(x,y) in enumerate(zip(*data)):
            a1[t], h[t], Ax[t], Wxb[t], out[t] = net.forward(x, debug=True)      
        acc = net.accuracy(data[1], out)
        
        novelIdx = (data[1]==0).squeeze()
        if novOrFam == 'familiar':
            novelIdx = ~novelIdx
        
        if corrOrCov == 'cov':
            corr = np.cov(Ax[novelIdx,:], Wxb[novelIdx,:], rowvar=False)       
        elif corrOrCov == 'corr':
            corr = np.corrcoef(Ax[novelIdx,:], Wxb[novelIdx,:], rowvar=False)       
            
        corr = np.corrcoef(Ax[novelIdx,:], Wxb[novelIdx,:], rowvar=False)       
        corr_hA_hW = np.diag(corr[:Nh, Nh:,]) 
        corrAvg[i] = corr_hA_hW.mean()
        corrStd[i] = corr_hA_hW.std()
    
    #    if R in [1,14,50,100,1000]:
    #        fig,ax = plt.subplots()
    #        im = ax.matshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    #        fig.colorbar(im)
    #        ax.axvline(Nh-0.5, color='k')
    #        ax.axhline(Nh-0.5, color='k')
    #        ticks_off(ax)
    #        ax.set_title('R={}, ${}(h_A,h_W)$={:.3f}$\pm${:.3f}'.format(corrOrCov, R, corrAvg[i], corrStd[i]) )
    #        fig.savefig('corrmat_{}_R={}'.format(novOrFam, R))
            
    ax = axs[a]
    line = ax.semilogx(Rs, corrAvg, color='red')[0]
    ax.fill_between(Rs, corrAvg+corrStd, corrAvg-corrStd, alpha=0.5, color=line.get_color())
    ax.set_xlabel('$R_{test}$')
    ax.set_ylabel(corrOrCov)
    ax.set_title('{}(Wx+b, Ax), averaged across units, {} only'.format(corrOrCov, novOrFam))
    fig.set_size_inches(7,6)
    fig.tight_layout()

#%%
import torch
from plotting import plot_multi_hist
from torch.utils.data import TensorDataset

D = 200
n = 8 
d = D+n
P = 0.5
f = prob_repeat_to_frac_novel(P)

net = DetHebb(D, n, f=f)
gen_data = lambda R,T: generate_recog_data(T=T, d=d, R=R, P=P, multiRep=False)
plot_multi_hist(net, gen_data, Rmp=1000, Rmc=5000)


#%%

for m in range(1,20):
    for n in range(1,m):
        P = float(n)/float(m)
        f = float(P.as_integer_ratio()[0])/sum(P.as_integer_ratio())
        femp = generate_recog_data(T=10000, d=1, R=1, P=P, multiRep=False).tensors[1].mean()     
        print('P = {:.2f}/{:.2f}={:.2f}, f={:.2f}={:.2f}'.format(n,m,P,femp,f))
        
#%%







