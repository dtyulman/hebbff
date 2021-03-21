import numpy as np
import torch
torch.set_printoptions(precision=4, sci_mode=False, threshold=5000)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rc('savefig', format='pdf', bbox='tight')
plt.rc('font', size=8)
#plt.rc('legend', fontsize=8)  

from data import generate_recog_data, recog_chance
from net_utils import load_from_file

#%%
files = [
         ('HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl',    'randn, vec' ),
         ('HebbNet[25,25,1]_w1init=+-diag_b1init=randn_w2init=negs.pkl',    '+-diag, vec' ),
         ('HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl',  'randn, scal'),
         ('HebbNet[25,25,1]_w1init=+-diag_b1init=scalar_w2init=scalar.pkl', '+-diag, scal'),                     
         ]

for fname, label in files:
#    fname = files[2][0]
net = load_from_file(fname)
R = net.hist['increment_R'][-1][1]-2
d,Nh = net.w1.shape

T = 200
#data = generate_recog_data(T=T, d=d, R=R)

#%%
fig,axs = plt.subplots(6,1, sharex=True) 
for tf in [False, True]:
    if tf:
        w = net.w1.detach()
        net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 

#%%
    with torch.no_grad():
        #burn-in net to get A to steady-state
        for t,(x,y) in enumerate(generate_recog_data(T=1000, d=d, R=R)):
            net.forward(x, debug=True)
            
        #run and log net vars     
        a1 = torch.empty(T,Nh)
        h = torch.empty(T,Nh)
        a2 = torch.empty_like(data.tensors[1])
        out = torch.empty_like(data.tensors[1])
        A = torch.empty(T,d,Nh)
        deltaA = torch.empty(T,d,Nh)
        for t,(x,y) in enumerate(data):
            A[t] = net.A     
            a1[t], h[t], a2[t], out[t] = net.forward(x, debug=True)  
            deltaA[t] = A[t]-net.A
        acc = net.accuracy(data.tensors)
    
    # Calculate corr
    for method in ['cov']:#['Pearson', 'dot', 'cov']:
        with torch.no_grad():
            t0 = 100
            x0 = data.tensors[0][t0]
            tEqual = []
            corr = {c : np.empty(T) for c in ['x', 'a1', 'h', 'deltaA', 'Wxb', 'Ax']}
            for t,(x,y) in enumerate(data):
                if (x==x0).all():
                    tEqual.append(t)
                
                if method == 'Pearson':
                    corr['x'][t] = np.corrcoef(x0, x)[0][1]
                    corr['h'][t] = np.corrcoef(h[t0], h[t])[0][1]
                    corr['a1'][t] = np.corrcoef(a1[t0], a1[t])[0][1]
                    corr['deltaA'][t] = np.corrcoef(deltaA[t0].flatten(), deltaA[t].flatten())[0][1]
                    corr['Wxb'][t] = np.corrcoef(torch.addmv(net.b1, net.w1, x), torch.addmv(net.b1, net.w1, x0))[0][1]
                    corr['Ax'][t] = np.corrcoef(torch.mv(A[t], x), torch.mv(A[t0], x0))[0][1]
        
                elif method == 'dot':
                    corr['x'][t] = np.dot(x0, x)
                    corr['h'][t] = np.dot(h[t0], h[t])
                    corr['a1'][t] = np.dot(a1[t0], a1[t])
                    corr['deltaA'][t] = np.dot(deltaA[t0].flatten(), deltaA[t].flatten())
                    corr['Wxb'][t] = np.dot(torch.addmv(net.b1, net.w1, x), torch.addmv(net.b1, net.w1, x0))
                    corr['Ax'][t] = np.dot(torch.mv(A[t], x), torch.mv(A[t0], x0))
        
                elif method == 'cov':
                    corr['x'][t] = np.cov(x0, x)[0][1]
                    corr['h'][t] = np.cov(h[t0], h[t])[0][1]
                    corr['a1'][t] = np.cov(a1[t0], a1[t])[0][1]
                    corr['deltaA'][t] = np.cov(deltaA[t0].flatten(), deltaA[t].flatten())[0][1]
                    corr['Wxb'][t] = np.cov(torch.addmv(net.b1, net.w1, x), torch.addmv(net.b1, net.w1, x0))[0][1]
                    corr['Ax'][t] = np.cov(torch.mv(A[t], x), torch.mv(A[t0], x0))[0][1]
        
        # Plot corr  
        toPlot = ['x', 'a1', 'h', 'deltaA', 'Wxb', 'Ax']
        ts = np.arange(T)
        for i,(c, ax) in enumerate(zip(toPlot, axs.flat)):  
            ax.plot(ts, corr[c], color='red' if tf else 'blue')
            ax.set_ylabel(c)
            if tf:
                ax.plot(tEqual, np.ones_like(tEqual)*max(corr[c]), marker='*', linestyle='')
        axs.flat[-1].set_xlabel('t')  
        title = '{}_{}{}'.format(method, label.replace(', ', '_'), '_shuf' if tf else '')
        axs.flat[0].set_title(title)  
        
fig.set_size_inches(7,6.3)
fig.tight_layout()
fig.savefig(title)
plt.close()
    
    
