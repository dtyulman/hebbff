import os

import matplotlib.pyplot as plt
plt.rc('savefig', format='pdf', bbox='tight')
plt.rc('legend', fontsize=8)    # legend fontsize

import torch

from data import  generate_recog_data
from plotting import plot_train_perf, plot_W, plot_B, plot_recog_generalization, maxabs
from networks import HebbNet
from net_utils import binary_classifier_accuracy, load_from_file
from dt_utils import subplots_square
#%%
d=25  
gen_data = lambda R: generate_recog_data(T=R*50, d=d, R=R, P=0.5, interleave=True, multiRep=True)

files = [
         ('HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl',    'randn, vec' ),
         ('HebbNet[25,25,1]_w1init=+-diag_b1init=randn_w2init=negs.pkl',    '+-diag, vec' ),
         ('HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl',  'randn, scal'),
         ('HebbNet[25,25,1]_w1init=+-diag_b1init=scalar_w2init=scalar.pkl', '+-diag, scal'),                     
         ]

#%% Plot untrained shuffled  
#for i,(fname, label) in enumerate(files):      
    Gax = None   
    net = load_from_file(fname)    
#    label='{} ($\lambda=${:.2f} $\eta=${:.2f})'.format(label, net.lam, net.eta)
            
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='base')
#    Gax.set_title(label)
    
    #Plot various permutations
    w = net.w1.detach()
    
#    net.w1.data = -w
#    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='neg ')
    
    net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf ')
    
    for i,row in enumerate(w):
        net.w1.data[i] = row[torch.randperm(w.shape[0])]
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf_r ')
    
    for j,col in enumerate(w.t()):
        net.w1.data[:,j] = col[torch.randperm(w.shape[1])]
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf_c ')
   
    net.w1.data = w[torch.randperm(w.shape[0]),:]  
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='rows ')
    
    net.w1.data = w[:,torch.randperm(w.shape[1])]       
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='cols ')
    
    net.w1.data = w[torch.randperm(w.shape[0]),:][:,torch.randperm(w.shape[1])]       
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='rowcol ')
        
    plt.gcf().tight_layout()

#%% Plot base performance
Gax = None
for i,(fname, label) in enumerate(files): 
    net = load_from_file(fname)         
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label=label)
    
    
#%% Plot SVD
_,Sax = plt.subplots()
for i,(fname, label) in enumerate(files): 
    net = load_from_file(fname)         
    print net.hist['increment_R'][-1]
    
#    plot_W([net.w1.detach()])
#    plt.gcf().axes[0].set_title(label)
  
    u,s,v = torch.svd(net.w1.data)
    plot_W([u,v])
    plt.gcf().axes[0].set_title('U')
    plt.gcf().axes[1].set_title('V')
    plt.gcf().axes[2].set_title(label)    
    
    Sax.plot(s, label=label)
    Sax.set_title('S')
    Sax.set_xlabel('i')
    Sax.set_ylabel('$\lambda_i$')
Sax.legend()
    
    
#_,Rax = plt.subplots()
#    Rax.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), linestyle=fmt, label=label)
#Rax.legend()
#Rax.set_xlabel('Iter')
#Rax.set_ylabel('R')
    
         
#%% Plot row/column sums       
plot_W([net.w1.detach()])
plt.title(label)

print label
print 'row sums ', net.w1.sum(0).sort()[0]
print 'col sums ', net.w1.sum(1).sort()[0]

w = net.w1.detach()
fig, ax = plt.subplots(2,2)
wIm = ax[0,0].matshow(w, cmap='RdBu_r')
ax[0,0].set_title(label)
fig.colorbar(wIm, label='w', orientation='horizontal')
rowSum = ax[0,1].matshow(w.sum(1).unsqueeze(1), cmap='RdBu_r')
fig.colorbar(rowSum, label='r', orientation='horizontal')
colSum = ax[1,0].matshow(w.sum(0).unsqueeze(0), cmap='RdBu_r')
fig.colorbar(colSum, label='c', orientation='horizontal')
for a in ax.flat:
    a.axis('off')  


#%% Plot shuffled post training    
for i,(fnamebase, label) in enumerate(files): 
    net = load_from_file(fnamebase)
    
    ax = None
    ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, fmt='--', label='base')
    
    net.w1.data = net.w1.data.flatten()[torch.randperm(net.w1.data.shape.numel())].reshape(net.w1.data.shape) 
    ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, fmt='--',  label='shuf')
    
    base = fnamebase[:-4] 
    for R0 in ['2', '10']:
        for shuf in ['shuffle', 'shuffle-c', 'shuffle-r']:
            if label.endswith('scal'):
                b1init = 'scalar0'
                w2init = 'scalar-1'
            elif label.endswith('vec'):
                b1init = w2init = 'randn'
            fname = '{}_train=cur{}_incr=plus1_w1init={}_b1init={}_w2init={}_freeze=w1.pkl'.format(base, R0, shuf, b1init, w2init)
            label_ = '{}, $R_0$={}, ($\lambda$={:.2f} $\eta$={:.2f})'.format(shuf.replace('fle', '').replace('-', '_'), R0, net.lam, net.eta)
            net = load_from_file(fname)
            ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label=label_)
    ax.set_title(label)
            
#%% Plot histograms for shuffle

for i,(fnamebase, label) in enumerate(files[0]): 
    net = load_from_file(fnamebase)
    
    ax = None
    ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, fmt='--', label='base')
    
    net.w1.data = net.w1.data.flatten()[torch.randperm(net.w1.data.shape.numel())].reshape(net.w1.data.shape) 
    ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, fmt='--',  label='shuf')
    
    base = fnamebase[:-4] 
    for R0 in ['2', '10']:
        for shuf in ['shuffle', 'shuffle-c', 'shuffle-r']:
            if label.endswith('scal'):
                b1init = 'scalar0'
                w2init = 'scalar-1'
            elif label.endswith('vec'):
                b1init = w2init = 'randn'
            fname = '{}_train=cur{}_incr=plus1_w1init={}_b1init={}_w2init={}_freeze=w1.pkl'.format(base, R0, shuf, b1init, w2init)
            label_ = '{}, $R_0$={}, ($\lambda$={:.2f} $\eta$={:.2f})'.format(shuf.replace('fle', '').replace('-', '_'), R0, net.lam, net.eta)
            net = load_from_file(fname)
            ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label=label_)
    ax.set_title(label)   
          




    