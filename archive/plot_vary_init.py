import os

import matplotlib.pyplot as plt
plt.rc('savefig', format='pdf')
plt.rc('axes', titlesize=8)     # fontsize of the axes title        
plt.rc('legend', fontsize=8)    # legend fontsize
import torch
import torch.nn as nn

from data import  generate_recog_data, recog_chance
from plotting import plot_train_perf, plot_W, plot_B, plot_recog_generalization, maxabs
from networks import HebbNet
from net_utils import binary_classifier_accuracy, load_from_file

from dt_utils import subplots_square

#%%
d=25  
gen_data = lambda R: generate_recog_data(T=R*20, d=d, R=R, P=0.5, interleave=True, multiRep=True)

fnames = list(filter(lambda f: f.find('.pkl')>0, os.walk('.').next()[2]))
labels = []
for fname in fnames:
    r = fname.find('w1init=')+len('w1init=')
    l = r + (fname[r:].find('_') if fname[r:].find('_')>=0 else fname[r:].find('.')) 
    labels.append( fname[r:l].replace('diag', '') )

##%%        
#ax1=ax2=ax3=None
#for i,(fname, label) in enumerate(zip(fnames, labels)):      
#    net = load_from_file(fname)  
#    label_ = '{} ($\lambda$={:.2f} $\eta$={:.2f}, R={})'.format(label, net.lam, net.eta, net.hist['increment_R'][-1][1])
#    if label.count('0') == 0:
#        ax3,_,_ = plot_recog_generalization(net, gen_data, ax=ax3, label=label_)
#    elif label.count('0') == 1:
#        ax2,_,_ = plot_recog_generalization(net, gen_data, ax=ax2, label=label_)
#    elif label.count('0') >= 2:
#        ax1,_,_ = plot_recog_generalization(net, gen_data, ax=ax1, label=label_)
#    else:
#        raise Exception
#
##%%
#ax=None
#for i,(fname, label) in enumerate(zip(fnames, labels)):  
#    net = load_from_file(fname)  
#    label_ = '{} ($\lambda$={:.2f} $\eta$={:.2f}, R={})'.format(label, net.lam, net.eta, net.hist['increment_R'][-1][1])    
#    if i%12 == 0 and i<49:
#        ax = None
#    ax,_,_ = plot_recog_generalization(net, gen_data, ax=ax, label=label_)
#    ax.figure.set_size_inches(10,4)
#    ax.figure.tight_layout()
        
#%%
Ws = []
for fname in fnames:
    net = load_from_file(fname) 
    Ws.append(net.g1.item()*net.w1.detach() )    
v = maxabs(Ws)
    
#fig, ax = subplots_square(len(fnames))
fig, ax = plt.subplots(3,7)        
for (fname, label, a) in zip(fnames, labels, ax.flat):
    net = load_from_file(fname)    
    im = a.matshow(net.g1.item()*net.w1.detach(), cmap='RdBu_r', vmin=-v, vmax=v)
#    a.set_title('{} ($g_1$={:.1f} $b_1$={:.1f}\n$w_2$={:.1f} $b_2$={:.1f}) $R_{{max}}={}$'.format(label, net.g1.item(), net.b1.item(), net.w2.item(), net.b2.item(), net.hist['increment_R'][-1][1]))
    a.set_title('{} ($g_1$={:.1f})\n$R_{{max}}={}$'.format(label, net.g1.item(), net.hist['increment_R'][-1][1]))
#    a.set_title(label)

[a.axis('off') for a in ax.flat]
plt.colorbar(im, ax=ax.flat[-1])        
          
fig.set_size_inches([8.96, 8.56])
fig.tight_layout()