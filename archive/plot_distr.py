import os

import numpy as np
import brokenaxes
import torch
torch.set_printoptions(precision=4, sci_mode=False, threshold=5000)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rc('savefig', format='pdf', bbox='tight')
plt.rc('font', size=8)
#plt.rc('legend', fontsize=8)  

from data import generate_recog_data, recog_chance
from net_utils import load_from_file
from plotting import plot_output_distr, ticks_off, plot_recog_generalization, plot_W, plot_B, plot_multi_hist




#%% Histograms for 25,50 fully trained randn-init network
#d = 25
#Nh = 50
#name = 'HebbNet[{},{},1]_train=cur2_incr=plus1.pkl'.format(d,Nh)
#folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-11-15/shuffle/'
#fname = folder+name
#net = load_from_file(fname)
#ax, Rmp, Rmc = plot_multi_hist(net, Rmp=20, Rmc=75)

#%% Plot distributions for shuffled (+trained) W1 matrices

#files = [
##         ('HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl',    'randn, vec' ),
##         ('HebbNet[25,25,1]_w1init=+-diag_b1init=randn_w2init=negs.pkl',    '+-diag, vec' ),
##         ('HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl',  'randn, scal'),
#         ('HebbNet[25,25,1]_w1init=+-diag_b1init=scalar_w2init=scalar.pkl', '+-diag, scal'),
##         ('HebbNet[25,25,1]_train=cur2_incr=plus1.pkl (0)',                  'randn, vec (2)' )                     
#         ]
#for i, (fnamebase, label) in enumerate(files):    
#    net = load_from_file(fnamebase)
#    
#    ax, Rmp, Rmc = plot_multi_hist(net, Rmp=net.hist['increment_R'][-1][1]-1, title='{}, Rmax={}, ($\lambda$={:.2f} $\eta$={:.2f})'.format(label, net.hist['increment_R'][-1][1], net.lam, net.eta))
##    fig = plt.gcf()
##    fig.savefig('{}_hist.pdf'.format(label.replace('+-','').replace(', ', '_')))
##    plt.close(fig)
#    
#    net.w1.data = net.w1.data.flatten()[torch.randperm(net.w1.data.shape.numel())].reshape(net.w1.data.shape) 
#    ax, Rmp, Rmc = plot_multi_hist(net, title='{} shuf'.format(label))
#    
##    fig = plt.gcf()
##    fig.savefig('{}_shuf_hist.pdf'.format(label.replace('+-','').replace(', ', '_')))
##    plt.close(fig)
#
#    for R0 in ['2', '10']:
#        for shuf in ['shuffle']:#, 'shuffle-c', 'shuffle-r']:
#            if label.endswith('scal'):
#                b1init = 'scalar0'
#                w2init = 'scalar-1'
#            elif label.endswith('vec'):
#                b1init = w2init = 'randn'
#            fname = '{}_train=cur{}_incr=plus1_w1init={}_b1init={}_w2init={}_freeze=w1.pkl'.format(fnamebase[:-4], R0, shuf, b1init, w2init)
#            label_ = '{} {}, $R_0$={}, ($\lambda$={:.2f} $\eta$={:.2f})'.format(label, shuf.replace('fle', '').replace('-', '_'), R0, net.lam, net.eta)
#            net = load_from_file(fname)
#            ax, Rmp, Rmc = plot_multi_hist(net, title=label_)
#    
##            fig = plt.gcf()
##            fig.savefig('{}_shuf+cur{}_hist.pdf'.format(label.replace('+-','').replace(', ', '_'), R0))
##            plt.close(fig)
#
#
##%% PLot distr for frozen W1 matrices (b1, W2, b2 trained) or diag-init matrices (fully trained)
#
#files = list(filter(lambda f: f.find('.pkl')>0, os.walk('.').next()[2]))
#for f in files:    
#    net = load_from_file(f)
##    label = f[f.find('_w1init=')+8:f.find('_freeze')]   #frozen
#    label = f[f.find('_w1init=')+8:f.find('_b1init=')]   #diag-init   
#    title='{}, Rmax={}, ($\lambda$={:.2f} $\eta$={:.2f})'.format(label, net.hist['increment_R'][-1][1], net.lam, net.eta)
#    ax, Rmp, Rmc = plot_multi_hist(net, Rmp=net.hist['increment_R'][-1][1]-1, title=title)    
#
#    plt.gcf().savefig('{}_hist.pdf'.format(label))
#    plt.close()
#           
#
#
##%% Plot histogram for each hidden unit
#fname = 'HebbNet[25,25,1]_train=cur2_incr=plus1.pkl (0)'
#net = load_from_file(fname)
#Rmax = net.hist['increment_R'][-1][1]
#title = 'randn $R_{{max}}$={}'.format(Rmax)
#ax, Rmp, Rmc = plot_multi_hist(net, Rmp=Rmax-1, title=title)    
#plot_per_unit_hist(net, R=Rmp, title=title)
#plot_per_unit_hist(net, R=Rmc, title=title)
#
##%%
##fname = 'HebbNet[25,25,1]_w1init=+-diag_b1init=scalar_w2init=scalar.pkl'
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl'
#net = load_from_file(fname)
#Rmax = net.hist['increment_R'][-1][1]
#title = 'randn_vec, $R_{{max}}$={}'.format(Rmax)
#a1Fig, hFig = plot_per_unit_hist(net, R=Rmax-1, title=title)
#a1Fig.savefig('randn_vec_units_a1_hist.pdf')
#hFig.savefig('randn_vec_units_h_hist.pdf')
#
#net.w1.data = net.w1.data.flatten()[torch.randperm(net.w1.data.shape.numel())].reshape(net.w1.data.shape) 
#title = 'randn_vec, shuffle'.format(Rmax)
#a1Fig, hFig = plot_per_unit_hist(net, R=1, title=title)
#a1Fig.savefig('randn_vec_shuf_units_a1_hist.pdf')
#hFig.savefig('randn_vec_shuf_units_h_hist.pdf')
#
##%%
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl'
#net = load_from_file(fname)
#ax, Rmp, Rmc = plot_multi_hist(net, Rmp=12, Rmc=84)
#
##%%
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl'
#net = load_from_file(fname)
#ax, Rmp, Rmc = plot_multi_hist(net, Rmp=12, Rmc=84)

#%%
#os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-11-15/w1_gain/w2_full')
#fname = 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=diag0+-.pkl'
os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2019-11-15/shuffle')
fname = 'HebbNet[25,25,1]_w1init=+-diag_b1init=randn_w2init=negs.pkl'
#fname = 'HebbNet[25,25,1]_w1init=+-diag_b1init=scalar_w2init=scalar.pkl'

#savefolder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-03-24/w1_indep_diags'   
#os.chdir(savefolder)


colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
for c,Ng in enumerate([2,3,4,5,6,10,25]):
    fname = 'HebbDiags[25,25,1]_train=cur2_incr=plus1_Ng={}.pkl'.format(Ng)
    try:
        os.makedirs('{}/Ng={}'.format(savefolder, Ng))
    except:
        pass
    
    net = load_from_file(fname)
    Nh,d = net.w1.shape
    R = net.hist['increment_R'][-1][1]
    gen_data = lambda R: generate_recog_data(T=R*50, d=d, R=R, P=0.5, interleave=True, multiRep=True)

    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt='{}-'.format(colors[c]), label='Ng={}'.format(Ng))
   
    for i in range(len(net.gList)): #shuffle gList
        tmp = net.gList[i]    
        j = np.random.choice(range(i) + range(i+1, len(net.gList)))
        net.gList[i] = net.gList[j]
        net.gList[j] = tmp
    
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt='{}--'.format(colors[c]), label='Ng={}'.format(Ng))
#    fig, axs = plot_W([net.w1.detach()])
#    ax = axs[0,0]
#    ax.set_title(ax.get_title() + '\n$R_{{max}}=${}'.format(net.hist['increment_R'][-1][1]))
#    fig.set_size_inches([3, 4.8])
#    plt.savefig('{}/Ng={}/w1.pdf'.format(savefolder, Ng))
#    plot_W([net.w2.detach()])
#    plt.savefig('{}/Ng={}/w2.pdf'.format(savefolder, Ng))
#    plot_B([net.b1.detach(), net.b2.detach()])
#    plt.savefig('{}/Ng={}/B.pdf'.format(savefolder, Ng))
#
##    w = net.w1.detach()
#    
#    ax, Rmp, Rmc = plot_multi_hist(net, Rmp=net.hist['increment_R'][-1][1], Rmc=60, title='Ng={}'.format(Ng))
#    plt.savefig('{}/Ng={}/hist.pdf'.format(savefolder, Ng))

#net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 
#ax, Rmp, Rmc = plot_multi_hist(net, Rmc=64, title='shuf')#, Rmp=14, Rmc=104)
#
#for i,row in enumerate(w):
#    net.w1.data[i] = row[torch.randperm(d)]
#ax, Rmp, Rmc = plot_multi_hist(net, Rmc=64, title='shuf_r')#, Rmp=14, Rmc=104)
#
#for j,col in enumerate(w.t()):
#    net.w1.data[:,j] = col[torch.randperm(Nh)]
#ax, Rmp, Rmc = plot_multi_hist(net, Rmc=64, title='shuf_c')#, Rmp=14, Rmc=104)    



