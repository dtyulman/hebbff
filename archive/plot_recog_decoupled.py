import math

import matplotlib.pyplot as plt
from matplotlib import gridspec
from torch.utils.tensorboard import SummaryWriter
import torch

from data import generate_recog_data
from net_utils import load_from_file
from plotting import maxval, maxabs, plot_W, plot_B, ticks_off, plot_train_perf, plot_recog_generalization


def evaluate_debug(net, data, folder=None):
    writer = SummaryWriter(folder)
    
    fig, ax = plot_W([net.w1.detach(), net.wP.detach(), net.w2.detach()])
    fig.set_size_inches(6, 2.8)
    ax[0].set_title('$W_{1}$')
    ax[1].set_title('$W_{P}$')
    ax[2].set_title('$W_2^T$')
    writer.add_figure('weight', fig, net.hist['iter'])     
    
    fig, ax = plot_B([net.bP.detach(), net.b1.detach(), net.b2.detach()])
    fig.set_size_inches(1.25, 3)
    ax[0].set_title('$b_{1}$')
    ax[1].set_title('$b_{P}$')
    ax[2].set_title('$b_2$')
    writer.add_figure('bias', fig, net.hist['iter'])
          
    debugLog = []
    print('Running net')
    A_init = net.A.detach()
    for t,x in enumerate(data.tensors[0]): 
        debugLog.append( net(x, debug=True) )
        debugLog[t].update({'A': net.A.detach()})
    debugLog.append({'A': A_init})     
    
    A_vlim = maxabs([db['A'] for db in debugLog])    
    # plot initial A matrix and associated info
    print('Plotting t=-1')
    fig,ax = plt.subplots()
    im = ax.matshow(debugLog[-1]['A'], cmap='RdBu_r', vmin=-A_vlim, vmax=A_vlim)
    ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.02])  
    fig.colorbar(im, cax=cax, orientation='horizontal')
    fig.set_size_inches(3,3)        
    writer.add_figure('A', fig, -1)
    writer.add_scalar('norm_A', debugLog[-1]['A'].norm(), -1)
    writer.add_histogram('A', debugLog[-1]['A'], -1)
    
    
    a_vmax =  maxval([ torch.cat((db['a1W'], db['a1A'], db['a1'], db['aP'])) for db in debugLog[:-1]])
    a_vmin = -maxval([-torch.cat((db['a1W'], db['a1A'], db['a1'], db['aP'])) for db in debugLog[:-1]])
    for t,(x,y) in enumerate(data):
        print('Plotting t={}'.format(t))
        # plot activity at this timepoint
        fig = plt.figure()
        gs = gridspec.GridSpec(4,5)
        x_ax  = plt.subplot(gs[1:3,0])
        a1A_ax = plt.subplot(gs[0:2,1])
        a1W_ax = plt.subplot(gs[0:2,2])
        a1_ax = plt.subplot(gs[0:2,3])
        aP_ax = plt.subplot(gs[2:4,3])
        h1_ax = plt.subplot(gs[0:2,4])
        hP_ax = plt.subplot(gs[2:4,4])
        
        x_ax.matshow(x.reshape(-1,1), cmap='binary') 
        a1_im = a1_ax.matshow(debugLog[t]['a1W'].unsqueeze(1), cmap='Reds', vmin=a_vmin, vmax=a_vmax)
        a1A_ax.matshow(debugLog[t]['a1A'].unsqueeze(1), cmap='Reds', vmin=a_vmin, vmax=a_vmax)
        a1W_ax.matshow(debugLog[t]['a1W'].unsqueeze(1), cmap='Reds', vmin=a_vmin, vmax=a_vmax)
        h1_im = h1_ax.matshow(debugLog[t]['h1'].unsqueeze(1), cmap='Greens', vmin=0, vmax=1)    

            
        aP_ax.matshow(debugLog[t]['aP'].unsqueeze(1), cmap='Reds', vmin=a_vmin, vmax=a_vmax)
        hP_ax.matshow(debugLog[t]['hP'].unsqueeze(1), cmap='Greens', vmin=0, vmax=1)        
        
        x_ax.set_xlabel('x')
        a1_ax.set_ylabel('$a_1$')
        a1A_ax.set_ylabel('$a_{1A}$')
        a1W_ax.set_ylabel('$a_{1W}$')
        aP_ax.set_ylabel('$a_P$')
        h1_ax.set_ylabel('$h_1$')
        hP_ax.set_ylabel('$h_P$')
        for ax in [x_ax, a1_ax, aP_ax, h1_ax, hP_ax, a1A_ax, a1W_ax]:
            ticks_off(ax)
        
        
        target = int(y.item())
        activ = debugLog[t]['a2'].item()
        out = debugLog[t]['y'].round().item()
        recog = debugLog[t]['y'].round().item()
        a1_ax.set_title('$a_2$={:.2f} $\sigma(a_2)$={:.2f}\n recog={} y={}'.format(activ, out, recog, target), 
                        color='Black' if recog==y else 'Red')
        
        fig.set_size_inches(2,5)
        
        fig.subplots_adjust(bottom=0.2)
        cax = fig.add_axes([0.1, 0.1, 0.3, 0.01])  
        fig.colorbar(a1_im, ticks=[math.ceil(a_vmin),math.floor(a_vmax)], cax=cax, label='input', orientation='horizontal') 
        
        cax = fig.add_axes([0.5, 0.1, 0.3, 0.01])  
        fig.colorbar(h1_im, ticks=[0,1], cax=cax, label='activation', orientation='horizontal') 
        
        writer.add_figure('activity', fig, t)
        
        # plot the A matrix and associated info
        fig,ax = plt.subplots()
        im = ax.matshow(debugLog[t]['A'], cmap='RdBu_r', vmin=-A_vlim, vmax=A_vlim)
        ax.axis('off')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        cax = fig.add_axes([0.1, 0.1, 0.8, 0.02])  
        fig.colorbar(im, cax=cax, orientation='horizontal')
        fig.set_size_inches(3,3)        
        writer.add_figure('A', fig, t)
        writer.add_scalar('norm_A', debugLog[t]['A'].norm(), t)
        writer.add_histogram('A', debugLog[t]['A'], t)
    return debugLog, writer
       
#%%
#data params
T = 10  #samples per trial
P = 0.5  #repeat probability
d = 25 #length of item
Nh = 50 #number of hidden unitspl
plt.ioff()


for netClass in ['HebbRecogDecoupled']:
    for R in [5]:  #repeat interval
        for suff in ['']:#, '_(2)']:
            fname = '{}_R={}{}.pkl'.format(netClass, R, suff)
            net = load_from_file(fname, dims=[d,Nh,1])
        
            data = generate_recog_data(T=T, d=d, R=R, P=P)
            folder = 'HebbRecogDecoupled_eval/'+fname[:-4]   
            
            debugLog, writer = evaluate_debug(net, data, folder)
         

#%%
#Nh = 50
#d = 25
#T = 1000
#
#R = 5
#fname = 'HebbRecogDecoupled_R={}.pkl'.format(R)
##fname = 'HebbNet_R={}.pkl'.format(R)
#net = load_from_file(fname, dims=[d, Nh, 1])
#
#data = generate_recog_data(T=T, d=d, R=R, P=0.5)
#print net.accuracy(data.tensors)
#
#


#%%
d = 25
gen_data = lambda R: generate_recog_data(T=R*50, d=d, R=R, P=0.5, interleave=True, multiRep=True)

fname = 'HebbNet_Nh=50_curriculum.pkl'
net = load_from_file(fname, dims=[d, 50, 1])
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='HebbNet50')
ax,_,_ = plot_recog_generalization(net, gen_data, label='HebbNet50')

fname = 'HebbNet_Nh=25_curriculum.pkl'
net = load_from_file(fname, dims=[d, 25, 1])
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='HebbNet25')
ax,_,_ = plot_recog_generalization(net, gen_data, label='HebbNet25', ax=ax)
        
fname = 'HebbRecogDecoupled_Nh=50_curriculum.pkl'
net = load_from_file(fname, dims=[d, 50, 1])
net.plastic = True
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='Decoupled50 $\\alpha=${:.2f} (learn)'.format(torch.sigmoid(net._alpha).item()))
ax,_,_ = plot_recog_generalization(net, gen_data, label='Decoupled50 $\\alpha=${:.2f} (learn)'.format(torch.sigmoid(net._alpha).item()), ax=ax)

fname = 'HebbRecogDecoupled_Nh=100_curriculum.pkl'
net = load_from_file(fname, dims=[d, 100, 1])
net.plastic = True
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='Decoupled100 $\\alpha=${:.2f} (learn)'.format(torch.sigmoid(net._alpha).item()))
ax,_,_ = plot_recog_generalization(net, gen_data, label='Decoupled100 $\\alpha=${:.2f} (learn)', ax=ax)

fname = 'HebbRecogDecoupledManual_Nh=50_curriculum.pkl'
net = load_from_file(fname, dims=[d, 50, 1])
net.plastic = True
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='RandOneHot50 $\\alpha=${:.2f} (learn)'.format(torch.sigmoid(net._alpha).item()))
ax,_,_ = plot_recog_generalization(net, gen_data, label='RandOneHot50 $\\alpha=${:.2f} (learn)'.format(torch.sigmoid(net._alpha).item(), ax=ax))

fname = 'HebbRecogDecoupled_Nh=50_curriculum_alpha=0.pkl'
net = load_from_file(fname, dims=[d, 50, 1])
net.plastic = True
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='Decoupled50 $\\alpha=0$')
ax,_,_ = plot_recog_generalization(net, gen_data, label='Decoupled50 $\\alpha=0$', ax=ax)

fname = 'HebbRecogDecoupledManual_Nh=50_curriculum_alpha=0.pkl'
net = load_from_file(fname, dims=[d, 50, 1])
net.plastic = True
#plt.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), label='RandOneHot50 $\\alpha=0$')
ax,_,_ = plot_recog_generalization(net, gen_data, label='RandOneHot50 $\\alpha=0$', ax=ax)

plt.xlabel('Iter')
plt.ylabel('R')
plt.legend()


#%%
d=25
gen_data = lambda R: generate_recog_data(T=R*25, d=d, R=R, P=0.5, interleave=True, multiRep=True)

_,Rax = plt.subplots()
Gax = None
for fname, label in [
#                     ('HebbRecogDecoupledManualSequential_Nh=50_R=curriculum.pkl','SeqOneHot'),
#                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0_R=curriculum.pkl','SeqOneHot_a0'), 
#                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0_R=curriculum_hebbianInit.pkl','SeqOneHot_a0_Hebb'),
#                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0_R=curriculum2.pkl','SeqOneHot_a0_$R_0$=2'), 

#                     ('HebbRecogDecoupledManual_Nh=50_R=curriculum.pkl','RandOneHot'),
#                     ('HebbRecogDecoupledManual_Nh=50_alpha=0_R=curriculum.pkl','RandOneHot_a0'),
#                     ('HebbRecogDecoupledManual_Nh=50_alpha=0_R=curriculum_hebbianInit.pkl','RandOneHot_a0_Hebb'), 
#                     ('HebbRecogDecoupledManual_Nh=50_alpha=0_R=curriculum2.pkl','RandOneHot_a0_$R_0$=2')
                     
                     
                     ('HebbRecogDecoupledManualSequential_Nh=50_R=curr2.pkl', 'Baseline ($\\alpha$=1)'),
                     
                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0.0_R=curr2.pkl', 'Seq'),
                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0.0_overwrite_R=curr2.pkl', 'Seq+ovwr'),
                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0.0_overwrite_customInit_R=curr2.pkl', 'Seq+ovwr, custom init'),
                     ('HebbRecogDecoupledManualSequential_Nh=50_alpha=0.0_overwrite_customInit_R=curr2_(2).pkl', 'Seq+ovwr, solved init'),
#
#                     
                     ('HebbRecogDecoupledManual_Nh=50_alpha=0.0_R=curr2.pkl', 'Rnd'),
                     ('HebbRecogDecoupledManual_Nh=50_alpha=0.0_overwrite_R=curr2.pkl', 'Rnd+ovwr'),
                     ('HebbRecogDecoupledManual_Nh=50_alpha=0.0_overwrite_customInit_R=curr2.pkl', 'Rnd+ovwr, custom init'),
                     ('HebbRecogDecoupledManual_Nh=50_alpha=0.0_overwrite_customInit_R=curr2_(2).pkl', 'Rnd+ovwr, solved init')                        
                      ]:
    
    net = load_from_file(fname, dims=[d, 50, 1])
        
    if label.startswith('Rnd'):
        fmt = '--'
    elif label.startswith('Seq'):
        fmt = '-'
    else:
        fmt = ':'
    label = '{} ($\lambda=${:.2f} $\eta=${:.2f})'.format(label, net.lam, net.eta)
    
    print(label)
    Rax.step(*zip(*net.hist['increment_R']), linestyle=fmt, label=label)
    Gax,_,_ = plot_recog_generalization(net, gen_data, fmt=fmt, label=label, ax=Gax)

#    print net.w1.abs().max()
#    plot_W([net.w1.detach()])
#    plot_B([net.b1.detach(), net.b2.detach()])
    

Rax.set_xlabel('Iter')
Rax.set_ylabel('R')
Rax.legend()

