import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from data import generate_recog_data, generate_recog_data_batch, GenRecogClassifyData
from net_utils import load_from_file
import plotting, networks


def choose_gen_data(net, chooseT=False, inputDataFile=None, **kwargs):
    Nh,d = net.w1.shape
    if type(net) == networks.HebbSplitSyn:
        d += net.A.shape[1]    

    multiRep = kwargs.pop('multiRep', False)
    interleave = kwargs.pop('interleave', True)
    softLabels = kwargs.pop('softLabels', False)
    
    if type(net) == networks.HebbNetBatched:
        gen_data = lambda R,T: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, batchSize=None, interleave=interleave, multiRep=multiRep, softLabels=softLabels)
        gen_data_R = lambda R: gen_data(R, T=max(R*20, 1000))
    elif type(net) == networks.HebbClassify:
        generate_recog_classify_data = GenRecogClassifyData(d=d, sampleSpace='sample_space.pkl')
        gen_data = lambda R,T: generate_recog_classify_data(T=T, R=R, P=0.5, multiRep=multiRep, batchSize=None)
        gen_data_R = lambda R: gen_data(R, T=max(500, R*20))
    elif inputDataFile is not None:
        images = torch.load(inputDataFile)
        sampleSpace = TensorDataset(images, torch.zeros(images.shape[0],1))
        generator = GenRecogClassifyData(sampleSpace=sampleSpace)        
        def generate_recog_images(T,d,R,P,multiRep=False,batchSize=None):
            x,y = generator(T, R, P, batchSize, multiRep).tensors
            return TensorDataset(x, y[..., 0:1])
        gen_data = lambda R,T: generate_recog_images(T=T, d=d, R=R, P=0.5, multiRep=multiRep, batchSize=None)        
        gen_data_R = lambda R: gen_data(R, T=len(images))
    else:
        gen_data = lambda R,T: generate_recog_data(T=T, d=d, R=R, P=0.5, interleave=interleave, multiRep=multiRep, softLabels=softLabels)
        gen_data_R = lambda R: gen_data(R, T=max(R*20, 1000))

    if chooseT:
        return gen_data
    return gen_data_R


def choose_input_data_file(label, datasetType='UniqueObjects'):
    inputDataFile = None
    if label.lower().find('bin')>=0:
        inputDataFile = 'BradyOliva2008_{}_ResNet18_d=50_binarize.pkl'.format(datasetType)
    elif label.lower().find('norm')>=0:
        inputDataFile = 'BradyOliva2008_{}_ResNet18_d=50_normalize.pkl'.format(datasetType)
    elif label.lower().find('down')>=0:
        inputDataFile = 'BradyOliva2008_{}_ResNet18_d=50.pkl'.format(datasetType)
    elif label.lower().find('feat')>=0:
        inputDataFile = 'BradyOliva2008_{}_ResNet18.pkl'.format(datasetType)
    elif inputDataFile is not None:
        raise ValueError('Label must have data input info.')
    print('{} Input data: {}'.format(label, inputDataFile))
    return inputDataFile


#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-06-12/'
os.chdir(folder) 
files = [
        ('HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '$R_{train}$=[1-14]'), 
##        ('HebbNet[25,25,1]_train=inf5.pkl', '$R_{train}$=5')
#        ('HebbNet[50,50,2]_ClassifyAndRecog_datasize=1e4_R=10.pkl', '50x50_C=1e4+R=10')
        ]


#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-03/rnn'
os.chdir(folder) 
files = [
         ('nnLSTM[100,100,1]_train=inf3.pkl', 'LSTM $R_{train}$=3'),
         ('nnLSTM[100,100,1]_train=inf6.pkl', 'LSTM $R_{train}$=6'),
         ('nnLSTM[100,100,1]_train=inf[1,2,3,4,5,6,7,8,9].pkl', 'LSTM $R_{train}$=[1-9]'),
         ('nnLSTM[100,100,1]_train=inf[3,6].pkl', 'LSTM $R_{train}$=[3,6]'),
         
#         ('VanillaRNN[100,100,1]_train=inf3.pkl', 'RNN $R_{train}$=3'),
#         ('VanillaRNN[100,100,1]_train=inf6.pkl', 'RNN $R_{train}$=6'),
#         ('VanillaRNN[100,100,1]_train=inf[1,2,3,4,5,6,7,8,9].pkl', 'RNN $R_{train}$=[1-9]'),
#         ('VanillaRNN[100,100,1]_train=inf[3,6].pkl', 'RNN $R_{train}$=[3,6]'),
         ]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-13'
os.chdir(folder) 
files = [
        ('HebbNet[25,25,1]_train=inf1_w1init=randn_b1init=scalar_w2init=scalar.pkl', 'R=1'),
        ('HebbNet[25,25,1]_train=cur2-7_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', 'R=7'),
        ('HebbNet[25,25,1]_train=cur2-14_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', 'R=14')
        ]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-31/recogAndClass'
os.chdir(folder) 
files = [
        ('HebbClassify[25,50,2]_train=cur1_incr=plus1.pkl', 'R=[1-13]'),
#        ('HebbClassify[25,50,2]_train=inf1.pkl', 'R=1'),
#        ('HebbClassify[25,50,2]_train=inf5.pkl', 'R=5'),
#        ('HebbClassify[25,50,2]_train=inf10.pkl', 'R=10'),
#         ('HebbNet[25,50,1]_train=cur2_incr=plus1.pkl', 'Recog R=[1-19]'), #control
        ]

#folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-08-07/recogAndClass'
#os.chdir(folder) 
#files = [
#        ('HebbClassify[25,50,2]_train=cur2_incr=plus1_b1init=scalar_w2init=scalar.pkl', 'SimpleAvg')
#        ]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-08-07/ResNetOut'
os.chdir(folder) 
files = [
         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', 'R=[1-17]'), #control, trained only on random
#         ('HebbNet[50,16,1]_train=cur2_incr=plus1_b1init=scalar_w2init=scalar.pkl', 'Rimg=[2-12]'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur2_incr=plus1.pkl', 'R=[1-17]_Rimg=[2-11]'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1.pkl', 'R=[1-17]_Rimg=15'),      
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur50_incr=plus1.pkl', 'R=[1-17]_Rimg=50'),
    ]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-08-21/softLabels'
os.chdir(folder) 
files = plotting.get_all_pkl(folder)

files = [
         ('HebbNet[25,25,1]_train=cur1_incr=plus1_softLabels0.1_b1init=scalar_w2init=scalar.pkl', 'soft=0.1'),
         ('HebbNet[25,25,1]_train=cur1_incr=plus1_softLabels0.01_b1init=scalar_w2init=scalar.pkl', 'soft=0.01'),
         ('HebbNet[25,25,1]_train=cur1_incr=plus1_softLabels0.001_b1init=scalar_w2init=scalar.pkl', 'soft=0.001'),
    ]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-08-14/splitSyn'
os.chdir(folder) 
files = plotting.get_all_pkl(folder)
files = [
         #Control: HebbFF with d=D or d=D+n (should be the same)
         ('HebbNet[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=all_binary_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl', 'HebbFF_d=100_allBinInit_'),
#         ('HebbNet[96,16,1]_train=cur2_incr=plus1_forceAnti_w1init=all_binary_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl',  'HebbFF_d=96_allBinInit_'),
         ('HebbNet[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=randn_b1init=scalar_w2init=scalar.pkl',                            'HebbFF_d=100_randInit'),
#         ('HebbNet[96,16,1]_train=cur2_incr=plus1_forceAnti_w1init=randn_b1init=scalar_w2init=scalar.pkl',                             'HebbFF_d=96_randInit'),
         
         #Expt: SplitSyn, d=100 ==> D=d-n=96, three different W1 (randn should approach allBin, allBin should not change, allBin+gain should do as well as control)
#         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_gain=w1_w1init=all_binary_b1init=scalar_w2init=scalar.pkl', 'SplitSyn_allBinGain'),
#         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=all_binary_b1init=scalar_w2init=scalar.pkl',         'SplitSyn_allBinInit'),
#         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=randn_b1init=scalar_w2init=scalar.pkl',              'SplitSyn_randInit'),
#         #same as above, hand-pick w2init=-10, b2init=10
         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_gain=w1_w1init=all_binary_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl', 'SplitSyn_allBinGain_'),
         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=all_binary_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl',         'SplitSyn_allBinInit_'),
         ('HebbSplitSyn[100,16,1]_train=cur2_incr=plus1_forceAnti_w1init=randn_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl',              'SplitSyn_randInit_'),
         ]
    
#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-08-21/ResNetOutput'
os.chdir(folder) 
files = [       
         #train from scratch, weighted readout       
#REDOING         ('HebbNet[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl.pkl',      'WA Down Scratch R=1'),
#         ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl.pkl',        'WA Down Scratch R=2'), #old
#         ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl.pkl',  'WA Bin Scratch R=[2-9]'), #contd
#         ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl.pkl', 'WA Norm Scratch R=[2-9]'), #contd
#         ('HebbFeatureLayer[50,16,1]_train=cur2_incr=plus1_Nx=512.pkl',                                                     '* WA Feat Scratch R=[2-14]'), #need to continue??
#         ('HebbFeatureLayer[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18.pkl_Nx=512.pkl', '* #2 WA Feat Scratch R=[2-12]'), #repeated by accident #need to continue??

        #train from scratch, simple readout
         ('HebbNet[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl_b1init=scalar_w2init=scalar.pkl',                'SA Down Scratch R=[1-2]'),
#old         ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl_b1init=scalar_w2init=scalar.pkl',            'SA Down Scratch R=2'), 
         ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl_b1init=scalar_w2init=scalar.pkl',       'SA Bin Scratch R=[2-12]'), #contd
        ('HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl_forceAnti_b1init=scalar_w2init=scalar.pkl', 'SA Norm Scratch R=[2-9]'),
        ('HebbFeatureLayer[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18.pkl_Nx=512_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl', 'SA Feat Scratch R=[1-19]'),


         #train on random, freeze W1. Can we get it to perform? Compare to re-trained W1?
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_freeze=w1.pkl','SA Down PreTr Frz_W1 R=15'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl_freeze=w1.pkl', 'SA Bin PreTr Frz_W1 R=15'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl_freeze=w1.pkl', 'SA Norm PreTr Frz_W1 R=15'),
#RUNNING         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_freeze=w1.pkl', 'SA Feat PreTr Frz_W1 R=15'),
        


          #train on random, fine-tune on ResNet
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur2_incr=plus1.pkl',  'SA Bin PreTr R=[2-11]'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur50_incr=plus1.pkl', 'SA Bin PreTr R=50'),
#         
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1.pkl',     'SA Bin PreTr R=15'),  
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_(2).pkl', 'SA Norm PreTr R=15'),
#         ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar__train=cur15_incr=plus1_(3).pkl', 'SA Down PreTr R=15'),        
        
        ('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', 'Baseline SA Rand R=[1-17]'),
        ]




#%% Plot weight matrices
save=False
for fname, label in files:
    net = load_from_file(fname)
    Nh, d = net.w1.shape    
    
    if not torch.isnan(net.g1):
        net.w1.data = net.w1.data*net.g1.item()
    
    #Attempt to re-sort matrices
#    net.w1.data = sort_w(net.w1.data.detach(), alg='hierarchical', metric='hamming', sort='cols', preproc=clean_up_w)

    #plot W        
    if (net.w1.abs().max() - net.w2.abs().max()).abs() / (net.w1.abs().max() + net.w2.abs().max()) < 0.1:
        fig, ax = plotting.plot_W([net.w1, net.w2])
        fig.set_size_inches([4.3, 4.8])
        if save: plt.savefig( 'w1_w2_{}.pdf'.format(label) )       
    else:   
        fig, ax = plotting.plot_W([net.w1])
        fig.set_size_inches([3.5, 4.8])
        if save: plt.savefig( 'w1_{}.pdf'.format(label) )
        
        fig, ax = plotting.plot_W([net.w2])
        ax[0,0].set_title('$W_2^T$')
        fig.set_size_inches([0.8, 4.8])    
        if save: plt.savefig( 'w2_{}.pdf'.format(label) )    
    
    #plot B    
    if (net.b1.abs().max() - net.b2.abs().max()).abs() / (net.b1.abs().max() + net.b2.abs().max()) < 0.1:
        fig, ax = plotting.plot_B([net.b1, net.b2])
        ax[0,0].set_title('$b_1$')
        ax[0,1].set_title('$b_2$')
        fig.set_size_inches([1, 4.8])    
        if save: plt.savefig( 'b1_b2_{}.pdf'.format(label) )
    else:
        fig, ax = plotting.plot_B([net.b1])
        ax[0,0].set_title('$b_1$')
        fig.set_size_inches([0.8, 4.8])    
        if save: plt.savefig( 'b1_{}.pdf'.format(label) )
        
        fig, ax = plotting.plot_B([net.b2])
        ax[0,0].set_title('$b_2$')
        fig.set_size_inches([0.8, 4.8])    
        if save: plt.savefig( 'b2_{}.pdf'.format(label) )
        
        
#%% Plot generalization performance
save=False
axGen = None
axTfp = None
labels = ''
multiRep=True
interleaveData=True
softLabels=False
upToR=10 #-float('inf')
stopAtR=200 #float('inf')
for fname, label in files:
    net = load_from_file(fname)
    Nh, d = net.w1.shape
    if type(net) == networks.HebbSplitSyn:
        d += net.A.shape[1]

    if type(net) == networks.HebbClassify:
        net.onlyRecogAcc = True

#    softLabels = float(label[label.find('=')+1:])
    inputDataFile = choose_input_data_file(label, datasetType='OneOfPairs')
      
    label = label + ' $\lambda$={:.2f}, $\eta={:.2f}$'.format(net.lam, net.eta)
    
    gen_data = choose_gen_data(net, inputDataFile=inputDataFile, multiRep=multiRep, softLabels=softLabels)

    axTfp,Rs,acc,truePos,falsePos = plotting.plot_recog_positive_rates(net, gen_data, upToR=upToR, stopAtR=stopAtR, burnin=False, ax=axTfp, label=label)
    axGen,Rs,acc = plotting.plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)

    if labels: labels += '+'
    labels += label 
    
axTfp.legend_.remove()
 
if save: axGen.get_figure().savefig( 'gen_{}.pdf'.format(labels) )
if save: axTfp.get_figure().savefig( 'tfp_{}.pdf'.format(labels) )


#%% Plot histograms of unit activations
save=False
Rmp = 1
Rmc = 55
multiRep=False
softLabels = False
for fname, label in files:
    net = load_from_file(fname)
    if type(net) == networks.HebbClassify:
        net.onlyRecogAcc = True
    Nh, d = net.w1.shape
    if type(net) == networks.HebbSplitSyn:
        d += net.A.shape[1]
     
    inputDataFile = choose_input_data_file(label, datasetType='OneOfPairs')          
    gen_data = choose_gen_data(net, chooseT=True, inputDataFile=inputDataFile, multiRep=multiRep, softLabels=softLabels)
          
    ax, Rmp, Rmc = plotting.plot_multi_hist(net, gen_data, title=label, Rmp=Rmp, Rmc=Rmc)
    if save: plt.savefig( 'histgm__testData_{}.pdf'.format(label) )


#%%  Plot sequence of hidden activations   
save=False
        
R = 5
T = 20000
datasetType = 'OneOfPairs' #'UniqueObjects'

for fname, label in files:
    net = load_from_file(fname)
    Nh, d = net.w1.shape
       
    inputDataFile = choose_input_data_file(label, datasetType=datasetType)
    gen_data = choose_gen_data(net, chooseT=True, inputDataFile=inputDataFile, multiRep=False, softLabels=False)
    data = gen_data(R,T)
    
#    wClean = clean_up_w(net.w1.data, cleanSmall=True, cleanLarge=True, largeVal=5, thres=3)
#    wSorted = sort_w(wClean, alg='hierarchical', metric='hamming', sort='both')
#    net.w1.data = wClean         
     
    db = net.evaluate_debug(data.tensors)
    globals().update(db) #TODO: this is a hack to get the vars Wxb, Ax, etc in the namespace    
    
    fig,ax = plt.subplots()
    h_il = plotting.interleave(h, torch.full(h.shape, float('nan'))).T
    im = ax.matshow(h_il, cmap='Reds', vmin=0, vmax=1)
    
    width,height = fig.get_size_inches()
    r,c = h_il.shape
    relWidth = c/r 
    fig.set_size_inches([width*(height/width)/2*relWidth, height*(width/height)/2])
    
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax)
    
    for i in range(2*T):
        if i%2==0 and data.tensors[1][i/2] == 1:
            ax.axvspan(i-0.5, i+0.5, ec='k', fill=False)
    
    ax.set_xticks(range(2*T))
    xticklabels = ['*' if data.tensors[1][t] != out[t].round() else '' for t in range(T)]
    xticklabels = [xticklabels[i/2] if i%2==0 else '' for i in range(2*len(xticklabels))]
    ax.set_xticklabels(xticklabels)
    ax.xaxis.tick_bottom()
    
    ax.set_xlabel('time') 
    ax.set_ylabel('neuron')
    Rmax = net.hist['increment_R'][-1][1] if 'increment_R' in net.hist.keys() else ''.join(ch for ch in fname[fname.find('inf')+3:fname.find('inf')+6] if ch.isdigit())
    ax.set_title('R={}, $R_{{max}}$={}, $\lambda$={:.3f}, $\eta$={:.2f}, acc={:.2f}'.format(R, Rmax, net.lam.item(), net.eta.item(), acc))        
    ax.set_frame_on(False)    
    ax.tick_params(
        axis='both',        
        which='both',
        top=False,      
        left=False,
        bottom=False,
        right=False,
        labeltop=False,
        labelleft=False,
#        labelbottom=False,
        labelright=False)
    
#    fig.tight_layout()
    if save: plt.savefig( 'hSeq_{}.pdf'.format(label) )

        
#%% Plot R as a function of iteration
ax = None
for i,(fname, label) in enumerate(files):
    net = load_from_file(fname) 
    ls = '-'      
    if label.lower().find('baseline') >= 0:
        color = 'tab:brown'
    else: 
        color = None
    
    iters, Rs = zip(*net.hist['increment_R'])
    iters = list(iters)
    Rs = list(Rs)
    iters.append( net.hist['iter'] )
    Rs.append( net.hist['increment_R'][-1][1] )
    ax = plotting.plot_R_curriculum(iters, Rs, label=label, ax=ax, linestyle=ls, color=color)
    ax.legend(loc='lower right')

#%% Plot average activation per neuron
save=False

R = 20
T = 20000
data = generate_recog_data(T=T, d=d, R=R, P=0.5) 
        
for fname, label in files:
    net = load_from_file(fname)
    Nh, d = net.w1.shape                  
    
    a1 = torch.empty(T,Nh)
    h = torch.empty(T,Nh)
    a2 = torch.empty_like(data.tensors[1])
    out = torch.empty_like(data.tensors[1])
    for t,(x,y) in enumerate(data):
        a1[t], h[t], a2[t], out[t] = net.forward(x, debug=True)
    acc = net.accuracy(data.tensors, out)    
    for xx in [a1,h,a2,out]:
        xx.detach_()
    
    novIdx = (data.tensors[1]==0).squeeze()
    hNovMean = h[novIdx,:].mean(0)
    hNovStd = h[novIdx,:].std(0)
    hFamMean = h[~novIdx,:].mean(0)
    hFamStd = h[~novIdx,:].std(0)

    fig,ax = plt.subplots()
    ax.plot(hNovMean, color='red', label='novel')
#    ax.plot(hNovMean+hNovStd, color='red', ls='--')
#    ax.plot(hNovMean-hNovStd, color='red', ls='--')
    
    ax.plot(hFamMean, color='blue', label='familiar')
#    ax.plot(hFamMean+hFamStd, color='blue', ls='--')
#    ax.plot(hFamMean-hFamStd, color='blue', ls='--')
    
    ax.legend(loc='upper right')
    ax.set_xlabel('Hidden unit')
    ax.set_ylabel('Average activation')
    ax.set_title('{}, Rmax={}'.format(label, net.hist['increment_R'][-1][1]))
    ax.set_ylim([0, 0.04])
    fig.set_size_inches(6,4)
    fig.tight_layout()
    
    if save: plt.savefig('hAvg_perUnit_{}.pdf'.format(label))
   
    
#%% Plot magnitude of A matrix over time to find steady-state
T = 20000
d = 25
for Rtest in [1]:#,100]:
    data = generate_recog_data(T=T, d=d, R=Rtest, P=0.5) 
    for burnT, burnR in [(2000,0), (2000,1), (2000,100), (0,0)]:
        fig,ax = plt.subplots()
        for fname, label in files:
            net = load_from_file(fname) 
            title ='Tburn={}_Rburn={}_Rtest={}'.format(burnT, burnR, Rtest)
            print(title)
            
            plotting.change_reset_fn_to_burned_in(net, burnT=burnT, burnR=burnR)      
            net.reset_state()
            
            out = torch.empty_like(data.tensors[1])
            A_abs_mean = torch.zeros(T)              
            for t,(x,y) in enumerate(data):
                out[t] = net.forward(x)
                A_abs_mean[t] = net.A.detach().abs().mean()           
            acc = net.accuracy(data.tensors, out)    
            
            net.reset_state()
            quickAcc = net.accuracy(generate_recog_data(T=max(Rtest*40,1000), d=d, R=Rtest, P=0.5).tensors)
            
#            ax.plot(A_abs_mean, label=label+' acc={:.2f}, quick acc={:.2f}'.format(acc, quickAcc))       
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Iter')
        ax.set_ylabel('mean(|A|)')
        fig.set_size_inches(12.8*0.7, 4.45*0.7)
        fig.tight_layout()
#        fig.savefig(title)
        
        
#%% Plot correlation of Wx+b and Ax as R increases
novOrFam = 'novel'# 'novel'# 
corrOrCov = 'corr'# 'cov'
multiRep = False

for fname, label in files:
    net = load_from_file(fname)    
    
    Nh, d = net.w1.shape 
    gen_data = lambda R: generate_recog_data(T=max(R*20,5000), d=d, R=R, P=0.5, multiRep=multiRep)                 
#    change_reset_fn_to_burned_in(net, gen_data, burnT=5000, burnR=5000)      
#    net.reset_state()  

    Rs = [1,10,100]#,1000,10000]
    corrAvg = np.zeros(len(Rs))
    corrStd = np.zeros(len(Rs))
    for i,R in enumerate(Rs):   
        print('R={}'.format(R))
        data = gen_data(R)
        
        db = net.evaluate_debug(data.tensors)
        globals().update(db) #TODO: this is a hack to get the vars Wxb, Ax, etc in the namespace
        
        idx = (data.tensors[1]==0).squeeze()
        if novOrFam == 'familiar':
            idx = ~idx
        
        if corrOrCov == 'cov':
            corr = np.cov(Ax[idx,:], Wxb[idx,:], rowvar=False)       
        elif corrOrCov == 'corr':
            corr = np.corrcoef(Ax[idx,:], Wxb[idx,:], rowvar=False)        

        corr_hA_hW = np.diag(corr[:Nh, Nh:,]) 
        corrAvg[i] = corr_hA_hW.mean()
        corrStd[i] = corr_hA_hW.std()

#        if R in [1,14,50,100,1000]:
#            fig,ax = plt.subplots()
#            im = ax.matshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
#            fig.colorbar(im)
#            ax.axvline(Nh-0.5, color='k')
#            ax.axhline(Nh-0.5, color='k')
#            ticks_off(ax)
#            ax.set_title('R={}, $\\rho(h_A,h_W)$={:.3f}$\pm${:.3f}'.format(R, corrAvg[i], corrStd[i]) )
#            fig.savefig('corrmat_fam_R={}'.format(R))
        
    fig,ax = plt.subplots()
    line = ax.semilogx(Rs, corrAvg)[0]
    ax.fill_between(Rs, corrAvg+corrStd, corrAvg-corrStd, alpha=0.5, color=line.get_color())
    ax.set_xlabel('$R_{test}$')
    ax.set_ylabel(corrOrCov)
    ax.set_title('{}(Wx+b, Ax), averaged across units, {} only'.format(corrOrCov, novOrFam))
    fig.set_size_inches(8,3.5)
    fig.tight_layout()
    
#%% PLot reaction time curves
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-31/recogAndClass'
os.chdir(folder) 
fname = 'HebbClassify[25,50,2]_train=cur1_incr=plus1.pkl'

net = load_from_file(fname)
Nh,d = net.w1.shape
net.recogOnly = True

gen_data = GenRecogClassifyData(d=d, sampleSpace='sample_space.pkl')    
genDataKwargs = dict(P=0.5, multiRep=False, batchSize=None)

plotting.plot_reaction_time_curves(net, gen_data, **genDataKwargs)

