import os, math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

import plotting 
from data import generate_recog_data, generate_recog_data_batch, GenRecogClassifyData 
from net_utils import load_from_file


#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-03/rnn'
os.chdir(folder) 
files = plotting.get_all_pkl(folder)
files = [
#         ('nnLSTM[100,100,1]_train=inf3.pkl', 'LSTM $R_{train}$=3'),
#         ('nnLSTM[100,100,1]_train=inf6.pkl', 'LSTM $R_{train}$=6'),
         ('nnLSTM[100,100,1]_train=inf[1,2,3,4,5,6,7,8,9].pkl', 'LSTM $R_{train}$=[1-9]'),
         ('nnLSTM[100,100,1]_train=inf[3,6].pkl', 'LSTM $R_{train}$=[3,6]'),
         
#         ('VanillaRNN[100,100,1]_train=inf3.pkl', 'RNN $R_{train}$=3'),
#         ('VanillaRNN[100,100,1]_train=inf6.pkl', 'RNN $R_{train}$=6'),
#         ('VanillaRNN[100,100,1]_train=inf[1,2,3,4,5,6,7,8,9].pkl', 'RNN $R_{train}$=[1-9]'),
#         ('VanillaRNN[100,100,1]_train=inf[3,6].pkl', 'RNN $R_{train}$=[3,6]'),
         ]

upToR= 10 #-float('inf')
for fname, label in files:
    axGen = None
    net = load_from_file(fname)        
    d = 100
    gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, interleave=True, multiRep=False)
    axGen,Rs,acc = plotting.plot_recog_generalization(net, gen_data, upToR=10, ax=axGen, label=label)
axGen.set_xscale('linear')
#%%

folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-04-24/'
os.chdir(folder) 
#files = plotting.get_all_pkl(folder); print(files)
files = [#('HebbNet[25,25,1]_train=cur2_incr=plus1.pkl', 'Anti'),
         ('HebbNet[25,25,1]_train=cur2_incr=plus1_lam0=0.8_eta0=0.2.pkl', 'Hebb')]

#axGen = None
for fname, label in files:
    net = load_from_file(fname)        
    Nh,d = net.w1.shape
    label = '$\lambda$={:.2f}, $\eta={:.2f}$'.format(net.lam, net.eta)
    gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, interleave=True, multiRep=False)
    axGen,Rs,acc = plotting.plot_recog_generalization(net, gen_data, ax=axGen, label=label)
