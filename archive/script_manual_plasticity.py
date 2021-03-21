import types, os
import torch
from torch.utils.data import TensorDataset
from net_utils import load_from_file
from plotting import get_all_pkl, plot_recog_generalization, plot_recog_positive_rates
from data import generate_recog_data

#%%
#folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-06-12/'
#os.chdir(folder) 
#files = [('HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '$R_{train}$=[1-14]')]

#%%
folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-03/fixLam'
os.chdir(folder) 
files = get_all_pkl(folder)
files = [
#         ('HebbNet[100,100,1]_train=inf5_lam0=0.84_eta0=-0.2_freeze=_lam.pkl', 'freeze '),
#         ('HebbNet[100,100,1]_train=inf5_lam0=0.84_eta0=0.1_freeze=_lam.pkl', 'freeze '),
         ('HebbNet[100,100,1]_train=inf5_lam0=0.97_eta0=-0.2_freeze=_lam.pkl', 'freeze '),
         ('HebbNet[100,100,1]_train=inf5_lam0=0.97_eta0=0.1_freeze=_lam.pkl', 'freeze '),

#         ('HebbNet[100,100,1]_train=inf5_lam0=0.84_eta0=-0.2.pkl', 'init=0.84 '),
#         ('HebbNet[100,100,1]_train=inf5_lam0=0.84_eta0=0.1.pkl', 'init=0.84 '),
#         ('HebbNet[100,100,1]_train=inf5_lam0=0.97_eta0=-0.2.pkl', 'init=0.97 '),
#         ('HebbNet[100,100,1]_train=inf5_lam0=0.97_eta0=0.1.pkl', 'init=0.97 '),
        ]


#%% Plot generalization performance
save=False
#axGen = None
#axTfp = None
labels = ''
interleaveData=True
upToR= -float('inf')
for multiRep in [False]:#, True]:
    for gndTruthPlast in [False, True]:
        for fname, label in files:
            label = label + ' multi-rep' if multiRep else ''
            net = load_from_file(fname) 
            net.groundTruthPlast = gndTruthPlast
            Nh, d = net.w1.shape
            Label = label + ' $\lambda$={:.2f}, $\eta={:.2f}$'.format(net.lam, net.eta)
            if gndTruthPlast:
                Label = Label + ' Ground-truth plasticity'
                
            gen_data = lambda R: generate_recog_data(T=max(R*40, 2000), d=d, R=R, P=0.5, interleave=interleaveData, multiRep=multiRep)
            
            axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, upToR=upToR, ax=axTfp, label=Label)
            axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=Label)
        
            if labels: labels += '+'
            labels += Label 
    
axTfp.legend_.remove()
 
if save: axGen.get_figure().savefig( 'gen_{}'.format(labels) )
if save: axTfp.get_figure().savefig( 'tfp_{}'.format(labels) )











