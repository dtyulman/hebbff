import torch, os
from copy import deepcopy
from data import generate_recog_data
from networks import HebbNetBatched, HebbNet
from plotting import plot_train_perf, plot_W, plot_recog_generalization, plot_recog_positive_rates, plot_multi_hist, clean_up_w, sort_w, shuffle_w, left_justify_w, plot_w1_row_col_sums 
from net_utils import load_from_file
import matplotlib.pyplot as plt

folder = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-06-05/'
os.chdir(folder) 
       
fname = 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'
net = load_from_file(fname)
torch.set_grad_enabled(False)
d,Nh = net.w1.shape
w = net.w1.clone().detach()

gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, interleave=True, multiRep=True)        

#%%
#axGen = axTfp = None
wClean = clean_up_w(w, cleanSmall=True, cleanLarge=True, largeVal=5, thres=3)
#wSorted = sort_w(wClean, alg='hierarchical', metric='hamming', sort='both')
#wJustified = sort_w(left_justify_w(wSorted, preventRepeatRows=False), alg='hierarchical', metric='hamming', sort='both')
for label in ['clean']:#['baseline', 'clean', 'sort', 'shuf', 'shuf_r', 'shuf_c']:
    net.w1.data = w.clone().detach()

    if label == 'baseline': 
        pass
    elif label == 'clean':
        net.w1.data = wClean
    elif label == 'sort':
        net.w1.data = wSorted
    elif label == 'shuf':
        net.w1.data = shuffle_w(wSorted, rows=True, cols=True)
    elif label == 'shuf_r':
        net.w1.data = shuffle_w(wSorted, rows=True, cols=False)
    elif label == 'shuf_c':
        net.w1.data = shuffle_w(wSorted, rows=False, cols=True)
    elif label == 'justify_naive':
        net.w1.data = wJustified
    else:
        raise ValueError
     
#    plot_w1_row_col_sums(net.w1.data, label=label, save=True)
    
#    fig, ax = plot_W([net.w1.data])
#    fig.set_size_inches([3.5, 4.8])
#    ax[0,0].set_title(label)
#    
#    Rmp=14; Rmc=85
#    ax, Rmp, Rmc = plot_multi_hist(net, title=label, Rmp=Rmp, Rmc=Rmc)
#    ax[0,0].get_figure().savefig('hist_W1_{}_to0or5'.format(thres))
    
    axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, ax=axTfp, label=label)
    axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)
axTfp.legend_.remove()


#%%
net.w1.data = w.clone().detach()
for thres in range(6):
    net.w1.data = clean_up_w(w, cleanSmall=True, cleanLarge=False, thres=thres)
    label = '|$W_1$|<{} to zero'.format(thres)
#    fig, ax = plot_W([net.w1]); fig.set_size_inches([3.5, 4.8])
#    ax[0,0].set_title(label)
#    fig.savefig('W1_{}_toZero'.format(thres))
    
    ax, Rmp, Rmc = plot_multi_hist(net, title=label, Rmp=14, Rmc=85)
    ax[0,0].get_figure().savefig('hist_W1_{}_toZero'.format(thres))

#    axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, ax=axTfp, label=label)
#    axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)
#axGen.get_figure().savefig( 'generalization_set_zero')
#axTfp.get_figure().savefig( 'trueFalsePos_set_zero')
        
#%%
net.w1.data = w.clone().detach()
for thres in range(5,0,-1):
    net.w1.data = clean_up_w(w, cleanSmall=False, cleanLarge=True, largeVal=6, thres=thres)
    label = '$|W_1|$>{} to +/-6'.format(thres)
    
#    fig, ax = plot_W([net.w1]); fig.set_size_inches([3.5, 4.8])
#    ax[0,0].set_title(label)
#    fig.savefig('W1_{}_to6'.format(thres))
    
    if thres == 5:
        Rmp=9; Rmc=50
    elif thres == 4:
        Rmp=7; Rmc=40
    elif thres == 3:
        Rmp=6; Rmc=30
    elif thres == 2:
        Rmp=6; Rmc=30
    elif thres == 1:
        Rmp=5; Rmc=20
    else:
        raise ValueError
    ax, Rmp, Rmc = plot_multi_hist(net, title=label, Rmp=Rmp, Rmc=Rmc)
    ax[0,0].get_figure().savefig('hist_W1_{}_to6_altRmpRmc'.format(thres))
    
#    axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, ax=axTfp, label=label)
#    axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)
#axTfp.legend_.remove()
#axGen.get_figure().savefig( 'generalization_to6')
#axTfp.get_figure().savefig( 'trueFalsePos_to6')

#%%
net.w1.data = w.clone().detach()
for thres in range(5,0,-1): 
    net.w1.data = clean_up_w(w, cleanSmall=False, cleanLarge=True, largeVal=5, thres=thres)
    label = '$|W_1|$>{} to +/-5'.format(thres)
    
#    fig, ax = plot_W([net.w1]); fig.set_size_inches([3.5, 4.8])
#    ax[0,0].set_title(label)
#    fig.savefig('W1_{}_to5'.format(thres))
    
    if thres == 5:
        Rmp=14; Rmc=90
    elif thres == 4:
        Rmp=11; Rmc=90
    elif thres == 3:
        Rmp=11; Rmc=80
    elif thres == 2:
        Rmp=10; Rmc=80
    elif thres == 1:
        Rmp=9; Rmc=70
    else:
        raise ValueError
    ax, Rmp, Rmc = plot_multi_hist(net, title=label, Rmp=Rmp, Rmc=Rmc)
    ax[0,0].get_figure().savefig('hist_W1_{}_to5_altRmpRmc'.format(thres))
    
#    axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, ax=axTfp, label=label)
#    axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)
#axTfp.legend_.remove()
#axGen.get_figure().savefig( 'generalization_to5')
#axTfp.get_figure().savefig( 'trueFalsePos_to5')

#%%
for thres in range(5,0,-1):
    net.w1.data = clean_up_w(w, cleanSmall=True, cleanLarge=True, largeVal=5, thres=thres)
    label = '$|W_1|$>/<{} to 0 or +/-5'.format(thres)
    
    fig, ax = plot_W([net.w1]); fig.set_size_inches([3.5, 4.8])
    ax[0,0].set_title(label)
    fig.savefig('W1_{}_to0or5'.format(thres))
    
#    if thres == 5:
#        Rmp=9; Rmc=50
#    elif thres == 4:
#        Rmp=7; Rmc=40
#    elif thres == 3:
#        Rmp=6; Rmc=30
#    elif thres == 2:
#        Rmp=6; Rmc=30
#    elif thres == 1:
#        Rmp=5; Rmc=20
#    else:
#        raise ValueError
#    Rmp=14; Rmc=85;     
#    ax, Rmp, Rmc = plot_multi_hist(net, title=label, Rmp=Rmp, Rmc=Rmc)
#    ax[0,0].get_figure().savefig('hist_W1_{}_to0or5'.format(thres))
    
#    axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, ax=axTfp, label=label)
#    axGen,Rs,acc = plot_recog_generalization(net, gen_data, ax=axGen, testR=Rs, testAcc=acc, label=label)
#axTfp.legend_.remove()
#axGen.get_figure().savefig( 'generalization_to0or5')
#axTfp.get_figure().savefig( 'trueFalsePos_to0or5')
    
    
    
    
    
    
    