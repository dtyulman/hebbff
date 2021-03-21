import sys
import joblib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

from dt_utils import Timer
from hebbian import HebbRecall
from plastic_gated import GSP1Recall
from recurrent_nets import VanillaRNNRecall, LSTMRecall
from data_utils import generate_delayed_recall_data
from neural_net_utils import plot_train_perf, plot_train_perf_from_log
from torch_net_utils import debug_net, plot_xhuy_seq, plot_W_seq

#%% data params
#R = 2
#T = 2000
#interleave = False
##%% netwk params
#net = 'RNN'
#d = 25
#Nh = 50 #should be able to go down to Nh=d in a recurrent net with perfect reconstruction
#dims = [d, Nh, d]
#
##%%
#R=2
##fname = 'VanillaRNNRecall_Nh=200_R={}_inf1_intrlv=True.pkl'.format(R)
#fname = 'VanillaRNNRecall_R={}_inf1_intrlv=False.pkl'.format(R)
#net = joblib.load(fname)
#testData = generate_delayed_recall_data(T=10000, d=d, R=R, interleave=True)
#plot_train_perf(net, 0, title='R={} ({:.4}%)'.format(R, net.accuracy(testData)*100))
#plt.savefig(fname[:-3]+'png') 
#
##%%
fname = 'script_delayed_recall.py_GSP1_2_inf1_False.out'
plot_train_perf_from_log(fname, 0, title='R=2')
plt.savefig('GSP1Recall_R=2_inf1_intlv=False.png') 

#%%
R=1
fname = 'GSP1Recall_R={}_inf1.pkl'.format(R)
net = joblib.load(fname)
#data = generate_delayed_recall_data(T=10000, d=net.J1.shape[1], R=R, interleave=True)
net.accuracy(data)
#plot_xhuy_seq(net,data)
#A, _ = plot_W_seq(net,data)
  