import argparse, importlib, math, types

from data import generate_recog_data_batch
import torch
import torch.nn as nn
import numpy as np
from net_utils import random_weight_init, load_from_file

#script = 'script_recog_decoupled.py'
#argsList = [
#            ('--netClass','HebbRecogDecoupledManual', '--Nh', 50, '--alpha', 0),
#            ('--netClass','HebbRecogDecoupledManualSequential', '--Nh', 50, '--alpha', 0),
#            ('--netClass','HebbRecogDecoupledManual', '--Nh', 50, '--alpha', 0, '--overwriteRow'),
#            ('--netClass','HebbRecogDecoupledManualSequential', '--Nh', 50, '--alpha', 0, '--overwriteRow'),
#            ('--netClass','HebbRecogDecoupledManualSequential', '--Nh', 50),
#            ('--netClass','HebbRecogDecoupledManualSequential', '--Nh', 50, '--alpha', 0, '--overwriteRow', '--customInit'),
#            ('--netClass','HebbRecogDecoupledManual', '--Nh', 50, '--alpha', 0, '--overwriteRow', '--customInit'),
#            ] 

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--Nh', type=int)
parser.add_argument('--netClass', type=str)
parser.add_argument('--alpha', type=float)
parser.add_argument('--overwriteRow', action='store_true')
parser.add_argument('--customInit', action='store_true')
args = parser.parse_args()
#%%
#data params
P = 0.5  #repeat probability
d = 25 #length of item

cacheSize = 1024
gen_data = lambda R: generate_recog_data_batch(T=R*25, batchSize=cacheSize, d=d, R=R, P=P)

#network params
Nh = 50 if args.Nh is None else args.Nh #number of hidden units
dims = [d, Nh, 1]
netClass = 'HebbRecogDecoupledManualSequential' if args.netClass is None else args.netClass #for netClass in ['HebbRecogDecoupled', 'HebbNet', 'HebbRecogDecoupledManual', 'HebbRecogManual']:
NetClass = getattr(importlib.import_module('networks'), netClass)        
net = NetClass(dims)

if args.alpha is not None:
    if args.alpha == 0:
        _alpha = -float('inf')
    elif args.alpha == 1:
        _alpha = float('inf')
    else:
        _alpha = np.log(args.alpha/(1-args.alpha))
    net._alpha = nn.Parameter(torch.tensor(_alpha), requires_grad=False)
alphaStr = '' if args.alpha is None else '_alpha={}'.format(args.alpha)

if args.overwriteRow:
    def update_hebb(self, pre, post):
        if self.plastic:
            self.A[post==1] = 0
            self.A = self.lam*self.A + self.eta*torch.ger(post,pre)
    net.update_hebb = types.MethodType(update_hebb, net) 
    net.lam.data = torch.tensor(1.)
    net.eta.data = torch.tensor(1.)
overwriteStr = '_overwrite' if args.overwriteRow else ''

if args.customInit:
    net.w1.data = torch.zeros_like(net.w1)
    net.b1.data = -20*torch.ones_like(net.b1)
    net.w2.data = 10*torch.ones_like(net.w2)
    net.b2.data = torch.tensor([-5.])
initStr = '_customInit' if args.customInit else ''    

#%%
fname = '{}_Nh={}{}{}{}_R=curr2.pkl'.format(net.name, Nh, alphaStr, overwriteStr, initStr)
folder = 'decoup_compare'
iters = 1000000000
net.fit('curriculum', gen_data, iters=iters, batchSize=None, filename=fname, folder=folder, R0=25, increment=lambda R:int(math.ceil(R*1.1)))
 




