import argparse, importlib, math

from data import generate_recog_data_batch
import torch
import torch.nn as nn
import numpy as np
from net_utils import random_weight_init, load_from_file
from networks import HebbNet

#haba_loop.py args
#script = 'script_recog_vary_w1init.py'
#argsList = [
#            ('--w1init', 'randn',      '--w2init', 'randn'), #controls: train everything # standard init 
#            ('--w1init', 'posneg_eye', '--w2init', 'randn'), # best known init  
#            ('--w1init', 'posneg_eye', '--w2init', 'negs'),  # possible better init?
#        
#            ('--w1init', 'shuffle',    '--w1fix', '--w2init', 'randn'), #fix w1
#            ('--w1init', 'posneg_eye', '--w1fix', '--w2init', 'randn'),  
#            ('--w1init', 'eye',        '--w1fix', '--w2init', 'randn'),  
#            ('--w1init', 'neg_eye',    '--w1fix', '--w2init', 'randn'),  
#            ('--w1init', 'randn',      '--w1fix', '--w2init', 'randn'), 
#            
#            ('--w1init', 'posneg_eye', '--b1init', 'uniform', '--w2init', 'uniform'), #w2 and b1 same for all h's
#            ('--w1init', 'randn',      '--b1init', 'uniform', '--w2init', 'uniform'),  
# 
#            ('--w1init', 'posneg_eye',            '--w2init', 'negs', '--w2fix'), #fix w2
#            ('--w1init', 'randn',                 '--w2init', 'negs', '--w2fix'),   
#
#            ('--w1init', 'shuffle',        '--w1fix', '--w2init', 'randn'), #shuffle&fix w1
#            ('--w1init', 'shuffle',        '--w1fix', '--w2init', 'negs'), 
#            ('--w1init', 'unif_shuf_diags', '--w1fix'),
#            ('--w1init', 'unif_shuf_randn', '--w1fix'), 
#        
#            ] #each line is a set of args for <script>


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--Nh', type=int)
parser.add_argument('--w1init', type=str)
parser.add_argument('--w1fix', action='store_true')
parser.add_argument('--w2init', type=str)
parser.add_argument('--w2fix', action='store_true')
parser.add_argument('--b1init', type=str)
args = parser.parse_args()


#%%
#data params
#R = 5 if args.R is None else args.R #repeat interval
#T = R*20 if args.T is None else args.T #samples per trial
P = 0.5  #repeat probability
d = 25 #length of item

cacheSize = 1024
gen_data = lambda R: generate_recog_data_batch(T=R*20, batchSize=cacheSize, d=d, R=R, P=P)

#network params
Nh = 25 if args.Nh is None else args.Nh #number of hidden units
dims = [d, Nh, 1]


assert(Nh == d)
W,b = random_weight_init([d,Nh,1], bias=True)
if args.w1init == 'eye':
    W[0] = np.eye(Nh)
elif args.w1init == 'neg_eye':
    W[0] = -np.eye(Nh)
elif args.w1init == 'posneg_eye':
    W[0] = np.concatenate([np.zeros([Nh,1]), np.concatenate([-np.eye(Nh-1), np.zeros([1,Nh-1])])], axis=1) + np.eye(Nh)
    W[0][-1,0] = -1
elif args.w1init == 'shuffle':
    net = load_from_file('shuffle_me.pkl', NetClass=HebbNet, dims=[25,25,1])
    w = net.w1.detach()
    W[0] = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape).numpy()
elif args.w1init == 'randn' or args.w1init is None:
    pass
elif args.w1init == 'unif_shuf_diags':
    state = torch.load('unif_shuf_diags.pkl')
    w = state['w1']
    W[0] = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape).numpy()      
    W[1] = state['w2'].numpy()        
    b[0] = state['b1'].numpy()
    b[1] = state['b2'].numpy()
elif args.w1init == 'unif_shuf_randn':
    state = torch.load('unif_shuf_randn.pkl')
    w = state['w1']
    W[0] = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape).numpy()       
    W[1] = state['w2'].numpy()     
    b[0] = state['b1'].numpy()
    b[1] = state['b2'].numpy()
elif args.w1init == 'cat':
    raise NotImplementedError
else: 
    raise ValueError('invalid w1init arg')

if args.w2init == 'negs':
    W[1] = -10*np.ones_like(W[1])
elif args.w2init == 'uniform':
    W[1] = np.array([[-1]])
elif args.w2init == 'randn' or args.w2init is None:
    pass
else: 
    raise ValueError('invalid w2init arg')    


if args.b1init == 'uniform':
    b[0] = np.array([0])
elif args.b1init == 'randn' or args.b1init is None:
    pass
else:
    raise ValueError('invalid b1init arg')    


net = HebbNet([W,b])
if args.w1fix:
    net.w1.requires_grad = False
if args.w2fix:
    net.w2.requires_grad = False
   


#%%
fname = 'w1={}-{}{}_w2={}-{}.pkl'.format(args.w1init, 'fix' if args.w1fix else 'train', 
                                         '_b1={}'.format(args.b1init) if args.b1init is not None else '',
                                         args.w2init, 'fix' if args.w2fix else 'train')
folder = 'vary_w'
iters = 1000000000
net.fit('curriculum', gen_data, iters=iters, batchSize=None, filename=fname, folder=folder, R0=2, increment=lambda R:int(math.ceil(R*1.1)))
 




