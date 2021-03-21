import argparse, math

from networks import HebbNet
from data import generate_recog_data_batch
from net_utils import train_curriculum_simple

def add(R, increment=1):
    return R+increment

def mul(R, scale=1.1):
    return int(math.ceil(R*scale))

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--Nh', type=int)
parser.add_argument('--Ns', type=int)
parser.add_argument('--incr', type=str)
args = parser.parse_args()

#data params
P = 0.5  #repeat probability
d = 25 #length of item
intlv = True #need to implement in generate_*_batch if want False
args.incr = 'add' if args.incr is None else args.incr

if args.incr == 'mul':
    incr = mul
elif args.incr == 'add':
    incr = add
else:
    raise ValueError

cacheSize = 64
gen_data = lambda R: generate_recog_data_batch(T=R*20, batchSize=cacheSize, d=d, R=R, P=P, interleave=intlv)

#network params
Nh = 1 if args.Nh is None else args.Nh #number of hidden units
dims = [d, Nh, 1]
#%% 
net = HebbNet(dims)

#%%
fname = '{}_curriculum={}_Nh={}{}.pkl'.format(net.name, args.incr, Nh, '' if args.Ns is None else args.Ns)
folder = 'HebbNet_curriculum'   
iters = 1000000000
net.fit(train_curriculum_simple, gen_data, increment=incr, iters=iters, batchSize=None, filename=fname, folder=folder)
     




