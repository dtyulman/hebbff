import argparse, importlib, math, types, random, sys, itertools

import torch
import torch.nn as nn

from data import generate_recog_data_batch, GenRecogClassifyData
from net_utils import load_from_file
import networks

## Parse command line args
parser = argparse.ArgumentParser()
groupT = parser.add_mutually_exclusive_group()
groupL = parser.add_mutually_exclusive_group()
groupF = parser.add_mutually_exclusive_group()

#data params
parser.add_argument('--R0',          default=[2],            type=int, nargs='*', help='Initial R before increment for curriculum training, or min of Rlist for infinite training. Pass multiple values to use pre-fixed list. If using --load, can set to -1 to continue where previous session left off.')
parser.add_argument('--Rf',          default=float('inf'), type=int, help='Largest R for curriculum training, or max of Rlist for infinite training')
groupT.add_argument('--T',                         type=int, help='Trial duration')
groupT.add_argument('--Tmul',       default=20,    type=int, help='Trial duration as a multiple of R (Useful for curricumul training, allows constant number of repeats independent of R)')
groupT.add_argument('--Tmin',       default=200,   type=int, help='In combination with --Tmul, sets T=max(Tmin, Tmul*R)')
parser.add_argument('--P',          default=0.5,   type=float, help='Repeat probability')
parser.add_argument('--d',          default=25,    type=int, help='Item vector length')
parser.add_argument('--noInterleave', default=False, action='store_true', help='Interleave repeats eg. A-B-A-B. If false, only allows eg. A-y-A-C-z-C')
parser.add_argument('--noMultiRep', default=False,  action='store_true', help='Allow multiple repeats of same item eg. A-b-A-c-A. If false only allows e.g. A-v-C-A-w-C-x-y-z')
parser.add_argument('--softLabels', default=False,  type=float, help='Use 0+<value> and 1-<value> as labels instead of 0 and 1')
parser.add_argument('--xDataVals', default='+-',     type=str, help='Values of input data to use during training')# choices=['+-', '01', 'uniform[_a_b]', 'normal')

#network params
parser.add_argument('--net',    default='HebbNet',  type=str, help='Class of network (from networks.py)')
parser.add_argument('--Nh',     default=50,         type=int, help='Number of hidden units')
parser.add_argument('--nonlin', default=None,       type=str, help='Nonlinearity in hidden layer', choices=['logistic', 'linear', 'relu', 'tanh'])

#network-specific network params
parser.add_argument('--alpha0', type=float, help='Initial value of alpha for HebbDecoupled')
parser.add_argument('--overwriteA', default=False, action='store_true', help='Overwrite synapses of active hidden units (rows of A matrix) on every update for HebbDecoupled')

#training params
parser.add_argument('--train', default='cur',        type=str, help='Training method', choices=['inf', 'cur', 'multiR', None])
parser.add_argument('--iters', default=float('inf'), type=int, help='Max number of training interations')
parser.add_argument('--batchSize',                   type=int, help='Number of sequences in minibatch per iteration')
parser.add_argument('--freeze', default=[],          type=str, nargs='*', help='Do not train listed parameters')
parser.add_argument('--gain', default=[],            type=str, nargs='*', help='Train only the gain of these parameters after initializing (only implemented for HebbNet.w1 so far)')
parser.add_argument('--learningRate', default=1e-3,  type=float, help='Learning rate of training (alpha parameter of Adam)')
groupL.add_argument('--load',                        type=str, help='Load specified network for the initilization parameters.')
groupL.add_argument('--cont', default=False,         action='store_true', help='Attempts to load network from filename (even if filename is auto-generated) and continue training it. Will overwrite filename.')
parser.add_argument('--gpu', default=False,          action='store_true', help='Uses GPU for training')

#algorithm-specific training params
parser.add_argument('--noEarlyStop',   default=False,    action='store_true', help='For infinite-data training, toggle whether to stop at 99% accuracy')
parser.add_argument('--itersToQuit', default=2*10**6,   type=int, help='Max number of iters to train without an increment in R before quitting')
parser.add_argument('--increment',   default='plus1', type=str, help='Increment function for curriculum training: plusN or timesF, where N is integer and F is float')
parser.add_argument('--cacheSize',   default=1024,    type=int, help='Number of items to cache per epoch for curriculum or infinte training')

#network-specific training params
parser.add_argument('--init',   type=str, help='Set all the parameters according to a hard-coded init')
parser.add_argument('--eta0',   type=float, help='Initial value of eta')
parser.add_argument('--lam0',   type=float, help='Initial value of lambda')
parser.add_argument('--lam_slider0', type=float, help='Initial value of slider for HebbVariableLam')
parser.add_argument('--Ng',     type=int, help='Number of gain-variable diagonals in W1 matrix (for HebbDiags only)')
parser.add_argument('--Nx',     type=int, help='Input dimension that will be reduced down to d (for HebbFeatureLayer only)')

groupF.add_argument('--forceHebb', default=False,  action='store_true', help='Force eta>0 during training')
groupF.add_argument('--forceAnti', default=False,  action='store_true', help='Force eta<0 during training')
parser.add_argument('--reparamLam', default=False, action='store_true', help='Reparameterize lambda as sigmoid(lambda)')
parser.add_argument('--groundTruthPlast', default=False, action='store_true', help='Use ground truth to activate plasticity, i.e. only update when novel.')
parser.add_argument('--sampleSpace', type=str, help='Either the size of the sample space or filename containing the sample space for HebbClassify, or filename containing a S-by-[input dim] tensor of input data')
parser.add_argument('--HebbFeatInit', type=str, help='HebbFF network to load into the HebbFeatureLayer network')
parser.add_argument('--w1init', type=str)#, choices=[None, 'all_binary', 'randn', 'sparse[#][-#]', 'diag[{+,-}*]', 'shuffle', 'shuffle-r', 'shuffle-c'])
parser.add_argument('--b1init', type=str)#, choices=[None, 'randn', 'scalar[#]'])
parser.add_argument('--w2init', type=str)#, choices=[None, 'randn', 'scalar[#]'])
parser.add_argument('--b2init', type=str)#, choices=[None,  'scalar#'])

#saving params
parser.add_argument('--filename', default='', type=str, help='Filename where to save network during training. Set to empty string to automatically generate')
parser.add_argument('--folder',   default='', type=str, help='Folder where to store TensorBoard logs during training. Set to empty string to automatically generate')

#%%
if len(sys.argv) == 1: #running on local  
    args = ( '--net', 'HebbNet', '--Nh', 32, '--d', 50, '--forceAnti', '--init', 'analytic', '--train', 'cur', '--R0', 1,  '--noMultiRep', '--folder', 'vary_Nh_allBin')
    args = [str(a) for a in args]
    args = parser.parse_args(args)
    if args.folder != '': #emulate folder structure made on Habanero
        import os
        if not os.path.exists(args.folder) and not os.path.exists(os.path.join('..', args.folder)):
            os.makedirs(args.folder)
        if not os.getcwd().strip('/').endswith(args.folder):
            sys.path.append(os.getcwd()) #append this to path so it can find the ongoing_plasticity library from different folder
            os.chdir(args.folder) 
else: #running on cluster
    args = parser.parse_args()

if len(args.R0) == 1:
    args.R0 = args.R0[0]

dims = [args.d, args.Nh, 2 if args.net=='HebbClassify' else 1]
 
if args.filename == '': 
#    if args.cont:
#        raise ValueError("Must provide filename if using '--cont'")
    if args.load:
        base = args.load[:-4]+'_'
    else:
        base = '{net}{dims}'        
    base += '_train={train}{R0}'
    #TODO: update to be "if args.X != parser.get_default('X'): base += '_X={X}"
    if args.Rf<float('inf'):                                base+='-{Rf}'
    if args.train=='cur':                                   base+='_incr={increment}'
    if args.xDataVals != '+-':                              base+='_xData={xDataVals}'
    if args.softLabels:                                     base+='_softLabels{softLabels}'            
    if args.sampleSpace is not None:                        base+='_sampleSpace={sampleSpace}'
    if args.HebbFeatInit is not None:                       base+='_HebbFeatInit=pretrained'
    if args.lam0 is not None:                               base+='_lam0={lam0}'
    if args.eta0 is not None:                               base+='_eta0={eta0}'
    if args.forceHebb:                                      base+='_forceHebb'
    if args.forceAnti:                                      base+='_forceAnti'
    if args.groundTruthPlast:                               base+='_groundTruthPlast'
    if args.lam_slider0 is not None:                        base+='_lamslider0={lam_slider0}'
    if args.nonlin is not None:                             base+='_nonlin={nonlin}'
    if args.alpha0 is not None:                             base+='_alpha0={alpha0}'
    if args.reparamLam:                                     base+='_reparamLam'
    if len(args.gain)>0:                                    base+='_gain='+','.join(args.gain)
    if args.Ng is not None:                                 base+='_Ng={Ng}'
    if args.Nx is not None:                                 base+='_Nx={Nx}'
    if args.w1init is not None:                             base+='_w1init={w1init}' 
    if args.b1init is not None:                             base+='_b1init={b1init}' 
    if args.w2init is not None:                             base+='_w2init={w2init}' 
    if args.b2init is not None and args.b2init != 'scalar': base+='_b2init={b2init}'
    if len(args.freeze)>0:                                  base+='_freeze='+','.join(args.freeze)
    if args.overwriteA:                                     base+='_overwriteA' 

    args.filename = (base.replace('.pkl', '')+'.pkl').format(dims=str(dims), **vars(args)).replace(' ','')
    
    
#%% Instantiate the network
if args.load or args.cont:  
    if args.cont:
        args.load = args.filename
            
    net = load_from_file(args.load)
    args.net = net.__class__.__name__
    try:
        args.Nh = net.w1.shape[0]
        args.d = net.w1.shape[1]
        outDim = net.w2.shape[0]
        dims = [args.d, args.Nh, outDim]    
    except:      
        dims = args.filename[ args.filename.find('[')+1 : args.filename.find(']') ]
        dims = [int(d) for d in dims.split(',')]
        args.d = dims[0]
        args.Nh = dims[1]
    
    if args.train == 'cur' and (args.R0 == -1 or args.cont):
        args.R0 = net.hist['increment_R'][-1][1]         

else:
    hebbArgs = {}
    if args.eta0 is not None: 
        hebbArgs.update(eta=args.eta0)
    if args.lam0 is not None: 
        hebbArgs.update(lam=args.lam0)
    
    if args.nonlin == 'logistic' or args.nonlin is None:
        f = torch.sigmoid
    elif args.nonlin == 'linear':
        f = lambda x: x
    else:
        raise NotImplementedError

    NetClass = getattr(importlib.import_module('networks'), args.net)
    if NetClass == networks.HebbDiags:
        net = NetClass([dims, args.Ng], f=f, **hebbArgs) 
    elif NetClass == networks.HebbNetBatched:
        net = NetClass(dims, batchSize=args.batchSize, f=f, **hebbArgs)
    elif NetClass == networks.HebbFeatureLayer:
        assert args.Nx is not None
        net = NetClass(dims, args.Nx, f=f, **hebbArgs)
        if args.HebbFeatInit is not None:
            net.load(args.HebbFeatInit)
    else:
        net = NetClass(dims, f=f, **hebbArgs)
        
    if NetClass == networks.HebbVariableLam:
        if args.lam_slider0 is not None:
            net.lam_slider.data = torch.tensor(args.lam_slider0)

if not args.cont:
    if args.init == 'analytic':
        from scipy.special import erfcinv
        
        n = int(math.log(args.Nh,2))
        f = 2./3
        Pfp = 0.05
        Ptp = 0.995
        E = math.sqrt(2)*(erfcinv(2*Pfp)-erfcinv(2*Ptp))
        lam = math.sqrt(1-math.e*E**2*f/(args.Nh*args.d))
        alpha = math.sqrt( f/(args.d*args.Nh*(1-lam**2)) )
        eta = -5./args.d
        K = -args.d*eta
        b1 = -(erfcinv(2*Pfp)*alpha-n)*args.d*eta
        w2 = -2*float(args.d) #need a principled way to set these...
        b2 = -w2/2.-10 
        
        net.init_hebb(eta=eta, lam=lam) 
        net.w1.data = torch.zeros(args.Nh,args.d)
        net.w1.data[:,:n] = K*torch.tensor(list(itertools.product([1.,-1.], repeat=n)))
        net.b1.data = torch.tensor([b1])
        net.w2.data = torch.tensor([w2])
        net.b2.data = torch.tensor([b2])
    
    
    if args.reparamLam:
        net.reparamLam = torch.tensor(True)   
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) 
         
    if args.forceHebb: 
        net.forceHebb = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item())        
    elif args.forceAnti:
        net.forceAnti = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) 
    
    if args.groundTruthPlast:
        net.groundTruthPlast = torch.tensor(True)
    
    if args.alpha0 is not None:
        if args.alpha0 == 0:
            _alpha = -float('inf')
        elif args.alpha0 == 1:
            _alpha = float('inf')
        else:
            _alpha = math.log(args.alpha0/(1-args.alpha0)) #alpha = sigmoid(_alpha)
        net._alpha = nn.Parameter(torch.tensor(_alpha), requires_grad=False)
    
    if args.overwriteA:
        def update_hebb(self, pre, post):
            if self.plastic:
                self.A[post==1] = 0
                self.A = self.lam*self.A + self.eta*torch.ger(post,pre)
        net.update_hebb = types.MethodType(update_hebb, net) 
    
    if args.w1init is None:
        pass
    elif args.w1init == 'ones':
        net.w1.data = torch.ones(args.Nh,args.d)
    elif args.w1init == 'randn':
        net.w1.data = torch.randn(args.Nh,args.d)/math.sqrt(args.d)
    elif args.w1init.startswith('all_binary'):
        K=1
        if len(args.w1init)>len('all_binary'):
            K = float(args.w1init[len('all_binary'):])
        n = int(math.floor(math.log(args.Nh, 2)))
        assert n <= args.d    
        net.w1.data = torch.zeros(args.Nh,args.d)
        net.w1.data[:,:n] = K*torch.tensor(list(itertools.product([1.,-1.], repeat=n)))
    elif args.w1init.startswith('diag'):
        assert args.d==args.Nh, "'diag' init only valid if Nx==Nh"
        spec = args.w1init[4:] #if len(spec) is odd, middle entry is matrix diagonal. Else, left of middle is diagonal
        assert len(spec) <= args.d/2 
        net.w1.data = torch.zeros(args.Nh,args.d)
        for i,s in enumerate(spec):
            if s == '+':
                net.w1.data += torch.roll(torch.eye(args.Nh), -len(spec)/2+i+1, dims=1)
            elif s == '-':
                net.w1.data -= torch.roll(torch.eye(args.Nh), -len(spec)/2+i+1, dims=1)
            elif s == '0':
                pass
            else:
                raise ValueError("Invalid w1init arg: only include +,-,0 with 'diag'") 
    elif args.w1init.startswith('sparse'):
        spec = args.w1init[6:]
        k = int(spec[:spec.find('-')]) #number of connections per row (per post-synaptic neuron)
        assert k <= args.d
        m = int(spec[spec.find('-')+1:]) # number of positive connections
        assert m <= k
        net.w1.data = torch.zeros(args.Nh,args.d)
        net.w1.data[:,0:k-m] = -1
        net.w1.data[:,k-m:k] = 1
        for i,row in enumerate(net.w1.data):
            net.w1.data[i] = row[torch.randperm(args.d)]
    elif args.w1init == 'shuffle':
        assert args.load is not None
        net.w1.data = net.w1.flatten()[torch.randperm(net.w1.shape.numel())].reshape(net.w1.shape)
    elif args.w1init == 'shuffle-r':
        assert args.load is not None
        for i,row in enumerate(net.w1.data):
            net.w1.data[i] = row[torch.randperm(args.d)]
    elif args.w1init == 'shuffle-c':
        assert args.load is not None
        for j,col in enumerate(net.w1.data.t()):
            net.w1.data[:,j] = col[torch.randperm(args.Nh)]
    else:
        raise ValueError('Invalid w1init arg')
        
    if type(net) == networks.HebbSplitSyn:
        net.w1.data = net.w1.data[:,:net.n]          
      
    
    if args.b1init is None:
        pass
    elif args.b1init == 'randn':
        net.b1.data = torch.randn(args.Nh)
    elif args.b1init.startswith('scalar'):
        if len(args.b1init)>6:
            b1 = float(args.b1init[6:])
        else:
            bound = math.sqrt(args.Nh)
            b1 = random.uniform(-bound, bound)
        net.b1.data = torch.tensor([b1])
    else:
        raise ValueError('Invalid b1init arg') 
    
    
    if args.w2init is None:
        pass
    elif args.w2init == 'randn':
        net.w2.data = torch.randn(dims[2], args.Nh)/math.sqrt(args.Nh)
    elif args.w2init.startswith('scalar'):
        if len(args.w2init)>6:
            w2 = float(args.w2init[6:])
        else:
            w2 = random.gauss(0, math.sqrt(2./args.Nh)) #xavier
        net.w2.data = torch.tensor([[w2]])
        
        if type(net) == networks.HebbClassify:
            net.w2c = nn.Parameter(torch.randn(1, args.Nh)/math.sqrt(args.Nh))       
    else:
        raise ValueError('Invalid w2init arg') 
    
    
    if args.b2init is None:
        pass
    elif args.b2init.startswith('scalar'):
        if len(args.b2init)>6:
            b2 = float(args.b2init[6:])
            net.b2.data = torch.tensor([b2])
    else:
        raise ValueError('Invalid b2init arg')
        
        
    for param in args.freeze:
        getattr(net, param).requires_grad_(False)
    
    
    for param in args.gain:
        if issubclass(type(net), networks.HebbNet) and param == 'w1':
            net.w1.requires_grad_(False)
            net.g1 = nn.Parameter(torch.tensor(1, dtype=torch.float))
        else:
            raise NotImplementedError('Only implemented for HebbNet.w1')

        
#%% Train            
if args.gpu:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this device. Remove '--gpu' flag from input arguments")            
    net.to('cuda')
  

if type(net) == networks.HebbClassify:
    args.cacheSize = -1 
    
    if args.train == 'cur':      
        net.onlyRecogAcc = True

    if args.sampleSpace is None:
        raise ValueError('Must provide sample space file or size if using HebbClassify')
    elif args.sampleSpace.isdigit(): 
        teacher = networks.HebbNet([args.d, args.Nh, 1])
        teacher.requires_grad_(False)
        teacher.plastic = torch.tensor(False)
        teacher.w1.data = net.w1
        teacher.b1.data = net.b1
        teacher.w2.data = net.w2[1:2,:]
        teacher.b2.data = net.b2[1:2]
        generator = GenRecogClassifyData(d=args.d, teacher=teacher, datasize=int(args.sampleSpace), save='SampleSpace_'+args.filename)
    else:
        generator = GenRecogClassifyData(sampleSpace=args.sampleSpace)
    
    def generate_recog_data_batch(T,d,R,P,multiRep,batchSize,**kwargs):
        return generator(T, R, P, batchSize, multiRep)      
    

elif args.sampleSpace is not None:
    from torch.utils.data import TensorDataset
    
    args.cacheSize = -1 
    
    images = torch.load(args.sampleSpace)
    dummyClasses = torch.zeros(images.shape[0],1)
    sampleSpace = TensorDataset(images, dummyClasses)
    generator = GenRecogClassifyData(sampleSpace=sampleSpace)
    
    def generate_recog_data_batch(T,d,R,P,multiRep,batchSize,**kwargs):
        x,y = generator(T, R, P, batchSize, multiRep).tensors
        return TensorDataset(x, y[..., 0:1])

    if type(net) == networks.HebbFeatureLayer:
        assert images.shape[1] == args.Nx #sanity check
        
            
if args.train == 'cur':  
    gen_data = lambda R: generate_recog_data_batch(T=max(args.Tmin, R*args.Tmul) if args.T is None else args.T, 
                                                   d=args.d, 
                                                   R=R, 
                                                   P=args.P, 
                                                   softLabels=args.softLabels,
                                                   interleave=(not args.noInterleave), 
                                                   multiRep=(not args.noMultiRep), 
                                                   batchSize=args.cacheSize,
                                                   xDataVals=args.xDataVals,
                                                   device='cuda' if args.gpu else 'cpu')
    if args.increment.startswith('plus'):
        n = int(args.increment[4:])
        increment = lambda R: R+n
    elif args.increment.startswith('times'):
        n = float(args.increment[5:])
        increment = lambda R: int(math.ceil(R*n)) 
        
    net.fit('curriculum', gen_data, iters=args.iters, itersToQuit=args.itersToQuit, batchSize=args.batchSize, learningRate=args.learningRate,
            filename=args.filename, overwrite=args.cont, folder=args.folder, R0=args.R0, Rf=args.Rf, increment=increment)

elif args.train == 'inf':
    if type(args.R0) == list:
        Rlist = args.R0
    else:
        if not math.isinf(args.Rf):
            Rlist = [2**r for r in range(int(math.floor(math.log(args.R0,2))), int(math.ceil(math.log(args.Rf,2)))+1)]
        else:
            Rlist = [args.R0]
           
    gen_data = lambda: generate_recog_data_batch(T=max(args.Tmin, max(Rlist)*args.Tmul) if args.T is None else args.T, 
                                                 d=args.d, 
                                                 R=Rlist,
                                                 P=args.P, 
                                                 softLabels=args.softLabels,
                                                 interleave=(not args.noInterleave), 
                                                 multiRep=(not args.noMultiRep), 
                                                 batchSize=args.cacheSize,
                                                 xDataVals=args.xDataVals,
                                                 device='cuda' if args.gpu else 'cpu')
    net.fit('infinite', gen_data, iters=args.iters, batchSize=args.batchSize, learningRate=args.learningRate, earlyStop=(not args.noEarlyStop),
            filename=args.filename, overwrite=args.cont, folder=args.folder)


elif args.train == 'multiR':           
    gen_data = lambda Rlist: generate_recog_data_batch(T=max(args.Tmin, max(Rlist)*args.Tmul) if args.T is None else args.T, 
                                                 d=args.d, 
                                                 R=Rlist,
                                                 P=args.P, 
                                                 softLabels=args.softLabels,
                                                 interleave=(not args.noInterleave), 
                                                 multiRep=(not args.noMultiRep), 
                                                 batchSize=args.cacheSize,
                                                 xDataVals=args.xDataVals,
                                                 device='cuda' if args.gpu else 'cpu')
    net.fit('multiR', gen_data, Rlo=args.R0, Rhi=args.Rf, batchSize=args.batchSize, learningRate=args.learningRate,
            filename=args.filename, overwrite=args.cont, folder=args.folder)
    

else:
    raise ValueError("Invalid 'train' argument")



