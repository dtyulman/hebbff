import os, copy

import numpy as np
import matplotlib.pyplot as plt
import torch

import plotting
import networks
from net_utils import load_from_file


def compute_FLD_weights(net, R, neuronsMask=None, SCC=False, suppressOffDiagCov=False, multiRep=False, P=0.5, addNoise=False, **genDataKwargs):
    Nh,d = net.w1.shape
    data = generate_recog_data(T=1000, R=R, d=d, P=P, multiRep=multiRep, **genDataKwargs)
    db = net.evaluate_debug(data.tensors)    
    data = db['data']
    
    if addNoise:
        db['h'] = torch.sigmoid(db['a1'] + torch.randn(db['a1'].shape) )
        
    if np.isscalar(R):
        famIdx = (data.tensors[1]==1).nonzero()[:,0].flatten()
        hNov = db['h'][famIdx-R,:] #only use the novel stimuli that were subsequently repeated
    else:
        hNov = db['h'][(data.tensors[1]==0).flatten(),:]
    hFam = db['h'][(data.tensors[1]==1).flatten(),:]
    
    if neuronsMask is not None:
        neuronsMask = neuronsMask.astype(bool)
        hFam = hFam[:,neuronsMask]
        hNov = hNov[:,neuronsMask]
    else:
        neuronsMask = np.ones(Nh).astype(bool)
    
    mN = hNov.mean(dim=0).numpy()
    mF = hFam.mean(dim=0).numpy()
    SN = np.cov(hNov, rowvar=False)
    SF = np.cov(hFam, rowvar=False)
    S = (SN+SF)/2.
    if suppressOffDiagCov: #this is what they actually do in the paper
        S *= np.eye(Nh)
    
    if SCC:
        w = -np.ones(Nh)*neuronsMask/neuronsMask.sum()
    else:
        Sinv = np.linalg.inv(S) if not np.isscalar(S) else np.array([1./S])
        w = np.zeros(Nh)
        w[neuronsMask] = -np.matmul(Sinv, mN-mF)
    b = -np.dot(w[neuronsMask], (mN+mF)/2.) #I'm confused why I need the negative sign here...  
    return w, b


def set_FLD_weights(net, wFLD=None, bFLD=None, R=None, neuronsMask=None, SCC=False, suppressOffDiagCov=False, multiRep=False, P=0.5, addNoise=False, **genDataKwargs):
    if wFLD is None and bFLD is None:
        wFLD, bFLD = compute_FLD_weights(net, R, neuronsMask, SCC, suppressOffDiagCov, multiRep, P, addNoise=addNoise, **genDataKwargs)        
    netFLD = copy.deepcopy(net)
    netFLD.w2.data[0] = torch.tensor(wFLD, dtype=torch.float32)
    netFLD.b2.data[0] = torch.tensor(bFLD, dtype=torch.float32)    
    return netFLD


def set_net_subpop(net, popSize, idx=None, decoder=None, R=None, suppressOffDiagCov=False, addNoise=False):
    if idx is None:
        _,idx = net.w2.data[0].sort() #get ranking of neurons

    if decoder is not None: #compute and set FLD/SCC readout from the top-ranked neurons
        mask = np.ones(Nh)
        mask[idx[popSize:]]=0
        netSub = set_FLD_weights(net, R=R, neuronsMask=mask, SCC=(decoder=='SCC'), addNoise=addNoise, suppressOffDiagCov=suppressOffDiagCov)
    else: #set all non-top-ranked neuron readout weights to zero
        netSub = copy.deepcopy(net)
        w2sub = net.w2.clone()
        w2sub[:,idx[popSize:]]=0
        netSub.w2.data = w2sub
    return netSub

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
torch.set_grad_enabled(False)
_,axsDist = plt.subplots(1, squeeze=False)
suppressOffDiagCov=True
for addNoise in [False]:#, True]:
    for fnum, (fname, label) in enumerate(files):
        print '\n\n\n ------- LOADING {} --------- \n\n'.format(fname)
        net = load_from_file(fname)
        Nh,d = net.w1.shape
        if net.w2.numel()==1:
            net.w2.data = net.w2.expand(1,Nh)   
        if type(net) == networks.HebbClassify:
            from data import GenRecogClassifyData
            net.onlyRecogAcc=True    
            generate_recog_classify_data = GenRecogClassifyData(d=d, sampleSpace='sample_space.pkl')
            def generate_recog_data(T, R, d, P, multiRep=False, batchSize=None):
                return generate_recog_classify_data(T=T, R=R, P=P, multiRep=multiRep, batchSize=batchSize)
        else:
            from data import generate_recog_data
        
           
        Rtraindecoder = np.unique(np.logspace(0, np.log10(100), dtype=int))
        idx = torch.arange(50) #compute_FLD_weights(net, R=Rtraindecoder)[0].argsort()    
    
    #    _,axAcc = plt.subplots()
    #    _,axDist = plt.subplots() 
        axDist = axsDist[fnum,0]           
        for decoder in ['FLD']: #SCC or FLD or None
            print '\n ------- decoder= {} --------- \n'.format(decoder)
           
    #       #%% plot w2 distribution
            popSize = Nh
            netSub = set_net_subpop(net, popSize, idx=idx, decoder=decoder, R=Rtraindecoder, addNoise=addNoise, suppressOffDiagCov=suppressOffDiagCov)
            if netSub.w2.shape[0]==2:
                w2 = netSub.w2[0]
            elif netSub.w2.shape[0]==1:
                w2 = netSub.w2
                w2 = w2[w2.abs()<100]
            else:
                raise Exception
            axDist.hist(w2.detach().numpy().flatten(), density=True, histtype='step', align='mid', label='{} {} {} {}'.format(netSub.name, decoder, label, '+noise' if addNoise else ''), bins=20)
            axDist.legend()
    axDist.set_xlabel('w2')
        
        #%% acc vs pop size
        data = generate_recog_data(T=5000, R=Rtraindecoder, d=d, P=0.5, multiRep=False)
        popSizeList = range(1,Nh,24)
        acc = np.empty(len(popSizeList))*np.nan
        print 'ACCURACY'
        for i,popSize in enumerate(popSizeList):
            print 'popSize={}'.format(popSize),    
            netSub = set_net_subpop(net, popSize, idx=idx, decoder=decoder, R=Rtraindecoder, addNoise=addNoise)
            acc[i] = netSub.accuracy(data.tensors)
            print 'acc', acc[i]
            
        lines = axAcc.plot(popSizeList, acc, label='{} {}'.format(net.name, decoder))
        axAcc.set_xlabel('Pop size')
        axAcc.set_ylabel('Acc')
        axAcc.legend()
        
        
        #%% generalization for various pop sizes
        axGen = None
        gen_data = lambda R: generate_recog_data(T=max(20*R, 1000), R=R, d=d, P=0.5, multiRep=False)
        popSizeList = [1,10,20,30,40,50]
        print 'GENERALIZATION'
        for i,popSize in enumerate(popSizeList):
            print 'popSize={}'.format(popSize)
            netSub = set_net_subpop(net, popSize, idx=idx, decoder=decoder, R=Rtraindecoder, addNoise=addNoise)
            axGen, testR, testAcc = plotting.plot_recog_generalization(netSub, gen_data, ax=axGen, upToR=10, label='popSize={}'.format(popSize))        
        axGen.set_title('{} {}'.format(net.name, decoder))
        axGen.get_figure().savefig( 'gen_{}_{}'.format(net.name, decoder) )
           
        
        #%% histogram for various pop sizes
        gen_data = lambda R, T: generate_recog_data(T, R=R, d=d, P=0.5, multiRep=False)        
        print 'HISTOGRAMS'
        popSizeList = [10,30,50]
        for i,popSize in enumerate(popSizeList):
            print 'popSize={}'.format(popSize)
            netSub = set_net_subpop(net, popSize, idx=idx, decoder=decoder, R=Rtraindecoder, addNoise=addNoise)
            axHist, Rmp, Rmc = plotting.plot_multi_hist(netSub, gen_data, Rmp=5, Rmc=30, title='{} {} popSize={}'.format(net.name, decoder, popSize))
            axHist[0][0].get_figure().savefig( 'hist_{}_{}_popSize={}'.format(net.name, decoder, popSize) )

#%%
#    axAcc.get_figure().savefig( 'accVpop_{}'.format(net.name) )
#%%
os.system('say "script finished"')
