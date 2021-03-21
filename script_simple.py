import torch
import networks as nets
from data import generate_recog_data, generate_recog_data_batch
from plotting import plot_generalization, get_recog_positive_rates

#choose parameters
netType = 'HebbNet' # HebbFF or LSTM
d = 100             # input dim
N = 100             # hidden dim
force = None        # ensure either Hebbian or anti-Hebbian plasticity
trainMode = 'dat'   # train on single dataset or infinite data
R = 3               # delay interval
T = 500             # length of dataset
save = False

#initialize net
if netType == 'nnLSTM':
    net = nets.nnLSTM([d,N,1])
elif netType == 'HebbNet':
    net = nets.HebbNet([d,N,1])
    if force == 'Hebb':
        net.forceHebb = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item()) #need to re-init for this to work
    elif force == 'Anti':
        net.forceAnti = torch.tensor(True)
        net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
    elif force is not None:
        raise ValueError
else:
    raise ValueError

#train
if trainMode == 'dat':
    trainData = generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    validBatch = generate_recog_data(T=T, d=d, R=R, P=0.5, multiRep=False).tensors
    net.fit('dataset', epochs=float('inf'), trainData=trainData,
            validBatch=validBatch, earlyStop=False)
elif trainMode == 'inf':
    gen_data = lambda: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
    net.fit('infinite', gen_data)
else:
    raise ValueError

#optional save
if save:
    fname = '{}[{},{},1]_{}train={}{}_{}.pkl'.format(
                netType, d, N, 'force{}_'.format(force) if force else '',
                trainMode, R, 'T={}'.format(T) if trainMode != 'cur' else ''
                )
    net.save(fname)

#plot generalization
gen_data = lambda R: generate_recog_data_batch(T=T, d=d, R=R, P=0.5, multiRep=False)
testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)
plot_generalization(testR, testAcc, truePosRate, falsePosRate)
