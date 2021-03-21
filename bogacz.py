import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

from data import generate_recog_data
from net_utils import load_from_file

class BogaczAntiHebb():
    """As described in Bogacz and Brown 2003."""
    def __init__(self, init, eta=.5):
        """init is either an integer (number of novelty neurons) or W"""
        if type(init) == int:
            self.N = init
            self.W = torch.randn(self.N,self.N)
        elif torch.is_tensor(init):
            self.W = init
            self.N = self.W.shape[1]
        else:
            raise ValueError('init must be a tensor or an integer')

        self.W = self.normalize_weights(self.W)

        if self.N % 2 != 0:
            raise ValueError('N must be even because half of novelty neurons active and half silent')

        self.eta = eta


    def forward(self, data, plastic=True):
        out = torch.empty(data.shape[0])
        for t,x in enumerate(data):
            h = torch.mv(self.W, x) #membrane potential
            y = (h>h.median()).float() #top half is activated (bottom half is silenced)
            out[t] = torch.dot(2*y-1, h) #readout familiarity
            if plastic: #turn off plasticity for testing
                self.W -= (self.eta/self.N)*torch.ger(y,x) #update synapses
                self.W = self.normalize_weights(self.W)  #(is this biological?...)
        return out


    def normalize_weights(self, W):
        """According to the paper, normalize to mean=0, var=1. This doesn't work.
        Need to normalize to var=1/sqrt(N)"""
        W = W - W.mean(dim=1).unsqueeze(1)
        W = W / (W.std(dim=1).unsqueeze(1) * self.N**0.5)
        return W


    def accuracy(self, out, target):
        """Threshold output based on fraction of familiar stimuli in dataset
        instead of simply based on out.mean()"""
        outBool = out < np.quantile(out, target.float().mean())
        acc = (outBool == target).float().mean().item()

        tp = outBool[target].float().mean().item() #p(+|+)
        fp = outBool[~target].float().mean().item() #p(+|-)

        return acc, tp, fp


#%% Test Bogacz with fixed dataset
N = 100 #number of novelty units
again = False #if True, present stimuli again in reverse order as in the paper

repeat = 50 #average performance over this many runs of the network
Plist = np.unique(np.logspace(0, 3, dtype=int)) #number of patterns to present
etaList = np.arange(0.1, 1, 0.2) #learning rate
acc = np.zeros((len(etaList), len(Plist)))
tp = np.zeros_like(acc)
fp = np.zeros_like(acc)
for _ in range(repeat):
    for i,eta in enumerate(etaList):
        print('eta={}'.format(eta))
        for j,P in enumerate(Plist):
            net = BogaczAntiHebb(N, eta=eta)
            trainData = torch.rand(P,N).round()*2-1
            testData = torch.rand(P,N).round()*2-1

            net.forward(trainData)
            if again:
                net.forward(trainData.flip(0))
            testData = torch.cat((trainData, testData))
            target = torch.cat((torch.ones(P, dtype=torch.bool), torch.zeros(P, dtype=torch.bool)))
            out = net.forward(testData, plastic=False)

            _acc, _tp, _fp = net.accuracy(out, target)
            acc[i,j] += _acc
            tp[i,j] += _tp
            fp[i,j] += _fp
acc /= repeat
tp /= repeat
fp /= repeat

res = {'Plist':Plist, 'etaList':etaList, 'acc':acc, 'tp':tp, 'fp':fp}
joblib.dump(res, 'bogacz/bogacz_non_continual.pkl')

fig, ax = plt.subplots(2,1, sharex=True)
lines = ax[0].semilogx(res['Plist'], res['acc'].T)
ax[0].set_ylabel('Accuracy')
ax[0].legend(lines, ['$\eta$={}'.format(eta) for eta in res['etaList']])

linesTP = ax[1].semilogx(res['Plist'], res['tp'].T)
linesFP = ax[1].semilogx(res['Plist'], res['fp'].T, '--')
for i,line in enumerate(linesTP):
    linesFP[i].set_color(line.get_color())
ax[1].set_xlabel('$P$ (# patterns presented)')
ax[1].set_ylabel('Probability')
ax[1].plot([], 'k-', label='True positive')
ax[1].plot([], 'k--', label='False positive')
ax[1].legend()


#%% Test Bogacz net or HebbFF with continual dataset
d = N = 100
W = torch.randn(N,N)
eta = .7

repeat = 50 #average performance over this many runs of the network
Rlist = [5,20]
Tlist = np.unique(np.logspace(np.log10(20), np.log10(5000), dtype=int)) #number of patterns to present

for netType in ['hebbff', 'bogacz']:
    acc = np.zeros((len(Rlist), len(Tlist)))
    tp = np.zeros_like(acc)
    fp = np.zeros_like(acc)
    for _ in range(repeat):
        for i,R in enumerate(Rlist):
            for j,T in enumerate(Tlist):
                if R >= T:
                    acc[i,j] = np.nan
                    continue

                trainData = generate_recog_data(T=T, R=R, d=N, P=0.5, multiRep=False)
                target = trainData.tensors[1].bool().flatten()

                if netType == 'bogacz':
                    net = BogaczAntiHebb(W, eta=eta)
                    out = net.forward(trainData.tensors[0])
                    _acc, _tp, _fp = net.accuracy(out, target)
                elif netType == 'hebbff':
                    if net is None:
                        net = load_from_file('antiHebb/HebbNet[100,100,1]_train=inf6_forceAnti.pkl')
                    out = net.evaluate(trainData.tensors).flatten()
                    outBool = out > 0.5
                    _acc = (outBool == target).float().mean().item()
                    _tp = outBool[target].float().mean().item() #p(+|+)
                    _fp = outBool[~target].float().mean().item() #p(+|-)

                f = 1-trainData.tensors[1].mean() #fraction novel
                __acc = (1-f)*_tp + f*(1-_fp) #sanity check
                print('R={}, T={}, tp={:.2f}, fp={:.2f}, acc={:.2f}={:.2f}'.format(R,T, _tp, _fp, _acc, __acc))

                acc[i,j] += _acc
                tp[i,j] += _tp
                fp[i,j] += _fp
    acc /= repeat
    tp /= repeat
    fp /= repeat

    res = {'Tlist':Tlist, 'Rlist':Rlist, 'acc':acc, 'tp':tp, 'fp':fp}
    joblib.dump(res, 'bogacz/{}_continual.pkl'.format(netType))

#%% Histograms of novel and familiar output unit activity for various T
task = 'continual' #'2afc'

R = 5
Tlist = [100,500,1000,5000]

out = []
target = []
for j,T in enumerate(Tlist):
    print('T={}'.format(T))
    repeat = int(10000./T)
    out.append(np.zeros((repeat, T)))
    target.append(np.zeros((repeat, T), dtype=bool))
    for i in range(repeat):
        if task == 'continual':
            net = BogaczAntiHebb(W, eta=eta)
            trainData = generate_recog_data(T=T, R=R, d=N, P=0.5, multiRep=False)
            target[-1][i] = trainData.tensors[1].bool().flatten()
            out[-1][i] = net.forward(trainData.tensors[0])
        elif task == '2afc':
            net = BogaczAntiHebb(N, eta=eta)
            trainData = torch.rand(T/2,N).round()*2-1
            testData = torch.rand(T/2,N).round()*2-1
            net.forward(trainData)
            testData = torch.cat((trainData, testData))
            target[-1][i] = torch.cat((torch.ones(T/2, dtype=torch.bool), torch.zeros(T/2, dtype=torch.bool)))
            out[-1][i] = net.forward(testData, plastic=False)

res = {'out':out, 'target':target, 'Tlist':Tlist, 'R':R}
joblib.dump(res, 'bogacz/bogacz_out_{}_hist.pkl'.format(task))
