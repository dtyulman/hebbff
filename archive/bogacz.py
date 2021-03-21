import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from data import generate_recog_data, recog_chance, check_recognition_data
from net_utils import binary_classifier_accuracy, load_from_file

 
def verify_no_repeats(train, test):
    for x1 in test:
        for x2 in train:
            if (x1==x2).all():
                raise ValueError('Test data contains item from train data')
                

class BogaczAntiHebb(object):
    """As described in Bogacz and Brown 2003. 
    """
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
        
        self.initW = self.W.clone()
        
        if self.N % 2 != 0:
            raise ValueError('N must be even because half of novelty neurons active and half silent')
        
        self.eta = eta #TODO: can I learn this?
        
    
    def reset(self):
        self.W = self.initW.clone()
    
        
    def gen_data(self, P, d):
        """x_i = {+1, -1}"""
        data = torch.rand(P,d).round()*2-1
        return data
        
        
    def forward(self, data, plastic=True):               
        out = torch.empty(data.shape[0])
        for t,x in enumerate(data):
            h = torch.mv(self.W, x) #membrane potential           
            y = (h>h.median()).float() #top half is activated (bottom half is silenced)            
            out[t] = torch.dot(2*y-1, h) #readout familiarity            
            if plastic:
                self.W -= (self.eta/self.N)*torch.ger(y,x) #update synapses
                self.W = self.normalize_weights(self.W)  #(is this biological?...)  
        return out

        
    def normalize_weights(self, M):
        """According to the paper, normalize to mean=0, var=1. This doesn't work.
        Need to normalize to var=1/sqrt(N)"""
        M = M - M.mean(dim=1).unsqueeze(1) 
        M = M / (M.std(dim=1).unsqueeze(1) * self.N**0.5) 
        return M    
#    def normalize_weights(self, W):
#        """ALternative normalization scheme from Bogacz and Brown 2002. Works out of the box.
#        sum_j(w_ij) = 0, sum_j(w_ij^2) = 1"""
#        W = W - W.mean(1).unsqueeze(1)
#        W = W / (W**2).sum(1).sqrt().unsqueeze(1)         
#        return W  


    def accuracy(self, out, tgt):
#        out_bool = torch.empty(len(out), dtype=bool)
#        for i in range(len(out)):
#            out_bool[i] = out[i] < out[:i+1].mean()           
        out_bool = out < out.mean() #out lower for familiar than for novel
        acc = (out_bool == tgt).float().mean().item()
        tp = out_bool[tgt].float().mean().item() #p(+|+)
        fp = out_bool[~tgt].float().mean().item() #p(+|-)
        tn = (~out_bool[~tgt]).float().mean().item() #p(-|-)
        fn = (~out_bool[tgt]).float().mean().item() #p(-|+)
        return acc, tp, fp, tn, fn

    
    def train_and_test(self, P):
        d = W.shape[1]
        trainData = self.gen_data(P,d)
        self.forward(trainData)
        self.forward(trainData.flip(0)) #present stimuli again in reverse order
        
        testData = self.gen_data(P,d)
        testData = torch.cat((trainData, testData)) 
        tgt = torch.cat((torch.ones(P, dtype=torch.bool), torch.zeros(P, dtype=torch.bool)))
        out = self.forward(testData, plastic=False)
        acc = self.accuracy(out, tgt)
        return acc 
        


def find_best_eta(W, P, lo=0.3, hi=0.7, step=0.05):    
    bestAcc = -np.inf
    bestEta = []
    for eta in np.arange(lo, hi+step, step):
        net = BogaczAntiHebb(W, eta=eta)
        acc = net.train_and_test(P)
        if acc >= bestAcc:
            bestAcc = acc
            bestEta.append( eta )    
    eta = sum(bestEta)/len(bestEta)
    return eta
 

def find_capacity(W):    
    P = 1
    acc = 1
    while acc >= 0.99:        
        acc = evaluate_performance(W, P)
        P = P*2
    
    #binary search P between lo and hi, assume acc monotonically decreasing with increasing P
    lo = P/4
    hi = P/2
    P = (lo+hi)/2
    while hi-lo > 1:
        acc = evaluate_performance(W, P) 
        if acc >= 0.99:
            lo = P
            P = (P+hi)/2
        else:
            hi = P
            P = (P+lo)/2
    return P
    

def evaluate_performance(W, P, eta=None):
    iters = 5000/P
    acc = 0
    if eta is None:
        eta = find_best_eta(W, P)

    for _ in xrange(iters):
        net = BogaczAntiHebb(W, eta=eta)
        acc += net.train_and_test(P)        
    acc /= iters

    print 'P={}, eta={}, acc={}'.format(P, eta, acc)
    return acc
 
if __name__ == '__main__':     
    #%% Plot the weight matrix before and after burn-in
               
    N = 100 #number of novelty units
    d = N #dimension of data (number of input units)
    
    eta = .45 #learning rate, should be optimized, found to be between 0.3 and 0.7 
    W = torch.randn(N,N)
    net = BogaczAntiHebb(W, eta=eta)
    
    B = 1000
    burnData = net.gen_data(B, d)
    
    # "burn-in" a bunch of random memories to initialize W 
    # This is not in the paper. If the burn-in sequence is long enough, this breaks the network.
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,:])
    ax3.hist(net.W.flatten(), label='random init')
    ax1.matshow(net.W, cmap='RdBu_r')
    ax1.set_xlabel('random init')
    
    net.forward(burnData)
    
    ax3.hist(net.W.flatten(), label='burn-in')
    ax2.matshow(net.W, cmap='RdBu_r')
    ax2.set_xlabel('burn-in')
    ax3.legend()
    
    
    #print find_capacity(net.W)
    
    #%% Run the network as described in Bogacz and Brown 2003, with additional burn-in step
    
    N = 26 #number of novelty units
    d = N #dimension of data (number of input units)
    eta = 0.65
    W = torch.randn(N,N)
    
    fig, ax = plt.subplots()
    Plist = np.unique(np.logspace(0,2.5,10, dtype=int))
    for B in [0]:#, 100, 1000, 10000]:
        acc = []
        for P in Plist:
            net = BogaczAntiHebb(W, eta=eta)
            burnData = net.gen_data(B, d)
            net.forward(burnData)        
            acc.append( evaluate_performance(net.W, P, eta=eta) )
        ax.semilogx(Plist, acc, label='burnin={}'.format(B))
    plt.xlabel('P (# patterns stored)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('N={}, $\eta$={}'.format(N, eta))
    
    #%% Run the network with streaming input (same as HebbNet, etc)
    #NOTE: accuracy depends on T! Interestingly, if T large, accuracy drops to ~0.05-40% (so if I flip the readout I can get 80%)
    
    N = 100
    #W = torch.randn(N,N)
    d = N
    eta = .5
    
    fig, ax = plt.subplots()
    Rlist = [1,2,3,5,10,20,50,100]
    for T in [100, 200, 500, 1000, 5000, 10000, 50000]:
        accs = []
        for R in Rlist:
            if R >= T:
                accs.append( np.nan )
                continue
            net = BogaczAntiHebb(W, eta=eta)
    
            trainData = generate_recog_data(T=T, R=R, d=d, P=0.5, multiRep=False)        
            tgt = trainData.tensors[1].bool().flatten()
            chance = recog_chance(trainData)
            
            out = net.forward(trainData.tensors[0])
            acc = net.accuracy(out, tgt)
            print 'T={}, R={}, acc={:.2f}, chance={}'.format(T, R, acc, chance)
            accs.append( acc )
        ax.semilogx(Rlist, accs, label='T={}'.format(T))
    plt.xlabel('R (delay interval)')
    plt.ylabel('Accuracy')
    plt.title('Steaming data. N={}, $\eta$={}, capacity~110'.format(N, eta))
    ax.semilogx(Rlist, 0.66*np.ones_like(Rlist), 'k--', label='chance')
    plt.legend()
    
    
    #%% Run HebbNet with burn-in for comparison. Not affected by T, nor by burn-in!
    
    d = 25 #dimension of data (number of input units)
    fig, ax = plt.subplots()
    Plist = np.unique(np.logspace(0,3,15, dtype=int))
    for B in [0, 100, 1000, 10000]:
        accs = []
        for P in Plist:
            net = load_from_file('HebbNet_R=5.pkl', dims=[d,50,1])
    
            #burn-in (no repeats!)
            if B>0:
                burnData = generate_recog_data(T=B, R=B, d=d, P=0.5, multiRep=False)
                for t,x in enumerate(burnData.tensors[0]): 
                    net(x) 
            
            #store P patterns
            trainData = generate_recog_data(T=P, R=P, d=d, P=0.5, multiRep=False)
            for t,x in enumerate(trainData.tensors[0]):
                net(x)
                
            #evaluate
            testData = generate_recog_data(T=P, R=P, d=d, P=0.5, multiRep=False)
            testData.tensors = ( torch.cat((trainData.tensors[0], testData.tensors[0])),
                                 torch.cat((torch.ones(P), torch.zeros(P))) )
            out = torch.empty_like(testData.tensors[1]) 
            for t,x in enumerate(testData.tensors[0]):
                out[t] = net(x)           
            acc = binary_classifier_accuracy(out, testData.tensors[1])
    
            print 'B={}, P={}, acc={:.2f}'.format(B,P,acc)
            accs.append( acc )
        ax.semilogx(Plist, accs, label='burnin={}'.format(B))
    plt.xlabel('P (# patterns stored)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('HebbNet_R=5, d=25, Nh=50')
    
    
    #%% Run HebbNet with varying T for comparison
    
    d=25
    net = load_from_file('HebbNet_R=5.pkl', dims=[d,50,1])
    
    fig, ax = plt.subplots()
    Rlist = [1,2,3,5,10,20,50,100]
    for T in [100, 200, 500, 1000, 5000, 10000, 50000]:
        accs = []
        for R in Rlist:
            if R >= T:
                accs.append( np.nan )
                continue
            testData = generate_recog_data(T=T, R=R, d=d, P=0.5, multiRep=False)
            out = torch.empty_like(testData.tensors[1]) 
            for t,x in enumerate(testData.tensors[0]):
                out[t] = net(x)
            acc = binary_classifier_accuracy(out, testData.tensors[1])
            chance = recog_chance(testData)
            print 'acc={:.2f}, chance={}'.format(acc, chance)
            accs.append( acc )
        ax.semilogx(Rlist, accs, label='T={}'.format(T))
    plt.xlabel('R (delay interval)')
    plt.ylabel('Accuracy')
    plt.title('Streaming data. HebbNet, R=5. Nh=50, d=25')
    ax.semilogx(Rlist, 0.66*np.ones_like(Rlist), 'k--', label='chance')
    plt.legend()


#%%
class BogaczHebb():  
    def __init__(self, N):
        self.N = N
        self.W = torch.zeros(N,N)
        
          
    def forward(self, data, plastic=True):
        out = torch.empty(data.shape[0])
        for t,x in enumerate(data):
            Wp = self.W * (torch.ones(self.N, self.N) - torch.diag(torch.ones(self.N)))
            hp = torch.mv(Wp, x) #membrane potential, excluding "strong connections"
            out[t] = torch.dot(x,hp)
            if plastic:
                self.W += torch.ger(x,x)/self.N
        return out
    
    
    def accuracy(self, out, tgt):
        out = out > out.mean() #out higher for familiar 
        return (out == tgt).float().mean()

  
#P = 900 #number of patterns to store
#N = 200 #number of novelty units
#d = N #dimension of data (number of input units)
#    
#net = BogaczHebb(N)
#
#trainData = gen_data(P,d)
#net.forward(trainData)
##net.forward(trainData.flip(0)) #present stimuli again in reverse order
#
#testData = gen_data(P,d)
#verify_no_repeats(trainData, testData)
#testData = torch.cat((trainData, testData)) 
#tgt = torch.cat((torch.ones(P, dtype=torch.bool), torch.zeros(P, dtype=torch.bool)))
#
#out = net.forward(testData, plastic=False)
#acc = net.accuracy(out, tgt)
#print acc



#%%
class BogaczAntiHebb2(BogaczAntiHebb):
    """As described in Bogacz and Brown 2002. Equivalent to Bogacz and Brown 2003 except for normalization scheme.
    Note that effective learning rate is doubled, eta_2002 = 2*eta_2003, and so is the familiarity readout value
    """
    def __init__(self, init, a=0.5, eta=0.25):        
       super(BogaczAntiHebb2, self).__init__(init, eta)
       self.a = a #a*N neurons with highest membrane potential are active
       self.eta = self.eta
  
    
    def gen_data(self, P, d):
        """x_i = {0,1}        
        (Although for some reason, this also manages to sometimes work for x_i = {+1, -1})
        """
        data = torch.rand(P,d).round()
        return data
    
    
    def forward(self, data, plastic=True):
        out = torch.empty(data.shape[0])
        for t,x in enumerate(data):
            h = torch.mv(self.W, x) #membrane potential           
            
            idx = torch.sort(h, descending=True)[1]
            y = torch.zeros_like(h)
            y[idx[:int(self.a*self.N)]] = 1. #aN neurons with highest membrane potential are active
                
            out[t] = torch.dot(y-self.a, h) #readout familiarity            
            if plastic:
                self.W -= ( self.eta / (self.N*self.a*(1-self.a)) ) * torch.ger(y,x-self.a) #"homosynaptic LTD"
                self.W = self.normalize_weights(self.W)
        return out    
      
           
    def normalize_weights(self, W):
        W = W - W.mean(1).unsqueeze(1)
        W = W / (W**2).sum(1).sqrt().unsqueeze(1)         
        return W 