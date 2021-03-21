import math, itertools
import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import scipy.special as sps


from plotting import plot_W, plot_B
from net_utils import StatefulBase, Synapse, check_dims, random_weight_init, \
                      nan_mse_loss, binary_classifier_accuracy, nan_recall_accuracy, \
                      nan_bce_loss, nan_binary_classifier_accuracy

#%%############
### Hebbian ###
###############

class HebbNet(StatefulBase):     
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(HebbNet, self).__init__()        
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            W,b = random_weight_init([Nx,Nh,Ny], bias=True)
        else:
            W,b = init
#            check_dims(W,b)
            
        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.g1 = nn.Parameter(torch.tensor(float('nan')), requires_grad=False) #Can add this to network post-init. Then, freeze W1 and only train its gain
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))
        
        self.loss_fn = F.binary_cross_entropy 
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut    
                             
        self.register_buffer('A', None) 
        try:          
            self.reset_state()
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbFF.__init__'.format(e))
            
        self.register_buffer('plastic', torch.tensor(True))        
        self.register_buffer('forceAnti', torch.tensor(False))
        self.register_buffer('forceHebb', torch.tensor(False))
        self.register_buffer('reparamLam', torch.tensor(False))
        self.register_buffer('reparamLamTau', torch.tensor(False))
        self.register_buffer('groundTruthPlast', torch.tensor(False))
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule

        
    def load(self, filename):
        super(HebbNet, self).load(filename)
        self.update_hebb(torch.tensor([0.]),torch.tensor([0.])) #to get self.eta right if forceHebb/forceAnti used        
    
    
    def reset_state(self):
        self.A = torch.zeros_like(self.w1)                
  
    
    def init_hebb(self, eta=None, lam=0.99):
        if eta is None:
            eta = -5./self.w1.shape[1] #eta*d = -5
            
        """ A(t+1) = lam*A(t) + eta*h(t)x(t)^T """
        if self.reparamLam:
            self._lam = nn.Parameter(torch.tensor(np.log(lam/(1.-lam))))
            if self.lam: 
                del self.lam
            self.lam = torch.sigmoid(self._lam)
        elif self.reparamLamTau:
            self._lam = nn.Parameter(torch.tensor(1./(1-lam)))
            if self.lam: 
                del self.lam
            self.lam = 1. - 1/self._lam
        else:
            self._lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
            self.lam = self._lam.data
            
        #Hebbian learning rate 
        if self.forceAnti:
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta)))) #eta = exp(_eta)
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta)))) #eta = exp(_eta)
            self.eta = torch.exp(self._eta)
        else:
            self._eta = nn.Parameter(torch.tensor(eta))    
            self.eta = self._eta.data    
    
    
    def update_hebb(self, pre, post, isFam=False):
        if self.reparamLam:
            self.lam = torch.sigmoid(self._lam)
        elif self.reparamLamTau:
            self.lam = 1. - 1/self._lam
        else:
            self._lam.data = torch.clamp(self._lam.data, min=0., max=1.) #if lam>1, get exponential explosion
            self.lam = self._lam 
        
        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
            
        if self.plastic:
            if self.groundTruthPlast and isFam:
                self.A = self.lam*self.A
            else:
                self.A = self.lam*self.A + self.eta*torch.ger(post,pre)

    
    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
                
        w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
            
        a1 = torch.addmv(self.b1, w1+self.A, x) #hidden layer activation
        h = self.f( a1 ) 
        self.update_hebb(x,h, isFam=isFam)        
       
        if self.w2.numel()==1:
            w2 = self.w2.expand(1,h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, h) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            return a1, h, a2, y 
        return y
     
     
    def evaluate(self, batch):
        self.reset_state()
        out = torch.empty_like(batch[1]) 
        for t,(x,y) in enumerate(zip(*batch)): 
            out[t] = self(x, isFam=bool(y)) 
        return out
        
    
    @torch.no_grad()    
    def evaluate_debug(self, batch, recogOnly=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        self.reset_state()

        Nh,d = self.w1.shape  
        T = len(batch[1])
        db = {'a1' : torch.empty(T,Nh),
              'h' : torch.empty(T,Nh),
              'Wxb' : torch.empty(T,Nh),
              'Ax' : torch.empty(T,Nh),
              'a2' : torch.empty_like(batch[1]),
              'out' : torch.empty_like(batch[1])}
        for t,(x,y) in enumerate(zip(*batch)):
            db['Ax'][t] = torch.mv(self.A, x) 
            try: isFam = bool(y)
            except: isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)      
            w1 = self.g1*self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, x) 
        db['acc'] = self.accuracy(batch).item()  
                        
        if recogOnly and len(db['out'].shape)>1:
            db['data'] = TensorDataset(batch[0], batch[1][:,0].unsqueeze(1))
            db['out'] = db['out'][:,0].unsqueeze(1)
            db['a2'] = db['a2'][:,0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()        
        return db
    
    
    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out, loss, acc)
        
        if hasattr(self, 'writer'):                                 
            if self.hist['iter']%10 == 0: 
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta',    self.eta, self.hist['iter'])       
                 
                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])

#                if self.hist['iter']%500 == 0: 
#                    self.writer.add_histogram('layer1/w1', self.w1, self.hist['iter'])       
#                    self.writer.add_histogram('layer1/A',  self.A,  self.hist['iter'])       
#                    self.writer.add_histogram('layer1/b1', self.b1, self.hist['iter']) 
#                    self.writer.add_histogram('layer2/w2', self.w2, self.hist['iter'])       
#                    self.writer.add_histogram('layer2/b2', self.b2, self.hist['iter'])
#
#                    fig, ax = plot_W([self.w1.detach(), self.w2.detach()])
#                    fig.set_size_inches(6, 2.8)
#                    self.writer.add_figure('weight', fig, self.hist['iter'])     
#                    
#                    fig, ax = plot_B([self.b1.detach(), self.b2.detach()])
#                    fig.set_size_inches(1.25, 3)
#                    self.writer.add_figure('bias', fig, self.hist['iter']) 



class HebbFeatureLayer(HebbNet):
    def __init__(self, init, Nx, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbFeatureLayer, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        _,d = self.w1.shape
        self.featurizer = nn.Linear(Nx, d)
        
    def forward(self, x, isFam=False, debug=False):
        xFeat = self.featurizer(x)
        return super(HebbFeatureLayer, self).forward(xFeat, isFam, debug)
        
    
    @torch.no_grad()    
    def evaluate_debug(self, batch, recogOnly=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        Nh,d = self.w1.shape  
        T = len(batch[1])
        db = {'a1' : torch.empty(T,Nh),
              'h' : torch.empty(T,Nh),
              'Wxb' : torch.empty(T,Nh),
              'Ax' : torch.empty(T,Nh),
              'a2' : torch.empty_like(batch[1]),
              'out' : torch.empty_like(batch[1])}
        for t,(x,y) in enumerate(zip(*batch)):
            db['Ax'][t] = torch.mv(self.A, self.featurizer(x)) 
            try: isFam = bool(y)
            except: isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)      
            w1 = self.g1*self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, self.featurizer(x)) 
        db['acc'] = self.accuracy(batch).item()  
                        
        if recogOnly and len(db['out'].shape)>1:
            db['data'] = TensorDataset(batch[0], batch[1][:,0].unsqueeze(1))
            db['out'] = db['out'][:,0].unsqueeze(1)
            db['a2'] = db['a2'][:,0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()        
        return db



class HebbSplitSyn(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(HebbSplitSyn, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)

        self.N, self.d = self.w1.shape
        n = np.log2(self.N)
        assert n.is_integer()
        self.n = int(n)
        self.D = self.d-self.n
                
        self.w1.data = self.w1.data[:,:self.n]          
        self.reset_state()
        
        
    def reset_state(self):
        self.A = torch.zeros(self.N, self.D)
    
    
    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
           
        xW = x[:self.n] #length n
        xA = x[self.n:] #length D
        
        w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        Ax = torch.mv(self.A, xA)
        Wxb = torch.addmv(self.b1, w1, xW)
        a1 =  Ax+Wxb #hidden layer activation
        h = self.f( a1 ) 
        self.update_hebb(xA,h, isFam=isFam)        
       
        if self.w2.numel()==1:
            w2 = self.w2.expand(1,h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, h) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            return a1, h, a2, y 
        return y     
    
    
    @torch.no_grad()    
    def evaluate_debug(self, batch, recogOnly=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        Nh,d = self.w1.shape  
        T = len(batch[1])
        db = {'a1' : torch.empty(T,Nh),
              'h' : torch.empty(T,Nh),
              'Wxb' : torch.empty(T,Nh),
              'Ax' : torch.empty(T,Nh),
              'a2' : torch.empty_like(batch[1]),
              'out' : torch.empty_like(batch[1])}
        for t,(x,y) in enumerate(zip(*batch)):
            db['Ax'][t] = torch.mv(self.A, x[self.n:]) 
            try: isFam = bool(y)
            except: isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)      
            w1 = self.g1*self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, x[:self.n]) 
        db['acc'] = self.accuracy(batch).item()  
                        
        if recogOnly and len(db['out'].shape)>1:
            db['data'] = TensorDataset(batch[0], batch[1][:,0].unsqueeze(1))
            db['out'] = db['out'][:,0].unsqueeze(1)
            db['a2'] = db['a2'][:,0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()        
        return db
        
    
    
class HebbClassify(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(HebbClassify, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        self.onlyRecogAcc = False
        self.w2c = nn.Parameter(torch.tensor(float('nan')), requires_grad=False) #add this to network post-init


    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
                
        w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
            
        a1 = torch.addmv(self.b1, w1+self.A, x) #hidden layer activation
        h = self.f( a1 ) 
        self.update_hebb(x,h, isFam=isFam)        
       
        if self.w2.numel()==1:
            w2 = self.w2.expand(1,h.shape[0])
        else:
            w2 = self.w2       
        if not torch.isnan(self.w2c).any():
            w2 = torch.cat((w2, self.w2c.data))            
        a2 = torch.addmv(self.b2, w2, h) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            return a1, h, a2, y 
        return y   
    

    def evaluate(self, batch):
        return super(HebbNet, self).evaluate(batch)
    
    
    def average_loss(self, batch, out=None):
        if out is None:
            out = self.evaluate(batch)
        
        recogLoss = self.loss_fn(out[:,0], batch[1][:,0])        
        classLoss = self.loss_fn(out[:,1], batch[1][:,1]) 
        loss = recogLoss + classLoss              
        
        currentIter = self.hist['iter']+1 #+1 b/c this gets called before hist['iter'] gets incremented
        if currentIter%10==0 and self.training: 
            self.hist['recog_loss'].append( recogLoss.item() )
            self.hist['class_loss'].append( classLoss.item() )
            if hasattr(self, 'writer'):                                 
                self.writer.add_scalars('train/loss_breakdown', {'total' : loss,
                                                                 'recog' : recogLoss,
                                                                 'class' : classLoss}, 
                                        currentIter)   
            print( '     {} recog_loss:{:.3f} class_loss:{:.3f}'.format(currentIter, recogLoss, classLoss) )
        return loss    
    
    
    def accuracy(self, batch, out=None):
        if out is None:
            out = self.evaluate(batch)
                    
        recogAcc = self.acc_fn(out[:,0], batch[1][:,0])        
        classAcc = self.acc_fn(out[:,1], batch[1][:,1])  
        if self.onlyRecogAcc:
            acc = recogAcc
        else:
            acc = (recogAcc + classAcc)/2.
        
        if self.training and self.hist['iter']%10==0:
            self.hist['recog_acc'].append( recogAcc.item() )
            self.hist['class_acc'].append( classAcc.item() )
            if hasattr(self, 'writer'):                                 
                self.writer.add_scalars('train/acc_breakdown', {'average' : acc,
                                                                'recog' : recogAcc,
                                                                'class' : classAcc},
                                        self.hist['iter']) 
            print( '     {} recog_acc:{:.3f} class_acc:{:.3f}'.format(self.hist['iter'], recogAcc, classAcc) )
        return acc
      

    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None):
        if self.hist is None:
            self.hist = {'epoch' : 0,
                         'iter' : -1, #gets incremented when _monitor() is called
                         'train_loss' : [], 
                         'train_acc' : [],
                         'grad_norm': [],
                         'recog_loss':[], 
                         'class_loss':[],
                         'recog_acc':[],
                         'class_acc':[]}
            if validBatch:
                self.hist['valid_loss'] = []
                self.hist['valid_acc']  = []
            self._monitor(trainBatch, validBatch=validBatch)
        else: 
            print('Network already partially trained. Continuing from iter {}'.format(self.hist['iter']))  
  


class DetHebb():    
    def __init__(self, D, n, f=0.5, Pfp=0.01, Ptp=0.99):     
        '''
        D: plastic input dim (total input dim is d=D+n)
        n: log_2(hidden dim)
        f: fraction of novel stimuli
        Pfp: desired probability of false positive
        Ptp: desired probability of true positive
        '''
        self.D = D #plastic input dim
        self.n = n 
        self.d = self.D+self.n
        self.N = 2**n #hidden dim

        self.Pfp = Pfp #true and false positive probabilities determine decay rate and bias
        self.Ptp = Ptp
        self.f = f 
        self.a = (sps.erfcinv(2*Pfp) - sps.erfcinv(2*Ptp))*np.sqrt(2*np.e)
        self.gam = 1 - (np.square(self.a)*f)/(2*self.D*self.N) #decay rate
        
        self.W = D*np.array(list(itertools.product([-1, 1], repeat=n))) #static, shape Nxn
        self.reset_A() #plastic
        self.b = sps.erfcinv(2*Pfp)*np.sqrt(2)/self.a - n
        self.B = D*self.b #bias, such that exactly one unit active for novel


    def reset_A(self):
        self.A = np.zeros((self.N, self.D)) #plastic
                
              
    def forward(self, x, debug=False):     
        xW = x[:self.n] #length n
        xA = x[self.n:] #length D
        Ax = self.A.dot(xA)
        Wxb = self.W.dot(xW) + self.B
        a = Wxb + Ax #pre-activation
        h = np.heaviside(a, 0) #h=1 if a>0, h=0 if a<=0
        yHat = np.all(h==0)

        self.A = self.gam*self.A - np.outer(h, xA)  

        if debug:
            return a, h, Ax, Wxb, yHat              
        return yHat


    def evaluate(self, X):
        Yhat = np.zeros((X.shape[0],1))
        for t,x in enumerate(X):
            Yhat[t] = self.forward(x)
        return Yhat
    
    
    def evaluate_debug(self, batch):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        T = len(batch[1])
        db = {'a1' : np.empty((T,self.N)),
              'h' : np.empty((T,self.N)),
              'Wxb' : np.empty((T,self.N)),
              'Ax' : np.empty((T,self.N)),
              'a2' : np.empty_like((batch[1])),
              'out' : np.empty_like((batch[1]))}
        for t,(x,y) in enumerate(zip(*batch)):
            db['a1'][t], db['h'][t], db['Ax'][t], db['Wxb'][t], db['a2'][t] = self.forward(x, debug=True)      
            db['out'][t] = db['a2'][t]
        db['acc'] = self.accuracy(db['out'], Yhat=batch[1].numpy())
        db['data'] = TensorDataset(batch[0], batch[1])
        return db
    
    
    def accuracy(self, Y, Yhat=None, X=None):
        #alternatively, acc = f*(1-Pfp)+(1-f)*Ptp
        if Yhat is None:
            assert X is not None
            Yhat = self.evaluate(X)
        acc = float((Y==Yhat).sum())/len(Y)
        return acc
              
    
    def true_false_pos(self, Y, Yhat=None, X=None):
        if Yhat is None:
            assert X is not None
            Yhat = self.evaluate(X)        
        posOutIdx = Yhat==1
            
        totPos = Y.sum()
        totNeg = len(Y)-totPos
    
        falsePos = (1-Y)[posOutIdx].sum()
        truePos = Y[posOutIdx].sum()
        
        falsePosRate = falsePos/totNeg 
        truePosRate = truePos/totPos 
        return truePosRate, falsePosRate


    def true_false_pos_analytic(self, R, corrected=False):
        if not corrected:        
            Pfp = 0.5*sps.erfc( self.a*(self.n+self.b)/np.sqrt(2) ) * np.ones(len(R))
            Ptp = 0.5*sps.erfc( self.a*(self.n+self.b-np.power(self.gam,R-1))/np.sqrt(2) )            
        else:
            PfpOld = 0.5*sps.erfc( self.a*(self.n+self.b)/np.sqrt(2) ) * np.ones(len(R))
            PtpOld = 0.5*sps.erfc( self.a*(self.n+self.b-np.power(self.gam,R-1))/np.sqrt(2) )                       
            while True:                
                fEff = (1-PfpOld)*self.f + (1-PtpOld)*(1-self.f) #fraction of items *reported* as novel
                a = self.a * np.sqrt(self.f/fEff)
                Pfp = 0.5*sps.erfc( a*(self.n+self.b)/np.sqrt(2) ) * np.ones(len(R))
                Ptp = 0.5*sps.erfc( a*(self.n+self.b-np.power(self.gam,R-1))/np.sqrt(2) )            
                
                if np.all( np.abs(PfpOld - Pfp) < 0.0001 ) and np.all( np.abs(PtpOld - Ptp) < 0.0001 ):
                    break
                PfpOld, PtpOld = Pfp, Ptp
                print( '{} {}'.format(np.max(np.abs(PfpOld - Pfp)), np.max(np.abs(PtpOld - Ptp))) )
        return Ptp, Pfp  
              

class DetHebbNoSplit(DetHebb):
    def __init__(self, d, n, f=0.5, Pfp=0.01, Ptp=0.99):     
        '''
        d:  input dim
        n: log_2(hidden dim)
        f: fraction of novel stimuli
        Pfp: desired probability of false positive
        Ptp: desired probability of true positive
        '''
        self.d = d #plastic input dim
        self.n = n 
        self.N = 2**n #hidden dim

        self.Pfp = Pfp #true and false positive probabilities determine decay rate and bias
        self.Ptp = Ptp
        self.f = f 
        self.a = (sps.erfcinv(2*Pfp) - sps.erfcinv(2*Ptp))*np.sqrt(2*np.e)
        self.gam = 1 - (np.square(self.a)*f)/(2*self.d*self.N) #decay rate
        
        self.W = d*np.array(list(itertools.product([-1, 1], repeat=n))) #static, shape Nxn
        self.reset_A() #plastic
        self.b = sps.erfcinv(2*Pfp)*np.sqrt(2)/self.a - n
        self.B = d*self.b #bias, such that exactly one unit active for novel


    def reset_A(self):
        self.A = np.zeros((self.N, self.d)) #plastic
                
              
    def forward(self, x, debug=False):     
        xW = x[:self.n] #length n
        Ax = self.A.dot(x)
        Wxb = self.W.dot(xW) + self.B
        a = Wxb + Ax #pre-activation
        h = np.heaviside(a, 0) #h=1 if a>0, h=0 if a<=0
        yHat = np.all(h==0)

        self.A = self.gam*self.A - np.outer(h, x)  

        if debug:
            return a, h, Ax, Wxb, yHat              
        return yHat

          
            
class HebbVariableLam(HebbNet):
    def init_hebb(self, eta=None, lam=0.99):
        super(HebbVariableLam, self).init_hebb(eta, lam)
        self.lam_variable = nn.Parameter(torch.tensor(lam))
        self.lam_slider = nn.Parameter(torch.tensor(0.5)) #set to zero to reduce to HebbFF
        
        
    def update_hebb(self, pre, post):
        self.lam.data = torch.min(self.lam.data, torch.tensor(1.)) #if lam>1, get exponential explosion
        
        if self.forceAnti:
            self.eta = torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = -torch.exp(self._eta)
        else:
            self.eta = self._eta
            
        if self.plastic:
            lam_h = torch.pow(self.lam_variable, post).repeat(self.A.shape[0], 1)
            lam = self.lam_slider*lam_h + (1-self.lam_slider)*self.lam            
            self.A = lam*self.A + self.eta*torch.ger(post,pre) 


    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbVariableLam, self)._monitor(trainBatch, validBatch, out, loss, acc)
        
        if hasattr(self, 'writer'):                                 
            if self.hist['iter']%10 == 0: 
                self.writer.add_scalar('params/lam_slider', self.lam_slider, self.hist['iter'])       
    

class HebbDiags(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs): 
        super(HebbNet, self).__init__() #(sic!) call super of parent class, not of self    
        
        if all([type(x)==int for X in init for x in (X if hasattr(X, '__iter__') else [X])]):                      
            [(Nx,Nh,Ny), Ng] = init
            assert Nx==Nh
            [_,w2], [b1,b2] = random_weight_init([Nx,Nh,Ny], bias=True)  
            gList = [5*np.random.rand() * (-1)**i  for i in range(Ng)]            
            wBasis = [torch.roll(torch.eye(Nx), i, dims=1) for i in range(len(gList))]                 
        else:
            gList, wBasis, b1, w2, b2 = init
        
        self.gList = nn.ParameterList([nn.Parameter(torch.tensor(g, dtype=torch.float)) for g in gList])
        self.wBasis = wBasis
        
        self.w1 = nn.Parameter(torch.zeros_like(wBasis[0]))
        self._compute_w1()
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(w2, dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float))
        
        self.loss_fn = F.binary_cross_entropy 
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut                           
                                    
        self.register_buffer('A', None)           
        self.reset_state()
        
        self.plastic = True        
        self.forceAnti = False
        self.forceHebb = False
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian                 
    
    
    def _compute_w1(self):
        w1 = torch.zeros_like(self.wBasis[0])
        for i in range(len(self.gList)):
            w1 += self.gList[i]*self.wBasis[i]  
        self.w1.data = w1
        return w1
    
    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
            
        a1 = torch.addmv(self.b1, self._compute_w1()+self.A, x) #hidden layer activation
        h = self.f( a1 ) 
        self.update_hebb(x,h)        
       
        if self.w2.numel()==1:
            w2 = self.w2.expand(1,h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, h) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            return a1, h, a2, y 
        return y        
    
    
    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out, loss, acc)
        
        if hasattr(self, 'writer'):                                 
            if self.hist['iter']%10 == 0: 
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta',    self.eta, self.hist['iter'])       

                for i,g in enumerate(self.gList):
                    self.writer.add_scalar('params/g{}'.format(i), g, self.hist['iter'])
                 
                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])        
        
        
class HebbRecogDecoupledManual(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbRecogDecoupledManual, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        self._alpha = nn.Parameter(torch.tensor(0.)) #weights readout neurons and gating neurons in plasticity update

    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        a1 = torch.addmv(self.b1, self.w1+self.A, x) #readout layer activation
        h1 = self.f(a1) 
        
        hP = torch.zeros_like(h1)
        hP[np.random.randint(len(hP))] = 1
        self.alpha = torch.sigmoid(self._alpha)
        self.update_hebb(x, self.alpha*h1+(1-self.alpha)*hP ) #if alpha==1, reduces to original network       
              
        a2 = torch.addmv(self.b2, self.w2, h1) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            localvars = locals()
            return {v:localvars[v].detach() for v in ('h1','a1','hP','a2','y')}            
        return y

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbRecogDecoupledManual, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):                                 
            if self.hist['iter']%10 == 0: 
                self.writer.add_scalar('params/alpha', self.alpha, self.hist['iter'])


class HebbRecogDecoupledManualSequential(HebbRecogDecoupledManual):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbRecogDecoupledManualSequential, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        self.plastIdx = 0

    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        a1 = torch.addmv(self.b1, self.w1+self.A, x) #readout layer activation
        h1 = self.f(a1) 
        
        hP = torch.zeros_like(h1)
        hP[self.plastIdx] = 1
        self.plastIdx = (self.plastIdx+1)%len(hP)
        
        self.alpha = torch.sigmoid(self._alpha)
        self.update_hebb(x, self.alpha*h1+(1-self.alpha)*hP ) #if alpha==1, reduces to original network       
              
        a2 = torch.addmv(self.b2, self.w2, h1) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            localvars = locals()
            return {v:localvars[v].detach() for v in ('h1','a1','hP','a2','y')}            
        return y


#%%                    
class HebbRecogDecoupled(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(HebbNet, self).__init__() #(sic!) call super of parent class, not of self    
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            (w1, w2), (b1, b2) = random_weight_init([Nx,Nh,Ny], bias=True)
            [wP], [bP] = random_weight_init([Nx,Nh], bias=True)              
        else:
            w1,wP,w2, b1,bP,b2 = init
            
        self.w1 = nn.Parameter(torch.tensor(w1, dtype=torch.float))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float))
        self.wP = nn.Parameter(torch.tensor(wP, dtype=torch.float))
        self.bP = nn.Parameter(torch.tensor(bP, dtype=torch.float))
        
        self.w2 = nn.Parameter(torch.tensor(w2, dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float))
        
        self.loss_fn = F.binary_cross_entropy 
        self.acc_fn = binary_classifier_accuracy

        self.f = f
        self.fOut = fOut    
               
        self._alpha = nn.Parameter(torch.tensor(0.)) #sigmoid(_alpha) weights readout neurons and gating neurons in plasticity update
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule
              
        self.register_buffer('A', None)           
        self.reset_state()
     
      
    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        if debug:
            a1W = torch.mv(self.w1, x)
            a1A = torch.mv(self.A,  x)
            a1 = a1W + a1A + self.b1 
        else:
            a1 = torch.addmv(self.b1, self.w1+self.A, x) #readout layer activation
        h1 = self.f(a1) 
        
        aP = torch.addmv(self.bP, self.wP, x) #plasticity-gate layer activation
        hP = self.f(aP)        
        self.alpha = torch.sigmoid(self._alpha)
        self.update_hebb(x, self.alpha*h1+(1-self.alpha)*hP ) #if alpha==1, reduces to original network       
              
        a2 =  torch.addmv(self.b2, self.w2, h1) #output layer activation
        y = self.fOut( a2 ) 
                           
        if debug:
            localvars = locals()
            return {v:localvars[v].detach() for v in ('h1','a1A','a1W','a1','aP','hP','a2','y')}            
        return y
                    
       
    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out, loss, acc) #(sic!) call super of parent class, not of self 
        
        if hasattr(self, 'writer'):                                 
            if self.hist['iter']%10 == 0:       
                self.writer.add_scalar('params/alpha',  self.alpha, self.hist['iter'])       
  
                if self.hist['iter']%500 == 0: 
                    self.writer.add_histogram('layer1/w1', self.w1, self.hist['iter'])     
                    self.writer.add_histogram('layer1/wP', self.wP, self.hist['iter'])       
                    self.writer.add_histogram('layer1/b1', self.b1, self.hist['iter'])
                    self.writer.add_histogram('layer1/bP', self.bP, self.hist['iter'])
                    self.writer.add_histogram('layer2/w2', self.w2, self.hist['iter'])       
                    self.writer.add_histogram('layer2/b2', self.b2, self.hist['iter'])
                                        
                    fig, ax = plot_W([self.w1.detach(), self.wP.detach(), self.w2.detach()])
                    fig.set_size_inches(6, 2.8)
                    ax[0].set_title('$W_{1}$')
                    ax[1].set_title('$W_{P}$')
                    ax[2].set_title('$W_2^T$')
                    self.writer.add_figure('weight', fig, self.hist['iter'])     
                    
                    fig, ax = plot_B([self.bP.detach(), self.b1.detach(), self.b2.detach()])
                    fig.set_size_inches(1.25, 3)
                    ax[0].set_title('$b_{1}$')
                    ax[1].set_title('$b_{P}$')
                    ax[2].set_title('$b_2$')
                    self.writer.add_figure('bias', fig, self.hist['iter'])  
                   

                    

                    
#%%
class HebbNetBatched(StatefulBase):     
    def __init__(self, init, batchSize=1, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        """
        NOTE: self.w2 is stored transposed so that I don't have to transpose it every time in batched version of forward()
        """
        super(HebbNetBatched, self).__init__()        
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            W,b = random_weight_init([Nx,Nh,Ny], bias=True)
        else:
            W,b = init
            check_dims(W,b)
            
        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float)) #shape=[Nh,Nx]
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float).unsqueeze(1)) #shape=[Nh,1] for broadcasting
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float).t()) #shape=[Nh,Ny] pre-transposed for faster matmul
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float)) #shape=[Ny]
        
        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy

        self.f = f
        self.fOut = fOut    
               
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule
        
        self.register_buffer('A', None)           
        self.reset_state(batchSize=batchSize)
        
    
    def reset_state(self, batchSize=None):
        if batchSize is None:
            batchSize,_,_ = self.A.shape
        self.A = torch.zeros(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Nh,Nx]    
        
    
    def init_hebb(self, eta=None, lam=0.99):
        if eta is None:
            eta = -5./self.w1.shape[1] #eta*d = -5            
        self.lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.eta = nn.Parameter(torch.tensor(eta)) #Hebbian learning rate       
    
    
    def update_hebb(self, pre, post):
        """Updates A using a (batched) outer product, i.e. torch.ger(post, pre)
        for each of the elements in the batch
            
        pre.shape = [B,Nx] (pre.unsq.shape=[B,1,Nx])
        post.shape = [B,Nh,1]
        """                
        self.lam.data = torch.clamp(self.lam.data, max=1.)
        self.A = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1)) #shape=[B,Nh,Nx]
    
    
    def forward(self, x, debug=False):
        """
        x.shape = [B,Nx]
        
        NOTE: This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        #b1.shape=[Nh,1], w1.shape=[Nh,Nx], A.shape=[B,Nh,Nx], x.unsq.shape=[B,Nx,1]
        a1 = torch.baddbmm(self.b1, self.w1+self.A, x.unsqueeze(2)) #shape=[B,Nh,1] (broadcast)
        h = self.f(a1) #hidden layer activation
        self.update_hebb(x,h)        
       
        #b2.shape=[Ny], h.sq.shape=[B,Nh] w2.shape=[Nh,Ny]
        a2 = torch.addmm(self.b2, h.squeeze(dim=2), self.w2) #shape=[B,Ny]
        y = self.fOut(a2) #output layer activation
        
        if debug:
            return a1, h, a2, y 
        return y



#%%
class HebbAugRecog(HebbNetBatched):
    def __init__(self, init, f=torch.sigmoid, c=0.5, **hebbArgs):
        super(HebbAugRecog, self).__init__(init, f=f, fOut=None, **hebbArgs)
        self.w2 = nn.Parameter(self.w2.data.t()) #shape=[Ny,Nh] undo transpose from parent class 
        self.b2 = nn.Parameter(self.b2.data.unsqueeze(1)) #shape=[Ny,1] unsqueeze for broadcasting
        
        del self.A, self.A1, self.A2
        self.register_buffer('A1', None)    
        self.register_buffer('A2', None)           
        self.reset_state()
        
        if c is not None:
            c = torch.tensor(c) #use MSE loss for value, BCE loss for recog, weight by c
            self.fVal = torch.tanh
        else:
            self.fVal = torch.sigmoid #use BCE loss for everything
        self.c = nn.Parameter(c, requires_grad=False) #scales relative contributions of MSE (value) and BCE (recognition) losses 
     
    
    def reset_state(self, batchSize=1):
        self.A1 = torch.zeros(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Nh,Nx]    
        self.A2 = torch.zeros(batchSize, *self.w2.shape, device=self.w2.device) #shape=[B,Nh,Ny] 
       
        
    def init_hebb(self, eta1=0.1, lam1=0.9, eta2=0.1, lam2=0.9):
        self.lam1 = nn.Parameter(torch.tensor(lam1)) #layer 1 
        self.eta1 = nn.Parameter(torch.tensor(eta1))  
        self.lam2 = nn.Parameter(torch.tensor(lam2)) #layer 2 
        self.eta2 = nn.Parameter(torch.tensor(eta2))  
       
            
    def forward(self, x):
        """
        x.shape = [B,Nx]
        
        NOTE: This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        #b1.shape=[Nh,1], w1.shape=[Nh,Nx], A.shape=[B,Nh,Nx], x.unsq.shape=[B,Nx,1]
        
        a1 = torch.baddbmm(self.b1, self.w1+self.A1, x.unsqueeze(2)) #shape=[B,Nh,1] (broadcast)
        h = self.f(a1) #hidden layer activation        
        self.A1 = self.lam1*self.A1 + self.eta1*torch.bmm(h, x.unsqueeze(1)) #shape=[B,Nh,Nx]

        #b2.shape=[Ny], h.sq.shape=[B,Nh] w2.shape=[Nh,Ny]
        a2 = torch.baddbmm(self.b2, self.w2+self.A2, h) #shape=[B,Ny,1] (broadcast)
        y = torch.empty_like(a2)
        y[:,0] = torch.sigmoid(a2[:,0]) # recognition signal 
        y[:,1] = self.fVal(a2[:,0]) # value associated with item    
        self.A2 = self.lam2*self.A2 + self.eta2*torch.bmm(y, h.permute(0,2,1)) #shape=[B,Nh,Ny]

        return y.squeeze()
    
    
    def average_loss(self, batch, out=None):
        """Computes BCE loss over the recognition marker and MSE loss over the associated value"""
        if out is None:
            out = self.evaluate(batch)
        recogLoss = F.binary_cross_entropy(out[:,:,0], batch[1][:,:,0])        
        if self.c.numel() == 0:
            valueLoss = nan_bce_loss(out[:,:,1], batch[1][:,:,1])              
            loss = nan_bce_loss(out, batch[1])
        else: 
            valueLoss = nan_mse_loss(out[:,:,1], batch[1][:,:,1]) 
            recogLoss = self.c*recogLoss
            valueLoss = (1-self.c)*valueLoss 
            loss = recogLoss + valueLoss              

        
        currentIter = self.hist['iter']+1 #+1 b/c this gets called before hist['iter'] gets incremented
        if currentIter%10==0 and self.training: 
            self.hist['recog_loss'].append( self.c*recogLoss.item() )
            self.hist['value_loss'].append( (1-self.c)*valueLoss.item() )
            if hasattr(self, 'writer'):                                 
                self.writer.add_scalars('train/loss_breakdown', {'total' : loss,
                                                                 'recog' : recogLoss,
                                                                 'value' : valueLoss}, 
                                        currentIter)   
            print( '     {} recog_loss:{:.3f} value_loss:{:.3f}'.format(currentIter, recogLoss, valueLoss) )
        return loss    
    
    
    def accuracy(self, batch, out=None):
        if out is None:
            out = self.evaluate(batch)
        recogAcc = binary_classifier_accuracy(out[:,:,0], batch[1][:,:,0])        
        if self.c.numel() == 0:
            valueAcc = nan_binary_classifier_accuracy(out[:,:,1], batch[1][:,:,1])              
            acc = nan_binary_classifier_accuracy(out, batch[1])
        else: 
            valueAcc = nan_recall_accuracy(out[:,:,1:], batch[1][:,:,1:])        
            N = (~torch.isnan(out[:,:,1:])).all(dim=2).sum().float() #samples in valueAcc      
            M = (out.shape[0]*out.shape[1]) #samples in recogAcc
            c = N/(N+M)
            acc = (1-c)*recogAcc + c*valueAcc
        
        if self.training and self.hist['iter']%10==0:
            self.hist['recog_acc'].append( recogAcc.item() )
            self.hist['value_acc'].append( valueAcc.item() )
            if hasattr(self, 'writer'):                                 
                self.writer.add_scalars('train/acc_breakdown', {'average' : acc,
                                                                'recog' : recogAcc,
                                                                'value' : valueAcc},
                                        self.hist['iter']) 
            print( '     {} recog_acc:{:.3f} value_acc:{:.3f}'.format(self.hist['iter'], recogAcc, valueAcc) )
        return acc
      

    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None):
        if self.hist is None:
            self.hist = {'epoch' : 0,
                         'iter' : -1, #gets incremented when _monitor() is called
                         'train_loss' : [], 
                         'train_acc' : [],
                         'grad_norm': [],
                         'recog_loss':[], 
                         'value_loss':[],
                         'recog_acc':[],
                         'value_acc':[]}
            if validBatch:
                self.hist['valid_loss'] = []
                self.hist['valid_acc']  = []
            self._monitor(trainBatch, validBatch=validBatch)
        else: 
            print('Network already partially trained. Continuing from iter {}'.format(self.hist['iter']))  
            
#%%
class HebbRecall(HebbNet):    
    def __init__(self, init, f=torch.sigmoid, fOut=torch.tanh, **hebbArgs):
        super(HebbRecall, self).__init__(init, f, fOut, **hebbArgs)
        self.loss_fn = nan_mse_loss
        self.acc_fn = nan_recall_accuracy


#%%
class LearnedSynHebb(HebbNet):      
    def init_hebb(self, lam, Ns):
        self.lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.synaptic_update = Synapse(Ns)
        
        
    def update_hebb(self, pre, post):
        self.A = self.lam*self.A + self.synaptic_update(pre, post)   
        
        
#%%##########
### Gated ###
#############       
        
class PlasticGatedBase(StatefulBase):
    def reset_state(self):
        self.h = torch.zeros_like(self.b1) #zero initial activity
        self.A = torch.zeros_like(self.J1) #zero initial plastic component
        
    
class ThreeGateSynapse(PlasticGatedBase):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid):        
        super(ThreeGateSynapse, self).__init__()
    
        if all([type(x)==int for x in init]):
            Nh = init[1]
            J,B = random_weight_init(init, bias=True)
            U,bU = random_weight_init([Nh,Nh], bias=True)
            V,bV = random_weight_init([Nh,Nh], bias=True)
            W,bW = random_weight_init([Nh,Nh], bias=True)  
        else:
            J,U,V,W, B,bU,bV,bW = init
        
        self.J1 = nn.Parameter(torch.tensor(J[0],  dtype=torch.float)) #layer 1 
        self.b1 = nn.Parameter(torch.tensor(B[0],  dtype=torch.float))        
        self.J2 = nn.Parameter(torch.tensor(J[1],  dtype=torch.float)) #layer 2
        self.b2 = nn.Parameter(torch.tensor(B[1],  dtype=torch.float)) 
        self.U  = nn.Parameter(torch.tensor(U[0],  dtype=torch.float)) #write gate
        self.bU = nn.Parameter(torch.tensor(bU[0], dtype=torch.float))
        self.V  = nn.Parameter(torch.tensor(V[0],  dtype=torch.float)) #forget gate
        self.bV = nn.Parameter(torch.tensor(bV[0], dtype=torch.float))
        self.W  = nn.Parameter(torch.tensor(W[0],  dtype=torch.float)) #read gate
        self.bW = nn.Parameter(torch.tensor(bW[0], dtype=torch.float))
                 
        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy 

        self.f = f
        self.fOut = fOut
        
        self.reset_state()
                            
        
    def forward(self, x):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        rd = torch.sigmoid( torch.addmv(self.bW, self.W, self.h) ) #read gate        
        a1 = torch.addmv(self.b1, self.J1, x) + rd*torch.mv(self.A, x) #hidden layer activation
        self.h = self.f( a1 )              
        
        wr = torch.sigmoid( torch.addmv(self.bU, self.U, self.h) ).unsqueeze(1) #write gate
        fg = torch.sigmoid( torch.addmv(self.bV, self.V, self.h) ).unsqueeze(1) #forget gate
        self.A = fg*self.A + wr*torch.ger(self.h,x) #hebbian update
                
        a2 = torch.addmv(self.b2, self.J2, self.h) #output layer activation
        y = self.fOut( a2 ) 
        
        return y
        
   

class GSP1(PlasticGatedBase):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid):
        super(GSP1, self).__init__()
                
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            J,B  = random_weight_init([Nx,Nh,Ny], bias=True)
            U,bU = random_weight_init([Nx,Nh,Nh], bias=True)
            W,bW = random_weight_init([Nx,Nh,Nh], bias=True)  
        else:
            J,W,U, B,bW,bU = init
                                
        self.J1 = nn.Parameter(torch.tensor(J[0],  dtype=torch.float)) #layer 1
        self.b1 = nn.Parameter(torch.tensor(B[0],  dtype=torch.float)) 
        
        self.J2 = nn.Parameter(torch.tensor(J[1],  dtype=torch.float)) #layer 2
        self.b2 = nn.Parameter(torch.tensor(B[1],  dtype=torch.float)) 
        
        self.Wx = nn.Parameter(torch.tensor(W[0],  dtype=torch.float)) #read gate
        self.Wh = nn.Parameter(torch.tensor(W[1],  dtype=torch.float)) 
        self.bW = nn.Parameter(torch.tensor(bW[0], dtype=torch.float))
        
        self.Ux  = nn.Parameter(torch.tensor(U[0],  dtype=torch.float)) #update gate
        self.Uh  = nn.Parameter(torch.tensor(U[1],  dtype=torch.float))
        self.bU = nn.Parameter(torch.tensor(bU[0], dtype=torch.float))       
                         
        self.bA = nn.Parameter(torch.tensor(bW[1], dtype=torch.float)) #(fixed) bias for plastic syn    
        
        self.f = f
        self.fOut = fOut        
        self.loss_fn = F.binary_cross_entropy 
        self.acc_fn = binary_classifier_accuracy 
        
        self.reset_state() #hidden unit activity 
                        
        
    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""

        c = torch.addmv(self.bA, self.A, x) #apical input (plastic synapses)
        d = torch.addmv(self.b1, self.J1, x) #basal input (fixed weight synapses)
        r = torch.sigmoid( torch.addmv(torch.addmv(self.bW, self.Wx, x), self.Wh, self.h) ) #read gate (recurrent!)
        a1 = r*d + (1-r)*c #hidden layer activation. r=1 read from fixed syn, r=0 read from plastic 
        self.h = self.f( a1 ) #hidden layer output          
        
        u = torch.sigmoid( torch.addmv(torch.addmv(self.bU, self.Ux, x), self.Uh, self.h) ) #update gate   
        self.A = u.unsqueeze(1)*self.A + torch.ger(1-u, x) #plastic weight update. u=1 retain, u=0 overwrite
                
        a2 = torch.addmv(self.b2, self.J2, self.h) #output layer activation
        y = self.fOut( a2 ) #output layer output
        
        if debug:
            execvars = locals()
            execvars = {var:execvars[var].clone().detach() for var in ('c','d','r','a1','u','a2','y')}
            execvars['h'] = self.h.clone().detach()
            execvars['A'] = self.A.clone().detach()
            execvars['WA'] = self.J1.clone().detach() + self.A.clone().detach()
            return execvars
        return y


class GSP1Recall(GSP1):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.tanh):
        """Change default arguments, pass through to parent constructor"""
        super(GSP1Recall,self).__init__(init, f, fOut)
        self.loss_fn = nan_mse_loss
        self.acc_fn = nan_recall_accuracy


    
    
#%%##############
### Recurrent ###
#################
        
class VanillaRNN(StatefulBase):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid):
        super(VanillaRNN,self).__init__()
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            W,b = random_weight_init([Nx,Nh,Nh,Ny], bias=True)
        else:
            W,b = init
                
        self.Wx = nn.Parameter(torch.tensor(W[0],  dtype=torch.float)) #input weights
        self.Wh  = nn.Parameter(torch.tensor(W[1],  dtype=torch.float)) #recurrent weights
        self.b  = nn.Parameter(torch.tensor(b[1], dtype=torch.float)) #recurrent neuron bias
        
        self.Wy  = nn.Parameter(torch.tensor(W[2],  dtype=torch.float)) #output weights
        self.bY = nn.Parameter(torch.tensor(b[2], dtype=torch.float)) #output neuron bias
    
        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy 
        self.f = f
        self.fOut = fOut
        
        self.reset_state()

        
    def reset_state(self):
        self.h = torch.zeros_like(self.b)
           
        
    def forward(self, x):
        #TODO: concat into single matrix for faster matmul
        a1 = torch.addmv(torch.addmv(self.b, self.Wx, x), self.Wh, self.h) #W*x(t) + W*h(t-1) + b
        self.h = self.f( a1 )
        a2 = torch.addmv(self.bY, self.Wy, self.h)
        y = self.fOut(a2)       
        return y
            

class VanillaRNNRecall(VanillaRNN):
    def __init__(self, init, f=torch.tanh, fOut=torch.tanh):
        """Change default arguments, pass through to parent constructor"""
        super(VanillaRNNRecall,self).__init__(init, f, fOut)
        self.loss_fn = nan_mse_loss
        self.acc_fn = nan_recall_accuracy


class LSTM(VanillaRNN):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid):
        super(VanillaRNN,self).__init__()
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            Wi,bi = random_weight_init([Nx,Nh,Nh], bias=True)
            Wf,bf = random_weight_init([Nx,Nh,Nh], bias=True)
            Wo,bo = random_weight_init([Nx,Nh,Nh], bias=True)
            Wc,bc = random_weight_init([Nx,Nh,Nh], bias=True) 
            Wy,by = random_weight_init([Nh,Ny], bias=True)           
        else:
            W,b = init
        
        self.Wix = nn.Parameter(torch.tensor(Wi[0],  dtype=torch.float))
        self.Wih = nn.Parameter(torch.tensor(Wi[1],  dtype=torch.float))
        self.bi  = nn.Parameter(torch.tensor(bi[1],  dtype=torch.float))
        
        self.Wfx = nn.Parameter(torch.tensor(Wf[0],  dtype=torch.float))
        self.Wfh = nn.Parameter(torch.tensor(Wf[1],  dtype=torch.float))
        self.bf  = nn.Parameter(torch.tensor(bf[1],  dtype=torch.float))
        
        self.Wox = nn.Parameter(torch.tensor(Wo[0],  dtype=torch.float))
        self.Woh = nn.Parameter(torch.tensor(Wo[1],  dtype=torch.float))
        self.bo  = nn.Parameter(torch.tensor(bo[1],  dtype=torch.float))
        
        self.Wcx = nn.Parameter(torch.tensor(Wc[0],  dtype=torch.float))
        self.Wch = nn.Parameter(torch.tensor(Wc[1],  dtype=torch.float))
        self.bc  = nn.Parameter(torch.tensor(bc[1],  dtype=torch.float))
        
        self.Wy = nn.Parameter(torch.tensor(Wy[0],  dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0],  dtype=torch.float))
        
        self.f = f
        self.fOut = fOut
        
        self.reset_state()

        
    def reset_state(self):
        self.h = torch.zeros_like(self.bc)
        self.c = torch.zeros_like(self.bc)

    
    def forward(self, x):
        #TODO: concat into single matrix for faster matmul
        ig = torch.sigmoid(torch.addmv(torch.addmv(self.bi, self.Wih, self.h), self.Wix, x)) #input gate
        fg = torch.sigmoid(torch.addmv(torch.addmv(self.bf, self.Wfh, self.h), self.Wfx, x)) #forget gate
        og = torch.sigmoid(torch.addmv(torch.addmv(self.bo, self.Woh, self.h), self.Wox, x)) #output gate
        cIn =       self.f(torch.addmv(torch.addmv(self.bc, self.Wch, self.h), self.Wcx, x)) #cell input
        self.c = fg*self.c + ig*cIn #cell state
        self.h = og*torch.tanh(self.c) #hidden layer activation i.e. cell output  
        
        y = self.fOut( torch.addmv(self.by, self.Wy, self.h) ) 
        return y
    
    
        
class nnLSTM(VanillaRNN): 
    """Should be identical to implementation above, but uses PyTorch internals for LSTM layer instead"""
    def __init__(self, init, f=None, fOut=torch.sigmoid): #f is ignored. Included to have same signature as VanillaRNN
        super(VanillaRNN,self).__init__()
        
        Nx,Nh,Ny = init #TODO: allow manual initialization       
        self.lstm = nn.LSTMCell(Nx,Nh)
        
        Wy,by = random_weight_init([Nh,Ny], bias=True)           
        self.Wy = nn.Parameter(torch.tensor(Wy[0],  dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0],  dtype=torch.float))
        
        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy 
        self.fOut = fOut
        
        self.reset_state()

        
    def reset_state(self):
        self.h = torch.zeros(1,self.lstm.hidden_size)
        self.c = torch.zeros(1,self.lstm.hidden_size)
    
    
    def forward(self, x):
        self.h, self.c = self.lstm(x.unsqueeze(0), (self.h, self.c))
        y = self.fOut( F.linear(self.h, self.Wy, self.by) ) 
        return y
        
    
class LSTMRecall(nnLSTM):
    def __init__(self, init, fOut=torch.tanh):
        """Change default arguments, pass through to parent constructor"""
        super(LSTMRecall,self).__init__(init, fOut)
        self.loss_fn = nan_mse_loss
        self.acc_fn = nan_recall_accuracy

            
        
        
        
        
        
        
