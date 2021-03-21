import numpy as np
import copy
from scipy.special import expit

from neural_net_utils import random_weight_init, check_dims

######################
### Nonlinearities ###
######################

class Nonlin(object):
    @staticmethod
    def f(x):
        raise NotImplementedError()
        
    @staticmethod
    def fp(x):
        raise NotImplementedError()
    
    
class Relu(Nonlin):    
    @staticmethod
    def f(x): 
        return np.maximum(0,x)  
    
    @staticmethod  
    def fp(x):
        return x>0
   
    
class Sigmoid(Nonlin):
    @staticmethod
    def f(x):
        return expit(x)
    
    @staticmethod
    def fp(x):
        return expit(x)*(1-expit(x))
    
    
class Tanh(Nonlin):
    pass #TODO

#%%###################       
### Loss functions ###
######################
    
class Loss(object):
    #TODO: make callable instead of defining fn
    def __init__(self, Nonlin):
        self.fpOut = Nonlin.fp
        
    def fn(self, y, yHat):
        raise NotImplementedError()
        
    def delta_L(self, aL, y, yHat):
        raise NotImplementedError()

        
class Quadratic(Loss):
    #Only for scalar network output
    def fn(self, y, yHat):
        return 0.5*(yHat-y)**2

    def delta_L(self, aL, y, yHat):
        return (yHat-y) * self.fpOut(aL)


class CrossEntropy(Loss):
    #Only for scalar network output
    def fn(self, y, yHat):
        return np.nan_to_num(- y*np.log(yHat) - (1-y)*np.log(1-yHat))

    def delta_L(self, aL, y, yHat):
        return (yHat-y)

#%%###############
### Optimizers ###
##################
class Optimizer(object):
    def __init__(self, **hyperparams):
        pass
    
    def run_epoch(trainData):
        pass
    
    
class Adam(Optimizer):
    def __init__(self, alpha=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.freeze_weights()
    
    def freeze_weights():
        pass
    
    def run_epoch(trainData):
        pass

#%%#############
### Networks ###
################
        
class FeedforwardNet(object):
    """ Vanilla binary classifier feedforward neural network.
    (Trains ~2x faster than PyTorch for 1 hidden layer w/ 100 units...)
    """
    
    def __init__(self, W, B=False, f=Sigmoid, fOut=Sigmoid, Loss=CrossEntropy, name='Feedforward'):
        """
        W: list of scalars to specify layer dimensions with random init or 
           list of numpy arrays to specify initial W matrices
        B: if W is list of scalars, this is True or False 
           if W is list of arrays, this is list of 1-D arrays or None
           (either False or None initializes B to zeros and does not update it during training)
        """ 
        
        if np.all([np.isscalar(w) for w in W]):
            self.W, self.B = random_weight_init(W, bias=B)
        else: 
            dims = check_dims(W)
            self.W = copy.deepcopy(W) #don't think deepcopy is necessary but I'm paranoid
            if B:
                self.B = copy.deepcopy(B)
            else:
                self.B = [np.zeros(dim) for dim in dims[1:]]

        
        self.L = len(self.W)+1 #number of layers        
        self.fOut = fOut.f     #output layer nonlinearity
        self.fpOut = fOut.fp
        self.f = f.f           #hidden unit nonlinearity
        self.fp = f.fp        
        self.loss = Loss(fOut) #loss fn and delta for final layer  
        self.name = name       #for printing, etc        
        self.hist = None       #will be initialized once training starts
        
        self.freezeW = None #multiply by binary mask to freeze weights during training       
        if B: 
            self.freezeB = None
        else:                
            self.freezeB = [np.zeros_like(b) for b in self.B] 
                
            
    def freeze_weights(self, layers, option=None, fraction=None):
        '''
        layers: list of hidden unit layer indices (e.g. 0 = input layer, 
                                                        1 = first hidden layer, 
                                                        self.L = output layer)
        option: list of {'in', 'out', 'both'}, one entry per entry in layers list
        fraction: list of floats f between 0 and 1 to freeze top f of units in corresponding layer
        '''
        if len(layers) == 0:
            self.freezeW = None
            return
        
        if option is None:
            option = ['both' for _ in range(len(layers))]
        if fraction is None:
            fraction = [1 for _ in range(len(layers))]
               
        self.freezeW = [np.ones_like(w) for w in self.W]
        for l, opt, f in zip(layers, option, fraction):
            if l == 0 and opt != 'out':
                raise ValueError("Must have option='out' for first layer")
            if l == self.L-1 and opt != 'in':
                raise ValueError("Must have option='in' for final layer")
                    
            try:
                n = int(f*self.W[l].shape[1])
            except: #final layer
                n = int(f*self.W[l-1].shape[0])
                        
            if opt == 'in':
                self.freezeW[l-1][:n] = 0
            elif opt == 'out':
                self.freezeW[l][:,:n] = 0
            elif opt == 'both':
                self.freezeW[l-1][:n] = 0
                self.freezeW[l][:,:n] = 0
        
                
    def freeze_bias(self, layers, fraction=None):
        if fraction is None:
            fraction = [1 for _ in range(len(layers))]
        
        self.freezeB = [np.ones_like(b) for b in self.B]
        for l, f in zip(layers, fraction):
            if l == 0:
                raise ValueError("First layer (index 0) has no bias")
            n = int(f*len(self.B[l-1]))
            self.freezeB[l-1][:n] = 0
            
           
    def feedforward(self, x):
        h = [x]
        a = []
        for w,b in zip(self.W[:-1], self.B[:-1]):
            a.append( np.dot(w,h[-1]) + b )
            h.append( self.f(a[-1]) )  
        a.append( np.dot(self.W[-1],h[-1]) + self.B[-1]) #do last layer separately
        h.append( self.fOut(a[-1]) ) #in case output nonlin different from hidden nonlin       
        return a, h
 
    
    def backprop(self, x, y):
        """Compute gradient dE/dW given one datapoint (x,y) """
        dEdW = [np.zeros_like(w) for w in self.W] 
        dEdB = [np.zeros_like(b) for b in self.B]        
        a, h = self.feedforward(x)
 
        delta = self.loss.delta_L(a[-1], y, h[-1])
        dEdW[-1] = np.outer(delta, h[-2])
        dEdB[-1] = delta
        for l in range(2, self.L): #l is the *negative* layer index so this actually loops backwards
            delta = np.dot(self.W[-l+1].T, delta) * self.fp(a[-l])
            dEdW[-l] = np.outer(delta, h[-l-1])
            dEdB[-l] = delta

        if self.freezeW:
            dEdW = [w*m for w,m in zip(dEdW, self.freezeW)]
        if self.freezeB:
            dEdB = [b*m for b,m in zip(dEdB, self.freezeB)]
        return dEdW, dEdB

    
    def average_loss(self, data):        
        return np.sum( [self.loss.fn(y, self.feedforward(x)[1][-1]) for x,y in data] 
                     )/np.float(len(data)) 
        
    
    def accuracy(self, data):
        return np.sum( [np.round(self.feedforward(x)[1][-1]) == y for x,y in data] 
                     )/np.float(len(data))

    
    def evaluate(self, data):
         return [np.round(self.feedforward(x)[1][-1]) for x,_ in data]   

    
    def _sgd_epoch(self, trainData, gam):
        """Run one epoch of SGD (iterates once through all of the training data, 
        one data point at a time)"""
        for x,y in trainData:
            dEdW, dEdB = self.backprop(x,y)
            self.W = [w - gam*dw for w, dw in zip(self.W, dEdW)]   
            self.B = [b - gam*db for b, db in zip(self.B, dEdB)]   


#    def train(self, trainData, testData=None, epochs=10):
#        self._monitor_init(trainData, testData)
#        self.optimizer.reset()
#        
#        for epoch in range(epochs):
#            self._train_epoch(trainData)
#            self._monitor(trainData, testData):
#        return self.hist
#             
#    
#    def _train_epoch(trainData):
#        self.optimizer.run_epoch(trainData)
        
    
    def _adam_epoch(self, M,V,t, trainData, alpha,b1,b2,eps):
        """Run one epoch of the Adam training algorithm"""
        for x,y in trainData:
            t += 1
            
            dEdW, dEdB = self.backprop(x,y)
            alpha_t = alpha*np.sqrt(1-b2**t)/(1-b1**t)     

            M[0] = [b1*m + (1-b1) * g    for m,g in zip(M[0], dEdW)]            
            V[0] = [b2*v + (1-b2) * g**2 for v,g in zip(V[0], dEdW)]  
            self.W = [w - alpha_t*m/(np.sqrt(v)+eps) for w, m, v in zip(self.W, M[0], V[0])]  
            
            M[1] = [b1*m + (1-b1) * g    for m,g in zip(M[1], dEdB)]
            V[1] = [b2*v + (1-b2) * g**2 for v,g in zip(V[1], dEdB)]            
            self.B = [b - alpha_t*m/(np.sqrt(v)+eps) for b, m, v in zip(self.B, M[1], V[1])]   
        return M,V,t
    
    
    def sgd(self, trainData, testData=None, epochs=10, gam=0.1):
        """Stochastic gradient descent, using one data point per iteration (no minibatching) 
        and sequential processing
        trainData: list of (x,y) tuples
        """        
        self._monitor_init(trainData, testData)
        for epoch in range(epochs):
            self._sgd_epoch(trainData, gam)
            self._monitor(trainData, testData)           
        return self.hist
         
    
    def adam(self, trainData, testData=None, epochs=10, earlyStop=False,
                   alpha=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """Adam optimizer: https://arxiv.org/pdf/1412.6980v8.pdf, using one data point 
        per iteration (no minibatching)
        trainData: list of (x,y) tuples
        """              
        self._monitor_init(trainData, testData)
        
        M = [[np.zeros_like(w) for w in self.W], 
             [np.zeros_like(b) for b in self.B]] 
        V = copy.deepcopy(M)
        t = 0
        for epoch in range(epochs):
            M,V,t = self._adam_epoch(M,V,t, trainData, alpha,b1,b2,eps) 
            self._monitor(trainData, testData)
            if earlyStop:#TODO: should really be doing this on a validation set...
#                if np.abs(self.hist['train_loss'][-1] - self.hist['train_loss'][-2]) < 0.01:
                if self.hist['train_acc'][-1] == 1:
                    break
        return self.hist
    

    def _monitor_init(self, trainData, testData=None):
        if self.hist is None:
            self.hist = {'epoch'      : -1,
                         'train_loss' : [], 
                         'train_acc'  : []}
            if testData:
                self.hist['test_loss'] = []
                self.hist['test_acc'] = []
            self._monitor(trainData, testData)
        else:
            print('Network already partially trained. '
                  'Continuing from epoch {}'.format(self.hist['epoch']))   
    
    
    def _monitor(self, trainData, testData=None):
        self.hist['epoch'] += 1
        
        loss = self.average_loss(trainData)  
        acc = self.accuracy(trainData)
        self.hist['train_loss'].append( loss )
        self.hist['train_acc'].append( acc )
        displayStr = 'Epoch:{} train_loss:{} train_acc:{}'.format(self.hist['epoch'], loss, acc)  

        if testData:
            loss = self.average_loss(testData)  
            acc = self.accuracy(testData)         
            self.hist['test_loss'].append( loss )
            self.hist['test_acc'].append( acc )
            displayStr += ' test_loss:{} test_acc:{}'.format(loss, acc)  
      
        print(displayStr)



class FastWeights(FeedforwardNet):
    """NOTE: In retrospect, I think this implementation is horribly wrong. I need to do backprop *through time*
    over the entire (length-T) training set but here backprop is being executed on each datapoint in isolation
    TODO: what was I doing and how was it managing to kinda-work???
    """
    def __init__(self, W, B=False, f=Sigmoid, fOut=Sigmoid, Loss=CrossEntropy, name='FastWeights', 
                 lam=0, eta=1):
        super(FastWeights, self).__init__(W, B, f, fOut, Loss, name)              
        self.init_A() #fast weights matrix
        self.lam = lam
        self.eta = eta


    def init_A(self):
        self.A = [np.zeros_like(w) for w in self.W]


    def update_A(self, h):
        for l in range(self.L-2): #leave A[-1] at 0
            self.A[l] = self.lam*self.A[l] + self.eta*np.outer(h[l+1], h[l])   
            
            
    def feedforward(self, x):
        W = self.W               
        self.W = [w+fw for w, fw in zip(self.W, self.A)]
        a,h = super(FastWeights, self).feedforward(x)
        self.update_A(h)               
        self.W = W        
        return a, h
    

    def backprop(self, x, y):
        W = self.W               
        self.W = [w+fw for w, fw in zip(self.W, self.A)]
        dEdW = super(FastWeights, self).backprop(x,y)       
        self.W = W        
        return dEdW      


    def _sgd_epoch(self, trainData, gam):
        self.init_A()
        super(FastWeights, self)._sgd_epoch(trainData, gam)
        
    
    def _adam_epoch(self, M,V,t, trainData, alpha,b1,b2,eps):
        self.init_A()
        return super(FastWeights, self)._adam_epoch(M,V,t, trainData, alpha,b1,b2,eps)        
    

    def average_loss(self, data):
        A = self.A
        self.init_A()
        E = super(FastWeights, self).average_loss(data)
        self.A = A
        return E
    
    
    def accuracy(self, data):
        A = self.A
        self.init_A()
        acc = super(FastWeights, self).accuracy(data)
        self.A = A
        return acc


    def evaluate(self, data):
        A = self.A
        self.init_A()
        yHat = super(FastWeights, self).evaluate(data)
        self.A = A
        return yHat
        
    
