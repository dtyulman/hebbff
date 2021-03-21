"""This has been moved to and rewritten in hebbian.py. Keeping this file around for 
loading old .pkl files"""
 
import torch
from torch import nn
import torch.nn.functional as F

from neural_net_utils import check_dims, random_weight_init


class PlasticNet(nn.Module): #TODO: rewrite to use neural_net_utils.NeuralNetBase
    """Lightweight implementation of fastweights.py. Simplify by hard-coding:
        - Three layers (input, hidden, output), with weights and bias
        - Plastic weights only between input and hidden layer
        - Sigmoid activations
        - Binary cross-entropy loss
        - Uses torch.* math instead of pre-baked torch.nn.* e.g. nn.Linear
        - Adam optimizer
        - Hard-coded variations of plasticity:
            -- Pure Hebbian
            -- Learned Hebbian weight + learning rate
            -- Parameterized Hebbian
            -- Hebbian Network
    """
    
    def __init__(self, W, B=None, f=torch.sigmoid, fOut=torch.sigmoid, lossfn=F.binary_cross_entropy,
                 name='PlasticNet', **hebb_args):
        """
        W: list of scalars to specify layer dimensions with random init or 
           list of torch tensors to specify initial W matrices
        B: if W is list of scalars, this is ignored and bias is random init
           if W is list of tensors, this is also list tensors 
        """ 
        super(PlasticNet, self).__init__()
        
        if all(type(d) == int for d in W):
            W, B = random_weight_init(W, bias=True)        
        check_dims(W,B)

        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.b1 = nn.Parameter(torch.tensor(B[0], dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(B[1], dtype=torch.float))          
        self.lossfn = lossfn 
        self.f = f
        self.fOut = fOut
                
        self.reset_A() #plastic weights matrix
        self.init_hebb(**hebb_args) #parameters of Hebbian rule
            
        self.name = name
        self.hist = None
                
        
    def forward(self, x, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        a1 = torch.addmv(self.b1, self.w1+self.A, x) #hidden layer activation
        h = self.f( a1 ) 
        
        a2 = torch.addmv(self.b2, self.w2, h) #output layer activation
        y = self.fOut( a2 ) 
        
        self.update_A(x,h)        

        if debug:
            return a1, h, a2, y 
        return y
    
    
    def reset_A(self):
        self.A = torch.zeros_like(self.w1) 

        
    def init_hebb(self, eta, lam, positiveEta=False):
        self.lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.eta = nn.Parameter(torch.tensor(eta)) #Hebbian learning rate     
        self.positiveEta = positiveEta #force Hebbian (not antiHebbian) rule     
    
    
    def hebb(self, pre, post):
#        if self.positiveEta:       
#            eta = torch.abs(self.eta) #disallow anti-hebbian rule
#        else:
#            eta = self.eta

        return self.eta*torch.ger(post,pre)
        
    
    def update_A(self, pre, post):
        self.A = self.lam*self.A + self.hebb(pre, post) #hebbian update

                 
    def fit(self, trainData, epochs=100, earlyStop=True):
        self._monitor_init(trainData)
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(epochs): 
            #TODO: minibatch this (use *contiguous* batches!)
            optimizer.zero_grad()
            out = self.evaluate(trainData)
            loss = self.average_loss(trainData, out=out)            
            loss.backward()
            optimizer.step()

            self._monitor(trainData, out=out, loss=loss)              
            if earlyStop and sum(self.hist['train_acc'][-5:]) >= 4.99:
                print('Converged. Stopping early.')
                break #early stop
            

    def evaluate(self, data):
#        A = self.A.clone().detach()
        self.reset_A()
        out = torch.empty(len(data), self.w2.shape[0])
        for i,x in enumerate(data.tensors[0]):
            out[i] = self(x)
#        self.A = A
        return out


    def accuracy(self, data, out=None):
        if out is None:
            out = self.evaluate(data)
        return (out.round() == data.tensors[1]).float().mean()
   
    
    def average_loss(self, data, out=None):
        if out is None:
            out = self.evaluate(data)
        return self.lossfn(out, data.tensors[1])
    
    
    @torch.no_grad()
    def _monitor_init(self, trainData):
        if self.hist is None:
            self.hist = {'epoch'      : -1,
                         'train_loss' : [], 
                         'train_acc'  : [],
                         'lam' : [],
                         'eta' : []
                         }
            self._monitor(trainData)
#        else:
#            print('Network already partially trained. '
#                  'Continuing from epoch {}'.format(self.hist['epoch']))   
    
    @torch.no_grad()
    def _monitor(self, data, out=None, loss=None, acc=None):
        self.hist['epoch'] += 1 
        
        if self.hist['epoch']%10 == 0: #TODO: allow choosing monitoring interval
            if out is None:
               out = self.evaluate(data)
            if loss is None:
               loss = self.average_loss(data, out)
            if acc is None:
               acc = self.accuracy(data, out)    
            
            self.hist['train_loss'].append( loss.item() )
            self.hist['train_acc'].append( acc.item() )            
#            self.hist['params'].append( self.state_dict() ) #TODO: need scalable way to save netwk
            self.hist['lam'].append( self.lam.item() )
            self.hist['eta'].append( self.eta.item() )

            
            displayStr = 'Epoch:{} train_loss:{} train_acc:{}'.format(self.hist['epoch'], loss, acc)         
            print(displayStr)
      

#%%

class PlasticRecall(PlasticNet):    
    def __init__(self, W, B, f=torch.sigmoid, lossfn=F.cross_entropy, name='PlasticRecall', **hebb_args):
        """
        W: list of scalars to specify layer dimensions with random init or 
           list of torch tensors to specify initial W matrices
        B: if W is list of scalars, this is ignored and bias is random init
           if W is list of tensors, this is also list tensors 
        """ 
        super(PlasticRecall, self).__init__(W, B, f=f, lossfn=lossfn, name=name, **hebb_args)
        del self.fOut
                               
        
    def forward(self, x):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        a1 = torch.addmv(self.b1, self.w1+self.A, x) #hidden layer activation
        h = self.f( a1 )         
        a2 = torch.addmv(self.b2, self.w2, h) #output layer activation        
        
        self.update_A(x,h)        
        
        return a2.unsqueeze(0)
    
           
    def accuracy(self, data, out=None):
        if out is None:
            out = self.evaluate(data)
        not_ignore_idx = [data.tensors[1] != -100]
        return (out[not_ignore_idx].argmax(1) == data.tensors[1][not_ignore_idx]).float().mean()
   
                
            
#%% Variations of plasticity rules
        
class LearnedHebb(PlasticNet): 
    """Synapse parameterized by a neural network"""
    def init_hebb(self, initW=None, initB=None, Nh=30, ):
        self.synapse = Synapse(Nh)


    def update_A(self, pre, post):
        return self.synapse(self.A, pre, post)
    
    


class TaylorHebb(PlasticNet):
    """Synapse fit by a second-order Taylor expansion"""
    def init_hebb(self, eta, lam, learnHebb):
        self.lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.eta = nn.Parameter(torch.tensor(eta)) #Hebbian learning rate 
        
    def hebb(self, pre, post):
        return self.eta[0] \
                + self.eta[1]*post.unsqueeze(1).expand_as(self.A) \
                + self.eta[2]*pre.expand_as(self.A) \
                + self.eta[3]*torch.ger(post,pre)


#%%  
class Synapse(nn.Module):
    def __init__(self, Nh):
        super(Synapse, self).__init__()
        self.w1 = nn.Parameter(torch.randn(Nh, 2))
        self.b1 = nn.Parameter(torch.randn(Nh))
        self.w2 = nn.Parameter(torch.randn(1, Nh))
        self.b2 = nn.Parameter(torch.randn(1))
       
        
    def forward(self, A, pre, post):
        """
        Inputs:
            A: MxN tensor of existing plastic weights
            pre:  length-N tensor of activations in layer L (presynaptic)
            post: length-M tensor of activations in layer L+1 (postsynaptic)
        Returns:
            MxN tensor of updated synaptic weights for all synapses between layers L and L+1
        """
        _post, _pre = torch.meshgrid(post,pre)
        
        #TODO: add manual nonlin features e.g. pre*post, pre^2, post^2
        l0 = torch.stack((A.flatten(), _pre.flatten(),_post.flatten())).t()
        l1 = torch.tanh( F.linear(l0, self.w1, self.b1) )
        l2 = torch.tanh( F.linear(l1, self.w2, self.b2) )
        
        return l2.reshape(len(post), len(pre))


        
    
    
    
# %%
'''
#quick monitoring code
avgLoss = torch.tensor(0.)
acc  = torch.tensor(0) 
#loop goes here
    #goes inside loop
    with torch.no_grad():
        avgLoss += loss
        acc += (out.round()==y)[0]
avgLoss /= len(trainData) 
acc = acc.float()/len(trainData)
print('Epoch:{} train_loss:{} train_acc:{}'.format(epoch+1, avgLoss, acc))
'''