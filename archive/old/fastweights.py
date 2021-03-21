import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neural_net_utils import check_dims, random_weight_init

#%%
class InitializedLinear(nn.Linear):
    def __init__(self, initW, initB=None):
        self.initW = torch.as_tensor(initW)
        out_features, in_features = initW.shape
        if initB is None:
            bias = False 
            self.initB = None
        else: 
            bias = True
            self.initB = torch.as_tensor(initB)
        super(InitializedLinear, self).__init__(in_features, out_features, bias)
    
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.copy_(self.initW)
            if self.bias is not None:
                self.bias.copy_(self.initB)
            
                           

class FeedforwardNet(nn.Module):
    def __init__(self, W, B=False, f=nn.Sigmoid(), fOut=nn.Sigmoid(), loss=nn.BCELoss(), name='Feedforward'):
        """
        W: list of scalars to specify layer dimensions with random init or 
           list of numpy arrays to specify initial W matrices
        B: if W is list of scalars, this is True or False 
           if W is list of arrays, this is list of 1-D arrays or None
           (either False or None initializes B to zeros and does not update it during training)
        """ 
        super(FeedforwardNet, self).__init__()

        self.init_layers(*self.parse_layer_init(W,B))
       
        self.f = f        
        self.fOut = fOut
        self.loss = loss 
        #TODO: if fOut(x)=Sigmoid(x) and loss=BCELoss, can set fOut(x)=x and loss=BCEWithLogitsLoss
        #      (but also need to update self.accuracy to take Sigmoid before rounding)
        
        self.hist = None
    
    
    def parse_layer_init(self, W,B):
        if np.all([np.isscalar(w) for w in W]):
            if B:
                W,B = zip([(torch.as_tensor(w), torch.as_tensor(b)) for w,b in random_weight_init(W, bias=True)])
            else:
                W,B = zip([(torch.as_tensor(w), None) for w,_ in random_weight_init(W)])
        else: 
            check_dims(W)
            if not B:
                B = [None for _ in W]
        return W,B
    
    
    def init_layers(self, W,B):
         self.layers = nn.ModuleList([InitializedLinear(w, b) for w,b in zip(W,B)])

       
    def get_W(self):
        return [layer.weight.data for layer in self.layers]
    
    
    def set_W(self, W):
        if len(W) != len(self.layers):
            raise ValueError("Must have len(W) equal to the number of layers. "
                             "Use 'None' to avoid setting an entry")
        for w, i in enumerate(W):
            if w:
                self.layers[i].weight.copy_(w)
             
                
    def get_B(self):
        return [layer.bias.data for layer in self.layers]
    
    
    def set_B(self, B):
        if len(B) != len(self.layers):
            raise ValueError("Must have len(B) equal to the number of layers. "
                             "Use 'None' to avoid setting an entry")
        for b, i in enumerate(B):
            if b:
                self.layers[i].bias.copy_(b)
                
    
    def forward(self, x):
        for L in self.layers[:-1]:
            x = self.f( L(x) ) 
        x = self.fOut( self.layers[-1](x) )
        return x
    
    
    def _optimizer_epoch(self, trainData, optimizer):
        self.train()
        for x,y in trainData:
            optimizer.zero_grad() 
            loss = self.loss(self(x), y)
            loss.backward()
            optimizer.step()
        
    
    def fit(self, trainData, testData=None, epochs=10, optimizer=None):       
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
            
        self._monitor_init(trainData, testData)
        for epoch in range(epochs):
            self._optimizer_epoch(trainData, optimizer)
            self._monitor(trainData, testData)
            
    
    def _monitor_init(self, trainData, testData=None):
        if self.hist is None:
            self.hist = {'epoch'      : -1,
                         'train_loss' : [], 
                         'train_acc'  : []}
            if testData:
                self.hist['test_loss'] = []
                self.hist['test_acc']  = []
            self._monitor(trainData, testData)
        else:
            print('Network already partially trained. Continuing from epoch {}'.format(self.hist['epoch']))   
        
    
    def _monitor(self, trainData, testData=None):
        self.eval()
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
 
    
    def accuracy(self, data):
        '''data is instance of TensorDataset'''
        return ( torch.round(self(data.tensors[0]))==data.tensors[1] ).float().mean()
      
        
    def average_loss(self, data):
        '''data is instance of TensorDataset'''
        return self.loss( self(data.tensors[0]), data.tensors[1] )


#%%     
class LinearWithPlasticity(InitializedLinear):
    def __init__(self, initW, initB=None, lam=0, eta=1):
        super(LinearWithPlasticity, self).__init__(initW, initB)
        self.lam = lam #plastic weights decay multiplier
        self.eta = eta #plastic weights learning rate
        self.init_A()  #plastic weights matrix
        self.syn = Synapse()

        
    def forward(self, x):    
        a = F.linear(x, self.weight+self.A, self.bias)
        self.update_A(x, a) 
        return a
     
        
    def update_A(self, pre, post):
#        torch.addr(self.lam, self.A, self.eta, torch.ones_like(xPost), xPre, out=self.A)
#       listPre = pre.view(-1,1).repeat(1, len(post)).view(-1,1)
#       listPost = post.repeat(len(pre), 1).view(-1,1)
#       listPrePost = torch.cat([listPre, listPost], 1)       
#       self.A += self.syn(listPrePost).view(self.A.shape)
        dA = torch.empty_like(self.A)
        for i in range(len(pre)):
            for j in range(len(post)):
                dA[j,i] = self.syn(pre[i], post[j])
        self.A = self.lam*self.A + self.eta*dA
       
        
    def init_A(self):
        self.A = torch.zeros_like(self.weight)   
        
        
    def reset_parameters(self):
        with torch.no_grad():
            self.init_A()
            self.weight.copy_(self.initW)
            if self.bias is not None:
                self.bias.copy_(self.initB)
                
    
    
class FastWeights(FeedforwardNet):
    """Feedforward net with an added synaptic plasticity. Learned weights W are as in FeedforwardNet, with
    an additional Hebbian (plastic) weight A (updated at every iteration) such that the effective weight 
    matrix is W+A"""
    def __init__(self, W, B=False, f=nn.Sigmoid(), fOut=nn.Sigmoid(), loss=nn.BCELoss(), name='FastWeights',
                 lam=0, eta=1):
        self.lam = lam
        self.eta = eta
        super(FastWeights, self).__init__(W, B, f, fOut, loss, name)  


    def init_layers(self, W,B):
         self.layers = nn.ModuleList([LinearWithPlasticity(w, b, self.lam, self.eta) for w,b in zip(W[:-1],B[:-1])])
         self.layers.append( InitializedLinear(W[-1], B[-1]) ) 

    
    def _optimizer_epoch(self, trainData, optimizer):
       self.init_A()
       super(FastWeights, self)._optimizer_epoch(trainData, optimizer)
         
    
    def get_A(self):
        return [layer.A.data if type(layer)==LinearWithPlasticity else None for layer in self.layers]


    def init_A(self):
        for layer in self.layers:
            if type(layer)==LinearWithPlasticity:
                layer.init_A() 


    def set_A(self, A):
        raise NotImplementedError()
      
        
    def average_loss(self, data):
#            A = get_A()
        yHat = torch.empty_like(data.tensors[1])
        for i,x in enumerate(data.tensors[0]):
            yHat[i] = self(x) 
#            set_A(A)
        return self.loss(yHat, data.tensors[1])
        
    
    def accuracy(self, data):
        yHat = torch.empty_like(data.tensors[1])
        for i,x in enumerate(data.tensors[0]):
            yHat[i] = torch.round(self(x))
        return (yHat == data.tensors[1]).float().mean()
    


class Synapse(nn.Module):
    """Lightweight neural network that learns the Hebbian rule"""
    def __init__(self, inp=2, width=10):
        super(Synapse, self).__init__()
        self.f = torch.tanh
        
        self.w = torch.empty(width,inp, requires_grad=True)
        self.b = torch.empty(width, requires_grad=True)
        torch.nn.init.xavier_normal_(self.w)
        torch.nn.init.normal_(self.b)

        self.w2 = nn.Linear(width, 1)
        
               
    def forward(self, pre, post):
        h = self.w[:,0]*pre + self.w[:,1]*post + self.b
        return self.f( self.w2( self.f( h ) ) )
    
    
    def plot(self):
        import matplotlib.pyplot as plt 
        pre = torch.arange(0,1,.05)
        post = torch.arange(0,1,.05)
        delta = self(pre,post)       
        _pre, _post = torch.meshgrid(pre,post)

        plt.plot_surface(_pre, _post, delta)



#%%#########    
### Misc ###    
############
def weights_to_vector(W, B=None):
    """Useful when computing numerical gradient"""
    #TODO: bias
    V = np.array([])
    for w in W:
        V = np.concatenate((V, w.flatten()))
    return V


def vector_to_weights(V, dims, bias=False):
    #TODO: bias
    W = []
    l = 0 #pointer to leftmost entry of W[0]
    r = 0 #rightmost
    for d in range(len(dims)-1):
        l = r 
        r = r + dims[d+1]*dims[d]
        W.append( V[l:r].reshape(dims[d+1], dims[d]) )
    return W      
    