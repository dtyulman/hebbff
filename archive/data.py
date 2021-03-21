import random
import numpy as np
import torch

from torch.utils.data import TensorDataset
#%%################
### Recognition ###
###################
               
def generate_recog_data(T=2000, d=50, R=1, P=0.5, interleave=True, multiRep=True, xDataVals='+-', softLabels=False):
    """Generates "image recognition dataset" sequence of (x,y) tuples. 
    x[t] is a d-dimensional random binary vector, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            x = data[t-R][0]
            y = 1
        else:
            if xDataVals == '+-': #TODO should really do this outside the loop...
                x = 2*np.round(np.random.rand(d))-1
            elif xDataVals.lower() == 'normal':
                x = np.sqrt(d)*np.random.randn(d)    
            elif xDataVals.lower().startswith('uniform'):
                upper, lower = parse_xDataVals_string(xDataVals)
                x = np.random.rand(d)*(upper-lower)+lower
            elif xDataVals == '01':
                x = np.round(np.random.rand(d))
            else:
                raise ValueError('Invalid value for "xDataVals" arg')           
            y = 0
            
        if softLabels:
            y*=(1-2*softLabels); y+=softLabels               
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)

#data = generate_recog_data(T=20, d=3, R=2, P=0.5, softLabels=0.1)
#print torch.cat( (data.tensors[0], data.tensors[1]), dim=1)

#%%  
def generate_recog_data_batch(T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=True, softLabels=False, xDataVals='+-', device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    if xDataVals == '+-':
        x = 2*torch.rand(T,batchSize,d, device=device).round()-1 #faster than (torch.rand(T,B,d)-0.5).sign()
    elif xDataVals.lower() == 'normal':
        x = torch.randn(T,batchSize,d, device=device)    
    elif xDataVals.lower().startswith('uniform'):
        upper, lower = parse_xDataVals_string(xDataVals)
        x = torch.rand(T,batchSize,d, device=device)*(upper-lower)+lower
    elif xDataVals == '01':
        x = torch.rand(T,batchSize,d, device=device).round()
    else:
        raise ValueError('Invalid value for "xDataVals" arg')  
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()
    if softLabels:
        y = y*0.98 + 0.01
    
    
  

    return TensorDataset(x, y)


class GenRecogClassifyData():
    def __init__(self, d=None, teacher=None, datasize=int(1e4), sampleSpace=None, save=False, device='cpu'):
        if sampleSpace is None:
            x = torch.rand(datasize,d, device=device).round()*2-1
            if teacher is None:
                c = torch.randint(2,(datasize,1), device=device, dtype=torch.float)
            else:
                c = torch.empty(datasize,1, device=device, dtype=torch.float)
                for i,xi in enumerate(x):
                    c[i] = teacher(xi)
                c = (c-c.mean()+0.5).round()
            self.sampleSpace = TensorDataset(x,c)
            if save:
                if type(save) == str:
                    fname = save
                else:
                    fname = 'sampleSpace.pkl'
                torch.save(self.sampleSpace, fname)
        elif type(sampleSpace) == str:
            self.sampleSpace = torch.load(sampleSpace) 
        elif type(sampleSpace) == TensorDataset:
            self.sampleSpace = sampleSpace
            
        self.datasize, self.d = self.sampleSpace.tensors[0].shape            
        
        
    def __call__(self, T, R, P=0.5, batchSize=-1, multiRep=True, device='cpu'):
        if np.isscalar(R):
            Rlist = [R]
        else:
            Rlist = R
        
        squeezeFlag=False
        if batchSize is None:
            batchSize=1
            squeezeFlag=True
        elif batchSize < 0:
            batchSize = self.datasize/T
            
        randomSubsetIdx = torch.randperm(len(self.sampleSpace))[:T*batchSize]
        x,c = self.sampleSpace[randomSubsetIdx]
        x = x.reshape(T,batchSize,self.d)
        c = c.reshape(T,batchSize,1)
        y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
        for t in range(max(Rlist), T):    
            R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
                   
            repeatMask = torch.rand(batchSize)>P   
            if not multiRep:
                repeatMask = repeatMask*(~y[t-R]) #this changes the effective P
              
            x[t,repeatMask] = x[t-R,repeatMask] 
            c[t,repeatMask] = c[t-R,repeatMask]            
            y[t,repeatMask] = 1
         
        y = y.unsqueeze(2).float()
        y = torch.cat((y,c), dim=-1)        
        data = TensorDataset(x,y)
        
        if squeezeFlag:
            data = TensorDataset(*data[:,0,:])
    
        return data
    

def generate_aug_recog_data(T=2000, d=25, k=1, R=1, P=0.5, interleave=True, multiRep=True, device='cpu'):
    data = []
    repeatFlag = False #true if will be repeating item x[t-R]
    r=0 #counter of how long we need to wait until repeating (only used if interleave=False)
    for t in range(T):                
        #decide if repeating
        if interleave:
            repeatFlag = np.random.rand()<P 
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                    
        #generate datapoint   
        if repeatFlag and t>=R and (multiRep or data[t-R][1][0]==0):
            xt = data[t-R][0][:d] #repeat x(t) = x(t-R)
            ftx = np.round(np.random.rand(k))*2-1 #generate new f_t(x_t)
            yt = 1 #report familiar 
            ftRx = data[t-R][0][d:] #report ftx from R timesteps ago i.e f_{t-R}(x(t))
        else:
            xt = np.round(np.random.rand(d))*2-1 #generate new x
            ftx = np.round(np.random.rand(k))*2-1
            yt = 0 #report novel
            ftRx = np.full(k, np.nan) #don't report any ftx 
        x = np.concatenate((xt, ftx))
        y = np.concatenate((np.array([yt]), ftRx))
        
        data.append( (x,y) )
    
    return data_to_tensor(data, device=device)

#%%
def generate_aug_recog_data_batch(T=2000, batchSize=1, d=25, k=1, R=1, P=0.5, interleave=True, multiRep=True, zeroOneVal=False):
    x = torch.rand(T,batchSize,d+k).round()*2-1
    
    if zeroOneVal: #assign value to be 0/1 so I can use BCE loss
        x[:,:,-k:] = torch.rand(T,batchSize,k).round() 
    
    yRec = torch.zeros(T,batchSize, dtype=torch.bool)
    yVal = torch.full((T,batchSize, k), float('nan'))
    
    for t in range(R, T):                
        #decide if repeating
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~yRec[t-R,:])
            
        x[t,repeatMask,:d] = x[t-R,repeatMask,:d] #repeat x(t) = x(t-R)
        yRec[t,repeatMask] = 1 #report familiar 
        yVal[t,repeatMask] = x[t-R,repeatMask,d:] #report aug from x(t-R)
    
    return TensorDataset( x, torch.cat((yRec.unsqueeze(2).float(), yVal), dim=2) )


#%timeit generate_aug_recog_data_batch(T=1024, batchSize=4, d=25, k=2)  
#print torch.cat((data[0], data[1]), dim=2)


#%%###########
### Recall ###
##############
def generate_delayed_recall_data(T=2000, d=50, R=1, intertrial=0, interleave=False):
    """Generates delayed recall dataset. Inputs are d-dimensional binary vectors from {+1, -1}^d
    Output is input delayed by R timesteps 
    
    NOTE: MUST USE neural_net_utils.nan_mse_loss() IF USING THIS DATA GENERATOR. TARGET IS EQUAL
    TO NAN FOR TIMESTEPS WHERE NETWORK OUTPUT CAN BE ARBITRARY (i.e. the first R timesteps, as 
    well as timesteps between the stimulus and the output if using non-interleaved data)
    
    NOTE: R=1, intelreave=True is NOT the same as R=1, interleave=False. 
    Compare: inp: a b c d  and  inp: a . b . c 
             tgt: . a b c       tgt: . a . b .    
    """
    #TODO: inter-trial-interval (only makes sense if interleave=False)
    IGNORE = np.nan*np.empty(d) #special symbol that is ignored when evaluating loss
    data = []
    r=-1 #counter how long ago a to-be-remembered input was presented 
    for t in range(T):
        x = np.round(np.random.rand(d))*2-1
        if t<R:
            y = IGNORE
        else:
            if interleave:
                y = data[t-R][0]
            else:
                if r>=0:
                    y = IGNORE
                else:
                    r=R
                    y = data[t-R][0] 
                r -= 1                
        data.append( (x,y) ) 
    return data_to_tensor(data)
  

def generate_delayed_recall_onehot_data(T=2000, d=50, R=1, intertrial=0, astensor=True, useNull=False):
    """Generates delayed recall dataset. Inputs are d-dimensional one-hot vectors, with additional 
    read and write inputs, so x[t] is (d+2)-dimensional. Output is the index of the category.
    
    useNull=True causes input to be all zeros for timepoints at which writing isn't required
    """
    if useNull:
        raise NotImplementedError('TODO: need to verify this is working properly... training is failing with this option!')
     
    data = []
    
    written = -R-intertrial-1 #time at which vector was written into mem (first vector gets written immediately)
    for t in range(T):
        x = np.zeros(d)
        c = np.random.randint(d)
        if not useNull:
            x[c] = 1 #one-hot input vector
        
        if t == written+R+intertrial+1:
            w,r = 1,0
            written = t
            c_written = c
            x[c] = 1 #one-hot input vector
        elif t == written+R:
            w,r = 0,1
        else:
            w,r = 0,0
                
        if r == 1:
           y = c_written #categorical label target value (note: network still outputs one-hot)
        else:
           y = -100 #default ignore_index for cross-entropy loss
        
        x = np.hstack( (x,r,w) )
        data.append( (x,y) )
        
    if astensor:
        return data_to_tensor(data, y_dtype=torch.long)
    return data


def generate_delayed_recall_from_pairs(data, T=2000):
    w_seq = data.tensors[0][:,-1].nonzero().flatten()
    r_seq = data.tensors[0][:,-2].nonzero().flatten()
    R = (r_seq[0] - w_seq[0]).item()
    
    library = set()
    for n in range(len(r_seq)):
        r_idx = r_seq[n]
        w_idx = w_seq[n]
        x = data.tensors[0][w_idx:r_idx+1]
        y = data.tensors[1][w_idx:r_idx+1]
        library.add((x,y))
        
    x_data,y_data = [],[]
    for _ in range(T/R):
        x,y = random.sample(library,1)[0]
        x_data.append(x)
        y_data.append(y)
    
    generated_data = TensorDataset(torch.cat(x_data), torch.cat(y_data))
    return generated_data

        
def generate_arithmetic_data_onehot(T=2000, d=50, R=1, intertrial=0, fn=sum):
    data = []
    
    written = -R-intertrial-1 #time at which vector was written into mem (first vector gets written immediately)
    listRback = [] #when r==1, contains the previous R categories that were inputs
    for t in range(T):
        x = np.zeros(d)
        c = np.random.randint(d)
        x[c] = 1 #one-hot input vector
        listRback.append(c)

        if t == written+R+intertrial+1:
            w,r = 1,0
            written = t
        elif t == written+R:
            w,r = 0,1
        else:
            w,r = 0,0
        
        if r == 1:
           y = fn(listRback) % d
           listRback = []
        else:
           y = -100 #default ignore_index for cross-entropy loss
        
        x = np.hstack( (x,r,w) )
        data.append( (x,y) )                 
 
    return data_to_tensor(data, y_dtype=torch.long)

  

#%%############
### Helpers ###   
###############
def parse_xDataVals_string(xDataVals):
    assert xDataVals.lower().startswith('uniform')
    delimIdx = xDataVals.find('_')
    if delimIdx > 0:
        assert delimIdx==7
        lims = xDataVals[delimIdx+1:]
        lower = float(lims[:lims.find('_')])
        upper = float(lims[lims.find('_')+1:])
    else:
        lower = -1
        upper = 1
    return upper, lower


def prob_repeat_to_frac_novel(P, multiRep=False):
    if multiRep:
        return P
    n,m = P.as_integer_ratio()
    return 1 - float(n)/(m+n)
    

def check_recognition_data(data, R):
    """Make sure there are no spurious repeats"""
    if len(data) == 0:
        return False
    for i in range(len(data)):
        for j in range(0,i-1):
            if all(data[i][0] == data[j][0]):   
                if i-j != R:
                    print( 'bad R', i, j )
                    return False
                if not data[i][1]:
                    print( 'unmarked', i, j )
                    return False
    return True


def recall_chance(data):
    inp = data.tensors[0][:,:-2].argmax(1)
    out = data.tensors[1]
    out_guess = -100*torch.ones_like(out)
    for c in inp.unique():
        idx = inp==c
        c_out = out[idx]
        c_guess = c_out[c_out!=-100].mode()[0].item()
        out_guess[idx] = c_guess
    
    nReads = data.tensors[0][:,-2:-1].sum()
    chance = (out[out!=-100] == out_guess[out!=-100]).sum()/nReads
    return chance.item()
     
        
def recog_chance(data):
    """Calculates expected performance if network simply guesses based on output statistics
    i.e. the number of zeroes in the data"""
    return 1-np.sum([xy[1] for xy in data], dtype=np.float)/len(data) 


def batch(generate_data, batchsize=1, batchDim=1, **dataKwargs):
    dataList = []
    for b in range(batchsize):
        dataList.append( generate_data(**dataKwargs) )
    x = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim)
    y = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim) 
    return TensorDataset(x,y)


def data_to_tensor(data, y_dtype=torch.float, device='cpu'):
    '''Convert from list of (x,y) tuples to TensorDataset'''
    x,y = zip(*data)
    return TensorDataset(torch.as_tensor(x, dtype=torch.float, device=device), 
                   torch.as_tensor(y, dtype=y_dtype, device=device))
    
    
#%%
#aug = generate_aug_recog_data_batch(T=T, batchSize=1, d=d, k=1, R=R, interleave=True, multiRep=False)
#augRec = TensorDataset(aug.tensors[0], aug.tensors[1][:,:,0])
#print recog_chance(augRec)
#
#aug = generate_aug_recog_data(T=T, d=d, k=1, R=R, interleave=True, multiRep=False)
#augRec = TensorDataset(aug.tensors[0], aug.tensors[1][:,0])
#print recog_chance(augRec)
#
#recog = generate_recog_data_batch(T=T, batchSize=1, d=d, R=R, interleave=True, multiRep=False)
#print recog_chance(recog)
#
#recog = generate_recog_data(T=T, d=d, R=R, interleave=True, multiRep=False)
#print recog_chance(recog)