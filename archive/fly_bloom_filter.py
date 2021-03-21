import torch
import numpy as np
import matplotlib.pyplot as plt
from data import generate_recog_data, recog_chance
from net_utils import binary_thres_classifier_accuracy

#%%
m=350 #filter size (m=350 <=> 350 KCs projectng to MBON_alpha'3)
r=6 #sampling rate random projection (r=6 <=> each KC gets input from ~6 PNs)
k=17 #number of hash fns (k=17 <=> top 5% of KCs activated for odor)   
d=50 #dimension of input (d=50 <=> 50 PNs projecting to KCs)


#m=25*50 #same number of plastic synapses as in HebbNet
#r=6 
#k=2 #tried 1,2,4,10... 2 works best. 
#d=25 #same size input as for HebbNet 

#random projection matrix
M = torch.zeros(m,d) 
for row in M:
    row[torch.randperm(d)[0:r]] = 1
    
    
def fly_bloom_filter(testData, delta=0, eps=0):
    """
    https://www.pnas.org/content/pnas/115/51/13093.full.pdf
    From paper: for temporal version, delta = 0.4, eps = 0.01
    """   
    #TODO: vary threshold for reporting recognition
                 
#    m=350 #filter size (m=350 <=> 350 KCs projectng to MBON_alpha'3)
#    r=6 #sampling rate random projection (r=6 <=> each KC gets input from ~6 PNs)
#    k=17 #number of hash fns (k=17 <=> top 5% of KCs activated for odor)   
#    d = testData.tensors[0].shape[1] #dimension of input (d=50 <=> 50 PNs projecting to KCs)
#
#    M = torch.zeros(m,d) #random projection matrix
#    for row in M:
#        row[torch.randperm(d)[0:r]] = 1
        
    B = torch.ones(m) #initialize Bloom filter    
    familiarity = torch.empty_like(testData.tensors[1])
    for t, (x,y) in enumerate(testData):
        
        KC = torch.mv(M,x) #Kenyon cell activation    
        idx = torch.sort(KC)[1]
        active = idx[:k] #top k KCs are activated 
        silent = idx[k:] #the rest are inhibited 
        
        familiarity[t] = 1-B[active].sum()/k #calculate familiarity score
        
        B[active] *= delta 
        B[silent] += eps*(1-B[silent])
        
    return familiarity


#%%
thres=0.5   

simple = []
temporal = []
chance = []
#Rs = range(1,20) + list(np.logspace(np.log10(20),np.log10(499)).astype(int))
Rs = list(np.logspace(np.log10(1),np.log10(499), 5).astype(int))
for R in Rs:
    T = R*20

    testData = generate_recog_data(T=T, R=R, d=d, interleave=True, multiRep=True)
    chance.append( recog_chance(testData) )

    out = fly_bloom_filter(testData) 
    acc = binary_thres_classifier_accuracy(out, testData.tensors[1], thres=thres)   
    simple.append( acc.item() )
      
    out = fly_bloom_filter(testData, delta=0.4, eps=0.01) #"temporal" filter. eps, delta from paper 
    acc = binary_thres_classifier_accuracy(out, testData.tensors[1], thres=thres)   
    temporal.append( acc.item() )
      
#%
plt.figure() 
plt.plot(Rs, simple, label='simple')
plt.plot(Rs, temporal , label='temporal') 
plt.plot(Rs, chance, 'k--', label='chance')  
plt.legend()    
plt.title('T={}'.format(T))
plt.ylim((0,1))
    
#%%


    