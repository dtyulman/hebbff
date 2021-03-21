#Built-in and external imports
import math
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

#My imports
from data import generate_recog_data_batch, generate_recog_data
from networks import HebbNet
from plotting import plot_train_perf, plot_recog_generalization, plot_recog_positive_rates, change_reset_fn_to_burned_in
from net_utils import binary_classifier_accuracy, load_from_file

def generate_juri_data(T=100):
    """Generate 100 novel, then 50/50 familiar/novel """
    original = generate_recog_data_batch(T=T, d=d, R=T+1, P=0, batchSize=1) #generate T original datapoints (set probability of repeat P=0)
    familiar = original[torch.randperm(T)[:T/2]] #randomly sample T/2 points from original dataset
    novel = generate_recog_data_batch(T=T/2, d=d, R=0, P=0, batchSize=1).tensors #generate T/2 novel datapoints
    shuffle = torch.randperm(T) #shuffle the order of the test set
    testX = torch.cat((familiar[0], novel[0]))[shuffle] #concatenate the familiar and novel into one dataset and shuffle
    testY = torch.cat((torch.ones_like(familiar[1]), torch.zeros_like(novel[1])))[shuffle]
    data = TensorDataset(torch.cat((original.tensors[0], testX)), 
                         torch.cat((original.tensors[1], testY))) #concantenate original+test and make into TensorDataset
    return data

#%% Make or load network
d = 25 #input layer dimension
N = 25 #hidden layer dimension

load=True #toggle this
if load:
    net = load_from_file('HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl') # Load 
else:
    net = HebbNet([d, N, 1]) # Make 

# Change loss and acc to look at only 2nd half of data
net.loss_fn = lambda out, y: F.binary_cross_entropy(out[len(y)/2:], y[len(y)/2:]) 
net.acc_fn = lambda out, y: binary_classifier_accuracy(out[len(y)/2:], y[len(y)/2:])

# Makes the reset_state() function put the plastic synapses into steady state. Otherwise, it resets them
# to zero at the start of each trial. This doesn't make much of a difference for the continual task, but 
# decreases performance by ~5-10% for Juri's task. Also, appears to change the network's method to solve 
# the task since training without burnin gives better performance on Juri's task but fails on the continal 
# task. Training with burnin gives worse performance on Juri's task and above-chance performance on continual. 
burnin=True #toggle this
gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, multiRep=False)
if burnin:
    change_reset_fn_to_burned_in(net, gen_data, burnT=5000)
    
    
#%% Train
train='infinite' #toggle this
if not train:
    pass
elif train == 'infinite' :        
    net.fit('infinite', generate_juri_data) #can't really find the gradient unless net is pre-trained
elif train == 'curriculum':
    T0 = 20 #first training set has only 4 novel stimuli
    Tf = 100 #stop if/when we get to 100 novel stimuli
    net.fit('curriculum', generate_juri_data, increment=lambda T:T+2, R0=T0, Rf=Tf) 
else:
    raise NotImplementedError
    
#%% Test and plot
testData = generate_juri_data()
testBatch = [dat.squeeze() for dat in testData.tensors] #need to reshape data back to shape [T,d] for bad implementation reasons 
acc = net.accuracy(testBatch)
print('Accuracy: {}'.format(acc))
fracFamiliar = testBatch[1][len(testBatch[1])/2:].mean()
fig, ax = plot_train_perf(net, chance=fracFamiliar, testAcc=acc)

#%% Plot generalization for comparison
plot_recog_generalization(net, gen_data, burnin=burnin)



