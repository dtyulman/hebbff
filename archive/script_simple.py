#External imports
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F

#My imports
from data import generate_recog_data_batch
from networks import HebbNet
from plotting import plot_train_perf
from net_utils import binary_classifier_accuracy

def generate_juri_data(T=100):
    """Generate 100 novel, then 50/50 familiar/novel """
    original = generate_recog_data_batch(T=T, d=d, R=T+1, P=0, batchSize=1) #generate T original datapoints (set probability of repeat P=0)
    familiar = original[torch.randperm(T)[:T/2]] #randomly sample T/2 points from original dataset
    novel = generate_recog_data_batch(T=T/2, d=d, R=0, P=0, batchSize=1).tensors #generate T/2 novel datapoints
    shuffle = torch.randperm(T) #shuffle the order of the test set
    testX = torch.cat((familiar[0], novel[0]))[shuffle] #concatenate the familiar and novel into one dataset and shuffle
    testY = torch.cat((torch.ones_like(familiar[1]), torch.zeros_like(novel[1])))[shuffle]
    data = TensorDataset(torch.cat((original.tensors[0], testX)), torch.cat((original.tensors[1], testY))) #concantenate original and test and make into TensorDataset
    return data

#%%
# Make network
d = 25 #input layer dimension
N = 25 #hidden layer dimension
net = HebbNet([d, N, 1]) 
net.loss_fn = lambda out, y: F.binary_cross_entropy(out[len(y)/2:], y[len(y)/2:]) #loss and acc on only 2nd half of data
net.acc_fn = lambda out, y: binary_classifier_accuracy(out[len(y)/2:], y[len(y)/2:])

# Generate artificial data
data = generate_recog_data_batch(T=200, d=d, R=5, P=0.5, batchSize=1) #data dim=[T,1,d] (the extra dimension is necessary for bad implementation reasons)
data = generate_juri_data()

# Train and plot result
net.fit('dataset', data, epochs=1000, filename=None)
plot_train_perf(net, chance=0.5)

# Test
testData = generate_juri_data()
print( net.accuracy(testData) )



