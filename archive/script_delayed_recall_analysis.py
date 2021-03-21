import sys
import torch.nn.functional as F
import numpy as np
import joblib
import matplotlib.pyplot as plt

from dt_utils import Timer
from plastic import PlasticRecall
from data_utils import generate_delayed_recall_data, generate_delayed_recall_from_pairs, \
                       recall_chance, generate_arithmetic_data_onehot
from neural_net_utils import random_weight_init, plot_train_perf

#%%
fname = 'PlasticArithmetic_standard_R={}' #UPDATE
for R in [1]:  #UPDATE 
    net = joblib.load(fname.format(R)+'.pkl') 
#    testData = generate_delayed_recall_data(T=5000, d=net.w1.shape[1]-2, R=R, intertrial=0)
    
#    testData = joblib.load(fname.format(R)+'_trainData.pkl')
#    testData = generate_delayed_recall_from_pairs(testData, T=5000)
    
    testData = generate_arithmetic_data_onehot(T=5000, d=net.w1.shape[1]-2, R=R, intertrial=0)
    
    title = 'R={} ({:.4}%) \n $\lambda={:.3}, \eta={:.3}$, '.format(R, net.accuracy(testData)*100, net.lam, net.eta)
    plot_train_perf(net, recall_chance(testData), title=title )
    
#    plt.savefig(fname.format(R)+'.png') 
    
#%%
trainData = joblib.load(fname.format(R)+'_trainData.pkl')
result = []
with Timer():
    for n in range(100):
        testData = generate_delayed_recall_data(T=5000, d=net.w1.shape[1]-2, R=R, intertrial=0)
#        testData = generate_delayed_recall_from_pairs(trainData, T=5000)
        result.append( net.accuracy(testData) )
    result = np.array(result)
    print( result.mean() )
    print( result.std()  )

#%%
plt.bar([1,2], [result.mean(), from_pairs.mean()], yerr=[result.std(), from_pairs.std()] )
plt.xticks([1,2])
plt.gca().set_xticklabels(['random', 'from_training'])
plt.ylabel('Generalization accuracy')
plt.title('Standard training, R=1')

