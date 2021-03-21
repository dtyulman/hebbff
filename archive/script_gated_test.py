import joblib
import matplotlib.pyplot as plt

from dt_utils import Timer
from plastic import PlasticNet
from plastic_gated import ThreeGateSynapse, GSP1
from recurrent_nets import VanillaRNN, LSTM, nnLSTM
from data_utils import generate_recognition_data, recog_chance
from neural_net_utils import plot_train_perf, infinite_data_training

#%%
#netwk params
d = 25
Nh = 50
dims = [d, Nh, 1]

#net = ThreeGateSynapse(dims)
#net = GSP1(dims)
#net = VanillaRNN(dims)
#net = PlasticNet(dims, lam=0.5, eta=-0.5, name='antiHebb')
net = nnLSTM(dims)

trainingMethod = 'inf1'
#%%
#data params
R = 5
T = 2000
P = 0.5

maxEpochs=100000
with Timer('{}, training {}'.format(net.name, trainingMethod)): 
    if trainingMethod.startswith('inf'):
        epochsPerBatch = int(trainingMethod[3:])
        infinite_data_training(net, generate_recognition_data, maxEpochs=maxEpochs, epochsPerBatch=epochsPerBatch,
                               T=T, d=d, R=R, P=P, interleave=True)        
    elif trainingMethod == 'std':
        trainData = generate_recognition_data(T=T, d=d, R=R, P=P, interleave=True)
        net.fit(trainData, epochs=maxEpochs)
    else: 
        raise ValueError('Invalid training method')
#%%
joblib.dump(net, '{}_recog_R={}_{}.pkl'.format(net.name,R,trainingMethod))

#%%
testData = generate_recognition_data(T=10000, d=d, R=R, P=P, interleave=True, astensor=True)
plot_train_perf(net, recog_chance(testData),
                title='R={} ({:.4}%)'.format(R, net.accuracy(testData)*100))
plt.savefig('{}_recog_R={}_{}.png'.format(net.name, R, trainingMethod)) 

#%%
