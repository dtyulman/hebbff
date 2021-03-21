import sys;
import numpy as np
import joblib

from dt_utils import Timer

from data_utils import generate_recognition_data
from plastic import PlasticNet


d = 50           #length of input vector
R = int(sys.argv[1])  #repeat interval
P = .5           #probability of repeat

np.random.seed(12345) #for reproducibility

bootstrap = int(sys.argv[2]) 
net = joblib.load('PlasticNet_R={}_bootstrap=7,30.pkl'.format(bootstrap))
initW = [net.w1.detach().numpy(),net.w2.detach().numpy()]
initB = [net.b1.detach().numpy(),net.b2.detach().numpy()]
eta = net.eta.item()
lam = net.lam.item()

filename = 'PlasticNet_R={}_bootstrap=7,30,{}_T=5000.pkl'.format(R, bootstrap)

#%% Infinite data training
net = PlasticNet(initW, initB, eta=eta, lam=lam)   
with Timer(filename):
    for epoch in range(200000):
        trainData = generate_recognition_data(T=5000, d=d, R=R, P=P, interleave=True, astensor=True)
        net.fit(trainData, epochs=1)
        if sum(net.hist['train_acc'][-5:]) >= 4.99:
            break #early stop
 
#%%       
joblib.dump(net, filename)



