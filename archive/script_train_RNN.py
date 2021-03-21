from data import generate_recog_data_batch, generate_recog_data
from networks import VanillaRNN
from plotting import plot_recog_generalization, plot_train_perf

#%%
d=25
N=625
R = [3,6]
gen_data = lambda: generate_recog_data_batch(T=500, d=d, R=R, P=0.5, batchSize=1, multiRep=False)

dims = [d,N,1]
net = VanillaRNN(dims)
#%%
fname = 'VanillaRNN{}_inf{}_noMR.pkl'.format(dims, R).replace(' ', '')
net.fit('infinite', gen_data, iters=float('inf'), batchSize=None, earlyStop=True, folder='RNN', filename=fname)

#%%
gen_data_R = lambda R: generate_recog_data(T=max(500, R*20), d=d, R=R, P=0.5, multiRep=False)
net.w1 = net.Wx
ax, testR, testAcc = plot_recog_generalization(net, gen_data_R, upToR=10, label='RNN, $R_{{train}}$={}'.format(R).replace(' ', ''))
ax.set_xscale('linear')

#%%
trainData = gen_data()
validBatch = gen_data()[:,0,:]
fname = 'VanillaRNN{}_dat{}_noMR.pkl'.format(dims, R).replace(' ', '')
net.fit('dataset', trainData, validBatch=validBatch, epochs=float('inf'), batchSize=None, earlyStop=True, folder='RNN', filename=fname)

#%%
fig,ax = plot_train_perf(net, 0.66, title='RNN, single dataset')
