import matplotlib.pyplot as plt
import joblib
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch

from data import generate_aug_recog_data_batch, generate_recog_data_batch, recog_chance
from plotting import plot_train_perf
from networks import HebbAugRecog

#%% Compare batchsizes 
#T=500
#B=1
#gpu=False
#
#for R in [2,5,10]:
#    fig, ax = plt.subplots(2,1)
#    for T in [500,2000]:
#        for B in [1,64]:
#            for gpu in [False]:
#    
#                fname = 'HebbAugRecog_inf1_R={}_T={}_B={}_{}.pkl'.format(R,T,B,'gpu_ascpu' if gpu else 'cpu')
#                net = joblib.load(fname)            
#                
#                
#                epochs = net.hist['epoch']
#                histlen = len(net.hist['train_loss']) #if recorded only every Nth epoch
#                N = epochs/(histlen-1)
##                epochs = np.arange(0, epochs+1, N)
#                epochs = np.linspace(0, net.hist['time']/60., histlen)
##                epochs = np.arange(5000)
#                
#                if T==500 and B==1:
#                    c='Blue'
#                elif T==500 and B==64:
#                    c='Green'            
#                elif T==2000 and B==1:
#                    c='Red'            
#                elif T==2000 and B==64:
#                    c='Purple'
#                 
#                acc = net.hist['test_acc'][:len(epochs)]
#                loss = net.hist['test_loss'][:len(epochs)]
#                ax[0].plot(epochs, acc, ':' if gpu else '-', color=c, linewidth=1, label='T={} B={} {}'.format(T,B,'gpu' if gpu else 'cpu'))            
#                ax[1].plot(epochs, loss, ':' if gpu else '-', color=c, linewidth=1, label='T={} B={} {}'.format(T,B,'gpu' if gpu else 'cpu'))
#    
#    plt.legend()
#    ax[0].set_title('Timesteps vs Batchsize \n (aug recog R={})'.format(R))
#    ax[0].set_ylabel('Accuracy')
#    ax[1].set_ylabel('Loss')
#    ax[1].set_xlabel('Epochs')
#    fig.tight_layout()
#%%
T=500
B=1
d=25
k=1
R=1
#%%
#testData = batch(generate_aug_recog_data, batchsize=B, T=2000,d=d,k=k,R=R,P=0.5,interleave=True)
#chancenet = HebbAugRecog([d+k, 50, 1+k])
#chancenet.fit(testData, epochs=10000)

#%%
for R in [1]:#[2,5,10,20]:
    for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fname = 'HebbAugRecog_R={}_c={}.pkl'.format(R, c)
        net = joblib.load(fname)  
        
        print '\n\nc={}\n{:.2f}, {:.2f}, \n{:.2f}, {:.2f}'.format(c, net.lam1, net.eta1, net.lam2, net.eta2)
        
        #%%
        testData = generate_aug_recog_data_batch(T=5000, batchSize=B, d=d, k=k, R=R, interleave=True)
        testAcc = net.accuracy(testData.tensors).item()
        testDataRecog = TensorDataset(testData.tensors[0], testData.tensors[1][:,:,0])
        chance = recog_chance(testDataRecog)
        fig, ax = plot_train_perf(net, chance, testAcc=testAcc, title='R={}, c={}'.format(R,c), color='Blue')
        
        iters = net.hist['iter']
        histlen = len(net.hist['train_loss']) #if recorded only every Nth epoch
        N = iters/(histlen-1)
        iters = np.arange(0, iters+1, N)
        
        ax[0].lines[0].set_label('Total')
        ax[0].plot(iters, net.debug_log['recog_acc'], color='Green', label='Recog')
        ax[0].plot(iters, net.debug_log['value_acc'], color='Purple', label='Value')
        ax[0].set_ylim([0.37, 1.03])

        
        ax[1].lines[0].set_label('Total')
        ax[1].plot(iters[:-1], net.debug_log['recog_loss'], color='Green', label='Recog')
        ax[1].plot(iters[:-1], net.debug_log['value_loss'], color='Purple', label='Value')
        ax[1].set_ylim((0,1.2))
        
#        ax = plt.gcf().get_axes()
#        xlim = (-1000, 50000)
#        ax[0].set_xlim(xlim)
#        ax[1].set_xlim(xlim)
        
        plt.legend()
        
#        plt.savefig(fname[:-3]+'png')
#%%
    
#testData = generate_aug_recog_data_batch(T=10, batchSize=B, d=d, k=1, R=R, interleave=True)
#out = net.evaluate(testData.tensors).detach()
#out[:,:,0] = out[:,:,0].round()
#out[:,:,1] = out[:,:,1].sign()
#
#torch.cat((testData.tensors[0].squeeze(), testData.tensors[1].squeeze(), out.squeeze()), dim=1).t()
    
    
#%%
import csv
def parse_csv(fname):
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile,)
        x = []
        y = []
        reader.next() #skip first line
        for r in reader:
            x.append(int(r[1])) #step
            y.append(float(r[2])) #value
    return x,y

       
    
    
for R in [1]:
    for c in [0.8, 0.9]:
        fig, axs = plt.subplots(2,1)        
        for m in ['loss', 'acc']:
            if m == 'loss':
                tList = ['recog', 'value', 'total']
                ax = axs[1]
            elif m == 'acc':
                tList = ['recog', 'value', 'average']
                ax = axs[0]
            for t in tList:            
                fname = 'run-HebbAugRecog_R={R}_c={c}_train_{m}_breakdown_{t}-tag-train_{m}_breakdown.csv'.format(R=R, c=c, m=m, t=t)
                steps, values = parse_csv(fname)
                if m=='loss':
                    if t=='recog':
                        values = [c*val for val in values]
                    elif t=='value':
                        values = [(1-c)*val for val in values]
                if t == 'recog':
                    color = 'Green'
                elif t == 'value':
                    color = 'Purple'
                elif t=='total' or t=='average':
                    color = 'Blue'                
                ax.plot(steps, values, color=color, label=t)
        fig.set_size_inches(3,5)
        axs[0].set_ylabel('Accuracy')
        axs[0].set_ylim([0.37, 1.03])
    
        axs[1].set_ylabel('Loss')
        axs[0].legend()
        axs[1].legend()
        axs[1].set_ylim((0,1.2))
    
        net = HebbAugRecog([d+k,50,1+k], c=c)
        net.load('HebbAugRecog_R={}_c={}.pkl'.format(R,c))
        net.eval()

        testData = generate_aug_recog_data_batch(T=5000, batchSize=B, d=d, k=k, R=R, interleave=True)
        testAcc = net.accuracy(testData.tensors).item()
        axs[0].plot(steps, testAcc*np.ones(len(steps)), 'r--')
        axs[0].set_title('R={}, c={}\n$\lambda_1$={:.2f}, $\eta_1$={:.2f},\n$\lambda_2$={:.2f}, $\eta_2$={:.2f}'.format(
                R,c, net.lam1, net.eta1, net.lam2, net.eta2))    
        fig.tight_layout()
        
        plt.savefig('HebbAugRecog_R={}_c={}.png'.format(R,c))



