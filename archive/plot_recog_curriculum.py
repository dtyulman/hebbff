import matplotlib.pyplot as plt
import joblib
import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn

from data import  generate_recog_data, recog_chance
from plotting import plot_train_perf, plot_W, plot_B, plot_recog_generalization
from networks import HebbNet
from net_utils import binary_classifier_accuracy, load_from_file

#%%
T=500
B=1
d=25
#R=1
P=0.5

#%%
Nh=25
ax = None


#    net = joblib.load(fname)  
    net = load_from_file(fname, dims=[d,Nh,1])
    
#    plot_W([net.w2.detach()])

#    plot_W([net.w1.detach(), net.w2.detach()])
#    fig, ax = plot_B([net.b1.detach(), net.b2.detach()])
#    fig.set_size_inches([1.75, 4.8])
#    plt.savefig(fname[:-4] + '_Bplot.png')
    
    gen_data = lambda R: generate_recog_data(T=2000, d=d, R=R, P=0.5, interleave=True, multiRep=True)
    ax, _, _ = plot_recog_generalization(net, gen_data, ax=ax, label=w1init)


#%%
d=25  
gen_data = lambda R: generate_recog_data(T=R*50, d=d, R=R, P=0.5, interleave=True, multiRep=True)
    
#_,Rax = plt.subplots()
Gax = None
for fname, label in [
#                     ('HebbNet_Nh=25_w1=randn-train_w2=randn-train.pkl',       'R/tr, R/tr'),
#                     ('HebbNet_Nh=25_w1=posneg_eye-train_w2=randn-train.pkl', '+-/tr, R/tr'), 
#                     ('HebbNet_Nh=25_w1=posneg_eye-train_w2=negs-train.pkl',  '+-/tr, N/tr'),
#              
#                     ('HebbNet_Nh=25_w1=posneg_eye-fix_w2=randn-train.pkl', '+-/f, R/tr'),
#                     ('HebbNet_Nh=25_w1=eye-fix_w2=randn-train.pkl',         'I/f, R/tr'),
##                     ('HebbNet_Nh=25_w1=neg_eye-fix_w2=randn-train.pkl',    '-I/f, R/tr'), #DOESN'T LOAD...
#                     ('HebbNet_Nh=25_w1=randn-fix_w2=randn-train.pkl',       'R/f, R/tr'),
#                     ('HebbNet_Nh=25_w1=shuffle-fix_w2=randn-train.pkl',     'S/f, R/tr'),
                      
#                     ('HebbNet_Nh=25_w1=posneg_eye-train_w2=negs-fix.pkl',' +-/tr, N/f'),
#                     ('HebbNet_Nh=25_w1=randn-train_w2=negs-fix.pkl',       'R/tr, N/f'),
 
                     ('HebbNet_Nh=25_w1=posneg_eye-train_w2=negs-train.pkl',                '+-diag, full' ),
                     ('HebbNet_Nh=25_w1=posneg_eye-train_b1=uniform_w2=uniform-train.pkl',  '+-diag, unif'),
                     ('HebbNet_Nh=25_w1=randn-train_b1=uniform_w2=uniform-train.pkl',       'randn, unif'),
                     
#                     ('w1=unif_shuf_randn-fix_w2=None-train.pkl', 'shuf&train randn, unif'),
#                     ('w1=unif_shuf_diags-fix_w2=None-train.pkl', 'shuf&train +-diag, unif'),
#                     ('w1=shuffle-fix_w2=negs-train.pkl', 'shuf&train_negs baseline'),
#                     ('w1=shuffle-fix_w2=randn-train.pkl', 'shuf&train_randn baseline'),
                      ]:      
#    if fname.endswith('w2=negs-fix.pkl'):
#        fmt = '--'
#    elif fname.endswith('-fix_w2=randn-train.pkl'):
#        fmt = ':'
#    else:
#        fmt = '-'
    
    net = load_from_file(fname, dims=[d, 25, 1])

    
#    label='{} ($\lambda=${:.2f} $\eta=${:.2f})'.format(label, net.lam, net.eta)
            
#    plot_W([net.w1.detach()])
#    plot_B([net.b1.detach(), net.b2.detach()])

#    Rax.step(net.hist['increment_R'], range(len(net.hist['increment_R'])), linestyle=fmt, label=label)

    
#    if label.find('randn,')>=0:
#        fmt = 'r'    
#    elif label.find('+-diag')>=0:
#        fmt = 'g'
#    elif label.find('baseline')>=0:
#        fmt = 'b'
#
#        
#    if label.find('shuf&train')>=0:       
#        if label.find('_negs')>0:
#            Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+'--', label=label)
#        else:
#            Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+'-.', label=label)
#        continue
#
    
#    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+'-', label=label)
    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label=label)

    
#    if label.find('shuf')<0:
    #Plot various permutations
#        w = net.w1.detach()
#        
#        net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 
#        Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+':', label='shuf ' + label)
#
#    net.w1.data = w.t()       
#    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+'-.', label='transp ' + label)
#
##    perm = torch.randperm(w.shape[0])
##    net.w1.data = w[perm,:]  
##    net.b1.data = net.b1[perm]     
##    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt, label='rows&b1 ' + label)
#    
##    net.w1.data = w[:,torch.randperm(w.shape[1])]       
##    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt, label='cols ' + label)
#    
#    net.w1.data = w[torch.randperm(w.shape[0]),:][:,torch.randperm(w.shape[1])]       
#    Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, fmt=fmt+'--', label='rowcol ' + label)
#        
#    plot_W([net.w1.detach()])
#    plt.title(label)
    
#    print label
#    print 'row sums ', net.w1.sum(0).sort()[0]
#    print 'col sums ', net.w1.sum(1).sort()[0]
#    
#    w = net.w1.detach()
#    fig, ax = plt.subplots(2,2)
#    wIm = ax[0,0].matshow(w, cmap='RdBu_r')
#    ax[0,0].set_title(label)
#    fig.colorbar(wIm, label='w', orientation='horizontal')
#    rowSum = ax[0,1].matshow(w.sum(1).unsqueeze(1), cmap='RdBu_r')
#    fig.colorbar(rowSum, label='r', orientation='horizontal')
#    colSum = ax[1,0].matshow(w.sum(0).unsqueeze(0), cmap='RdBu_r')
#    fig.colorbar(colSum, label='c', orientation='horizontal')
#    for a in ax.flat:
#        a.axis('off')
        


##Rax.legend()
##Rax.set_xlabel('Iter')
##Rax.set_ylabel('R')








#%%
for Nh in [2]:#[1, 10, 25, 100, 200]:
    fname = 'HebbNet_R=curriculum_Nh={}.pkl'.format(Nh)
#    fname = 'HebbNet_curriculum=add_Nh={}.pkl'.format(Nh)        
#    net = joblib.load(fname)  
     
    net = HebbNet([d,Nh,1])
    net.load(fname)
     
    #testData = generate_recog_data_batch(T=5000, batchSize=B, d=d, k=1, R=R, interleave=True)
    #testAcc = net.accuracy(testData.tensors).item()
    #chance = recog_chance(testData)
    
    iters = net.hist['iter']
    histlen = len(net.hist['train_loss']) #if recorded only every Nth epoch
    N = iters/(histlen-1)
    iters = np.arange(0, iters+1, N)
    
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('$Nh$={}, $R_{{max}}$={}\n$\lambda$={:.2f}, $\eta$={:.2f}'.format(Nh, len(net.hist['increment_R']), net.lam, net.eta))
    ax[0].plot(iters, net.hist['train_acc'], linewidth=0.25)
    
    ax[0].plot(net.hist['increment_R'], np.ones(len(net.hist['increment_R'])),'*')
    #ax[0].plot(iters, net.hist['valid_acc'], color='Purple', label='valid')
    ax[0].set_ylabel('accuracy')
    
    ax[1].plot(iters, net.hist['train_loss'], linewidth=0.25)
    #ax[1].plot(iters, net.hist['valid_loss'], color='Purple', label='valid')
    ax[1].set_ylabel('loss')
    ax[1].plot(net.hist['increment_R'], np.ones(len(net.hist['increment_R'])),'*')
 
#    xlim = (-100, 50000)
#    ax[0].set_xlim(xlim)
#    ax[1].set_xlim(xlim)
       
#    plt.savefig(fname[:-3]+'png')

#%%  Generalization
#testR = {}
#testAcc = {}
    
gen = joblib.load('generalization.pkl')
testR = gen['testR']
testAcc = gen['testAcc']
    
for Nh in [25, 100]: 
#    fname = 'HebbNet_R=curriculum_Nh={}.pkl'.format(Nh)    
#    net = joblib.load(fname)  
#    
#    net.acc_fn = binary_classifier_accuracy
#        
#    testR[str(Nh)] = []
#    testAcc[str(Nh)] = [] 
#    print '\nNh =', Nh
#    acc= float('inf')
#    chance = 0
#    R=1
#    while acc > chance:
#        testData = generate_recog_data(T=R*50, d=d, R=R, P=P, interleave=True)
#        acc = net.accuracy(testData.tensors).item()
#        chance = recog_chance(testData)
#        testAcc[str(Nh)].append( acc )
#        testR[str(Nh)].append(R)
#        print 'R={}, acc={:.3f}, ({})'.format(R, acc, chance)
#        R = int(np.ceil(R*1.1))

    plt.semilogx(testR[str(Nh)], testAcc[str(Nh)], '--' if Nh==50 or Nh==1 else '-',  label='Nh={}'.format(Nh))

plt.legend()
plt.xlabel('R_test')
plt.ylabel('Generalization acc')
plt.gcf().set_size_inches(8,3)
plt.gca().set_ylim((0.5,1.01))    
plt.tight_layout()

#%% Create Nh=100 out of four Nh=25 networks
Nh=25
Nh_new = 100
if float(Nh_new)/Nh % 1 != 0: raise ValueError()
net25 = joblib.load('HebbNet_R=curriculum_Nh={}.pkl'.format(Nh))
net = joblib.load('HebbNet_R=curriculum_Nh={}.pkl'.format(Nh_new))
with torch.no_grad(): 
    net.w1.data = torch.cat([net25.w1 for i in range(Nh_new/Nh)])  
    net.b1.data = torch.cat([net25.b1 for i in range(Nh_new/Nh)])  
#    net.w2.data = torch.cat([net25.w2 for i in range(Nh_new/Nh)], dim=1) 
#    net.b2.data = (Nh_new/Nh)*net25.b2.data
net.acc_fn = binary_classifier_accuracy
net.plastic = True 

#%% permute the rows of W1 matrix
Nh=25
Nh_new=25
net = joblib.load('HebbNet_R=curriculum_Nh={}.pkl'.format(Nh))
net.plastic = True 
net.acc_fn = binary_classifier_accuracy

perm = torch.randperm(Nh)
#net.w1.data = net.w1.data[perm] 
net.b1.data = net.b1.data[perm] 
#net.w2.data = net.w2.t().data[perm].t() #sanity check. works same as original if permute w2 the same way


#%%
k = '{} ({})'.format(Nh_new, Nh)    
testR[k] = []
testAcc[k] = [] 
print '\nNh =', k
acc = float('inf')
chance = 0
R=1
while acc > chance:
    testData = generate_recog_data(T=R*50, d=d, R=R, P=P, interleave=True)
    acc = net.accuracy(testData.tensors).item()
    chance = recog_chance(testData)
    testAcc[k].append( acc )
    testR[k].append(R)
    print 'R={}, acc={:.3f}, ({})'.format(R, acc, chance)
    R = int(np.ceil(R*1.1))

plt.semilogx(testR[k], testAcc[k], '--' if Nh==50 or Nh==1 else '-',  label='Nh={}'.format(k))
plt.legend()






