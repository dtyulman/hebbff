import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
from matplotlib import patches
import copy, math, os, types
from data import generate_recog_data, generate_recog_data_batch
import torch
from torch.utils.data import TensorDataset




def plot_w1_row_col_sums(w, label='', save=False):
    fig, ax = plt.subplots()
    v = w.abs().max()
    im = ax.matshow(w, cmap='RdBu_r', vmin=-v, vmax=v)    
    plt.tight_layout()
    fig.subplots_adjust(left=0.2)
    cax = fig.add_axes([0.1, 0.1, 0.03, 0.8])
    ticks = [math.ceil(-v), math.floor(v)] if v>=1 else [-v,v]
    fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    ax.axis('off')
    ax.set_title(label)
    fig.set_size_inches(3.5,3.5)
    if save: fig.savefig(label + '_rowcolsum_W1')
    
    fig, ax = plt.subplots()
    v = w.sum(1).abs().max()
    im = ax.matshow(w.sum(1).unsqueeze(1), cmap='RdBu_r', vmin=-v, vmax=v)
    plt.tight_layout()
    fig.subplots_adjust(right=0.2)
    cax = fig.add_axes([0.5, 0.1, 0.03, 0.8])
    ticks = [math.ceil(-v), math.floor(v)] if v>=1 else [-v,v]
    fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    ax.axis('off')
    fig.set_size_inches(1.7,3.8)
    if save: fig.savefig(label + '_rowcolsum_W1_rowsum')
    
    fig, ax = plt.subplots()
    v = w.sum(0).abs().max()
    im = ax.matshow(w.sum(0).unsqueeze(0), cmap='RdBu_r', vmin=-v, vmax=v)
    ticks = [math.ceil(-v), math.floor(v)] if v>=1 else [-v,v]
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.03])
    fig.colorbar(im, ticks=ticks, cax=cax, orientation='horizontal')
    ax.axis('off')
    fig.set_size_inches(3.5,1.5)
    if save: fig.savefig(label + '_rowcolsum_W1_colsum')


def plot_per_unit_hist(net, R, title=''):
    Nh,d = net.w1.shape  
    
    #get dims of grid
    a = int(np.ceil(np.sqrt(Nh)))
    b = int(np.round(np.sqrt(Nh)))     
    r = min(a,b)
    c = max(a,b)    
    #gs = GridSpec(r,c)
        
    T = 10000
    data = generate_recog_data(T=T, d=d, R=R, interleave=True, multiRep=True)
        
    a1, h, Wxb, Ax, a2, out, acc, data = evaluate_debug(net, data) #TODO: use net.evaluate_debug() instead
     
    figList = []
    for X, xlabel in zip([a1, h], ['$[(W_1+A_t)x_t+b_1]_i$', '$h_i(t)$']):
        fig, axs = plt.subplots(r,c, sharex=True)
        for Xi, ax in zip(X.t(), axs.flat):
            plot_output_distr(Xi.unsqueeze(1), data.tensors[1], mean=True, xlabel='', ax=ax)
            
            ax.legend().remove() 
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)                   
            
            if xlabel != '$h_i(t)$':           
                ax2 = ax.twinx()
                x = torch.arange(a1.min().item(), a1.max().item())
                ax2.plot(x, torch.sigmoid(x), lw=0.5, color='k')
                ax2.set_ylim((0,1))
                ax2.tick_params(right=False, labelright=False)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)   
            
        axs[0,c-1].legend()
        fig.set_size_inches(10,5)        
        axs[0, c/2].set_title('$R_{{test}}$={}, acc={:.2f}: {}'.format(R, acc, title))
        axs[-1, 0].set_xlabel(xlabel)
        [ax.axis('off') for ax in axs.flat[Nh:]] 

        figList.append(fig)
    return figList


def plot_multi_hist(net, gen_data, Rmp=None, Rmc=None, title=''):    
    fig, ax = plt.subplots(3,4, sharex='col')#, sharey='col')
         
    gen_dataR = lambda R: gen_data(R, T=max(1000,20*R)) 
    Rmp = get_R_max(net, gen_dataR, chance=0.98) if Rmp is None else Rmp
    Rmc = get_R_max(net, gen_dataR, chance=0.55) if Rmc is None else Rmc
    Rav = (Rmp+Rmc)/2
    
    for axRow,R in enumerate([Rmp, Rav, Rmc]): 
        T = 2000
        db = net.evaluate_debug(gen_data(R,T).tensors)
        globals().update(db) #TODO: this is a hack to get the vars Wxb, Ax, data, etc in the namespace

        axCol=0
#        plot_output_distr(Wxb.mean(1).unsqueeze(1), data.tensors[1], mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1     
#        plot_output_distr(Ax.mean(1).unsqueeze(1), data.tensors[1], mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1    
#        plot_output_distr(a1.mean(1).unsqueeze(1), data.tensors[1], mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1          
#        plot_output_distr(h.mean(1).unsqueeze(1), data.tensors[1], mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1  
        
        plot_output_distr(Wxb.flatten(), data.tensors[1].expand(-1, Wxb.shape[1]).flatten(), mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1          
        plot_output_distr(Ax.flatten(), data.tensors[1].expand(-1, Ax.shape[1]).flatten(), mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1    
        plot_output_distr(a1.flatten(), data.tensors[1].expand(-1, a1.shape[1]).flatten(), mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1          
#        plot_output_distr(h.flatten(), data.tensors[1].expand(-1, h.shape[1]).flatten(), mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1                

        plot_output_distr(a2, data.tensors[1], mean=True, xlabel='', ax=ax[axRow,axCol]); axCol+=1          
#        plot_output_distr(out, data.tensors[1], xlabel='', ax=ax[axRow,axCol]); axCol+=1  
        
        ax[axRow,0].set_ylabel('R={} \n acc={:.2f}'.format(R, acc))       

    axCol=0
    axDrawSigmoid = []
#    ax[-1,axCol].set_xlabel('$<W_1x_t+b_1>_i$'); axCol+=1
#    ax[-1,axCol].set_xlabel('$<A_t x_t>_i$'); axCol+=1
#    ax[-1,axCol].set_xlabel('$<(W_1+A_t)x_t+b_1>_i$'); axCol+=1
#    ax[-1,axCol].set_xlabel('$<h(t)>_i$'); axCol+=1
    
    ax[-1,axCol].set_xlabel('$W_1x_t+b_1$'); axCol+=1
    ax[-1,axCol].set_xlabel('$A_t x_t$'); axCol+=1
    ax[-1,axCol].set_xlabel('$(W_1+A_t)x_t+b_1$'); axDrawSigmoid.append(axCol); axCol+=1
#    ax[-1,axCol].set_xlabel('$h_i(t)$'); axCol+=1

    ax[-1,axCol].set_xlabel('$W_2h_t+b_2$'); axDrawSigmoid.append(axCol); axCol+=1

#    ax[-1,axCol].set_xlabel('$\hat{y}(t)$'); axCol+=1    
            
    
    
    ax_all = list(ax.flat)
    for ax1 in ax[:,axDrawSigmoid].flatten():
        ax2 = ax1.twinx()
        x = torch.arange(*ax2.get_xlim())
        ax2.plot(x, torch.sigmoid(x), lw=0.5, color='k')
        ax2.set_ylim((0,1))
        ax2.tick_params(right=False, labelright=False)
        ax_all.append(ax2)
    
    for a in ax_all:
        a.spines["top"].set_visible(False)
#        a.spines["bottom"].set_visible(False)
        a.spines["right"].set_visible(False)   
        try: a.legend().remove() 
        except: pass
#    ax[-1,-1].legend(loc='upper center')
    
    if title:
        ax[0,ax.shape[1]/2].set_title(title)
        
    fig.set_size_inches(1.6*ax.shape[1],3.9)

    return ax, Rmp, Rmc
    

def plot_R_curriculum(iters, Rs, label='', ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.step(iters, Rs, label=label, where='post', **kwargs)
    ax.set_xlabel('Iter')
    ax.set_ylabel('R')
    return ax

 
def plot_ROC(out, y, incr=0.001, label='', ax=None):
    thres = np.arange(0, 1+incr, incr)[::-1]
    falsePosRate = np.empty(thres.shape)
    truePosRate = np.empty(thres.shape)
    totPos = y.sum().item()
    totNeg = len(y)-totPos
    for i,th in enumerate(thres):
        posIdx = out>th
        falsePos = (1-y)[posIdx].sum().item()        
        truePos = y[posIdx].sum().item()
        falsePosRate[i] = falsePos/totNeg 
        truePosRate[i] = truePos/totPos 
        
    fprUniqueIdx = np.concatenate(([True], np.diff(falsePosRate)!=0))    
    AUC = scipy.integrate.trapz(y=truePosRate[fprUniqueIdx], x=falsePosRate[fprUniqueIdx])
       
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_xlim((-0.01, 1.01))
        ax.set_ylim((-0.01, 1.01))  
        fig.tight_layout()

    plt.plot([0,1], [0,1], 'k--')
    line = plt.plot(falsePosRate, truePosRate, label='{} AUC={:.4f}'.format(label, AUC))
    plt.legend()
      
    for th in [0, 0.25, 0.5, 0.75, 1]:
        idx = np.argmin(np.abs(thres-th))
        plt.scatter(falsePosRate[idx],  truePosRate[idx], color=line[0].get_color())
        plt.text(falsePosRate[idx],  truePosRate[idx], str(th))
        
    return ax, falsePosRate, truePosRate, AUC

  
def plot_output_distr(out, y, mean=False, xlabel='x', ax=None):
    try: out = out.detach()
    except: pass

    if ax is None:
        fig, ax = plt.subplots()
            
    bins = 50
    
    values,bins,_ = ax.hist(out[y==0], bins=bins, density=True, histtype='step', align='mid', color='red',   label='p($\cdot$|y=0)')    
    if np.abs(bins-bins.mean()).sum() < 0.01:
        ax.set_xlim(bins.mean()-0.01, bins.mean()+0.01)
    values,bins,_ = ax.hist(out[y==1], bins=bins, density=True, histtype='step', align='mid', color='green', label='p($\cdot$|y=1)')
    if np.abs(bins-bins.mean()).sum() < 0.01:
        ax.set_xlim(bins.mean()-0.01, bins.mean()+0.01)
    
    if mean:
        ax.axvline(out[y==0].mean(), linestyle='--', linewidth=0.5, color='red', )
        ax.axvline(out[y==1].mean(), linestyle='--', linewidth=0.5, color='green')

    ax.set_xlabel(xlabel)
    ax.legend()

    return ax 

        
def plot_W(W, B=None):
    """
    Args:
        W - list of weight matrices
    """
    if B:
        W = [np.hstack((w,b.reshape(len(b),1))) for w,b in zip(W,B)]
    
    #get largest absolute weight to set colorbar range across all subplots
    v = maxabs(W)
    
    #plot
    fig, axes = plt.subplots(1,len(W),squeeze=False)
    for i,w,ax in zip(range(len(W)), W, axes.flat): 
        try:
            w = w.detach()
        except:
            pass
                
        r,c = w.shape
        if 2*r < c: #plot wide weight matrices vertically for easier viz
            w = w.t()
            T = '^T'
            line = patches.ConnectionPatch((-0.5, w.shape[0]-1.5), (w.shape[1]-0.5, w.shape[0]-1.5), 'data', arrowstyle="-")
        else:
            T = ''
            line = patches.ConnectionPatch((w.shape[1]-1.5, -0.5), (w.shape[1]-1.5, w.shape[0]-0.5), 'data', arrowstyle="-")
        if B: ax.add_artist(line)
         
        im = ax.matshow(w, cmap='RdBu_r', vmin=-v, vmax=v)
        ax.axis('off')
#        ax.set_xticks([])
#        ax.set_yticks([])
        ax.set_title('${}_{}{}$ ({}x{})'.format('[W|b]' if B else 'W',i+1,T,r,c))
        
    #add colorbar    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
    ticks = [math.ceil(-v), math.floor(v)] if v>=1 else [-v,v]
    fig.colorbar(im, ticks=ticks, cax=cax, orientation='horizontal')
       
    return fig, axes


def plot_B(B):
    fig, axes = plot_W( [b.reshape(len(b),1) for b in B] )
    [ax.set_title(ax.get_title().replace('W', 'b')) for ax in axes.flat]
    return fig, axes
    

def plot_h_seq(net, data, opt):
    '''Plot sequence of hidden activities during evaluation of data
    opt can be 'x' (plot input layer activities), 
               'a' (plot hidden layer activations i.e. before nonlinearity)
               'h' (plot hidden layer outputs i.e. after nonlinearity)
    '''
    #TODO: this needs cleanup   
    net.init_A()    
    h_t = []    
    for i,(x,y) in enumerate(data):
        a,h = net.feedforward(x)
        if opt == 'x':
            result = h[0]
            cmap = 'binary'
        elif opt == 'h':
            result = h[1]
            cmap = 'Reds'
        elif opt == 'a':
            result = a[0]
            cmap = 'Greens'
        else:
            raise ValueError('Invalid opt')
        h_t.append( result.reshape(-1,1) )    
    hv = maxabs(h_t)    
    
    fig, axs = plt.subplots(1,len(data))
    axs = np.expand_dims(axs,0)
    for i in range(len(data)):
        imH = axs[0,i].matshow(h_t[i], cmap=cmap, vmin=0, vmax=hv)
        axs[0,i].set_title('y={}'.format(data[i][1]))
    
    [ax.axis('off') for ax in axs.flatten()]
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.93, .04+.33, 0.01, 0.25])  
    fig.colorbar(imH, cax=cax)
  
    
def plot_W_seq(net, data):
    '''Plot sequence of W and A matrices during evaluation of data'''
    #TODO: this needs cleanup

    net.init_A()
    WA_t = [net.W[0].T]
    A_t = []    
    for i,(x,y) in enumerate(data):
        a,h = net.feedforward(x)
        A_t.append( copy.deepcopy(net.A[0].T) )
        WA_t.append( net.W[0].T+net.A[0].T )    
    Av = maxabs(A_t)
    WAv = maxabs(WA_t)    
    
    fig, axs = plt.subplots(2,len(data)+1)
    for i in range(len(data)):     
        imA = axs[0,i+1].matshow(A_t[i], cmap='RdBu_r', vmin=-Av, vmax=Av)
        axs[0,i+1].set_title('y={}'.format(data[i][1]))
        imWA = axs[1,i+1].matshow(WA_t[i+1], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 
    imWA = axs[1,0].matshow(WA_t[0], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 
    [ax.axis('off') for ax in axs.flatten()]

    #axs[1,0].set_title('$W_{1,init}$')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.93, .05+.5, 0.01, 0.4])  
    fig.colorbar(imA, cax=cax)
    cax = fig.add_axes([0.93, .05, 0.01, 0.4])  
    fig.colorbar(imWA, cax=cax)  


def plot_train_perf(net, chance, testAcc=None, title=None, color=None, ax=None):
    '''Plot performance of the network during training'''
    epochs = net.hist['iter']
    histlen = len(net.hist['train_loss']) #if recorded only every Nth epoch
    N = epochs/(histlen-1)
    epochs = np.arange(0, epochs+1, N)
    
    if ax is None:
        fig, ax = plt.subplots(2,1)
    else:
        fig = ax[0].get_figure()
    
    ax[0].plot(epochs, net.hist['train_acc'], color=color, label='train')
    ax[0].set_title(net.name if title is None else title)
    ax[0].set_ylabel('Accuracy')
    if 'valid_acc' in net.hist.keys():
        ax[0].plot(epochs, net.hist['valid_acc'], label='valid')
        ax[0].legend()
    if testAcc is not None:
        ax[0].plot(epochs, testAcc*np.ones(len(epochs)), 'r--', label='test')
    ax[0].plot(epochs, chance*np.ones(len(epochs)), 'k--', label='chance')
  
    ax[1].plot(epochs, net.hist['train_loss'], color=color, label='train')
    ax[1].set_ylabel('Loss')
#    ax[1].set_xlabel('Epochs') 
    if 'valid_loss' in net.hist.keys():
        ax[1].plot(epochs, net.hist['valid_loss'], label='valid')
 
    fig.set_size_inches(3,5)
    fig.tight_layout()
    
    return fig, ax 


def plot_train_perf_from_log(fname, chance, title, color='Blue'):
    #extract data from log file
    data = {'Epoch':[],
            'train_loss':[],
            'train_acc':[]}
    with open(fname) as f:
        for line in f:
            for entry in data.keys():
                beg = line.find(entry) 
                if beg < 0:
                    break
                beg += len(entry) + 1
                end = line.find(' ', beg)
                data[entry].append( float(line[beg:end]) )
    
    #create fake `net` object with `hist` and `name` fields 
    data['epoch'] = data['Epoch'][-1]   
    class Struct(): pass #there's gotta be a better way    
    net = Struct()
    net.hist = data
    plot_train_perf(net, chance, title=title, color=color)          
    

def plot_train_params(net, ax=None):
    '''Plot parameters of the network during training'''
    epochs = net.hist['epoch']
    histlen = len(net.hist['train_loss']) #if recorded only every Nth epoch
    N = epochs/(histlen-1)
    epochs = np.arange(0, epochs+1, N)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    lam = []
    eta = []
    for p in net.hist['params']:
        lam.append( p['lam'] )
        eta.append( p['eta'] )
#    ax.set_title(net.name)
    ax.plot(epochs, lam, label='$\lambda$')
    ax.plot(epochs, eta, label='$\eta$')
    ax.set_xlabel('Epochs') 
    ax.legend()
    
    fig.set_size_inches(3,5)
    fig.tight_layout()


def plot_recog_generalization(net, gen_recog_data, upToR=float('-inf'), stopAtR=float('inf'), testR=[], testAcc=[], ax=None, label=None, burnin=True, **plotKwargs):    
    if len(testR) == len(testAcc) == 0:    
        testR,testAcc,_,_ = get_recog_positive_rates(net, gen_recog_data, upToR=upToR, stopAtR=stopAtR, burnin=burnin)
        
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('$R_{test}$')
        ax.set_ylabel('Accuracy')
        ax.set_ylim((min(testAcc)-0.01, 1.01))
        fig.set_size_inches(8,3)
        fig.tight_layout()
    ax.semilogx(testR, testAcc, label=label, **plotKwargs)
    ax.legend()
    
    return ax, testR, testAcc


def plot_recog_positive_rates(net, gen_recog_data, upToR=float('-inf'), stopAtR=float('inf'), ax=None, label=None, burnin=True, **plotKwargs):          
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('$R_{test}$')
        ax.set_ylabel('True/false positive rate')
        ax.set_ylim((-0.01,1.01))
        fig.set_size_inches(8,3)
        fig.tight_layout()
        ax.plot([], 'k-', label='$P_{TP}$')
        ax.plot([], 'k--', label='$P_{FP}$')
    
    testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_recog_data, upToR=upToR, stopAtR=stopAtR, burnin=burnin)
    line = ax.semilogx(testR, truePosRate, label=label, **plotKwargs)[0]
    ax.semilogx(testR, falsePosRate, ls='--', color=line.get_color(), **plotKwargs)
    ax.legend()  
    
    return ax, testR, testAcc, truePosRate, falsePosRate   


def plot_reaction_time_curves(net, gen_data, Rs=None, T=5000, **genDataKwargs):
    _,axHist = plt.subplots()
    Rs = [1,10,20,30,50,100,T+1]
    pFP = np.empty(len(Rs)) #p(nov|fam,R)
    for i,R in enumerate(Rs):
        print('R={}'.format(R))
        data = gen_data(T=T, R=R, **genDataKwargs)
        db = net.evaluate_debug(data.tensors) 
        y = db['data'].tensors[1]
        if R<T:
            values,bins,_ = axHist.hist(db['a2'][y==1], bins=50, density=True, histtype='step', align='mid', label='R={}'.format(R))
            pFP[i] = values[bins[:-1]<0].sum()*np.diff(bins).mean()    
        else:
            values,bins,_ = axHist.hist(db['a2'][y==0], bins=50, density=True, histtype='step', align='mid', label='Novel')
            pFN = values[bins[:-1]>0].sum()*np.diff(bins).mean() #p(fam|nov)   
    pTP = 1-pFP #p(fam|fam,R)

    pError = pFP 
    pError[-1] = pFN #p(fam|nov) or p(nov|fam,R)
    pCorrect = 1-pError #p(fam|fam,R) or p(nov|nov)
    
    rtCorrect = 115-45*pCorrect
    rtError = 115-45*pError   
    Rs[-1] = 150
    
    #_,ax = plt.subplots()
    #ax.semilogx(Rs[:-1], pError[:-1], color='blue', marker='.', label='Error trials')
    #ax.semilogx(Rs[:-1], pCorrect[:-1], color='red', marker='.',  label='Correct trials')
    #ax.semilogx(Rs[-1], pError[-1], color='blue', marker='.')
    #ax.semilogx(Rs[-1], pCorrect[-1], color='red', marker='.')
    #ax.set_xlabel('R')
    #ax.set_ylabel('Proportion')
    #ax.set_xticks(Rs)
    #ax.set_xticklabels(list(Rs[:-1])+['Novel'])
    #ax.legend()
    
    _,axRT = plt.subplots()
    axRT.semilogx(Rs[:-1], rtError[:-1], color='blue', marker='.', label='Error trials')
    axRT.semilogx(Rs[:-1], rtCorrect[:-1], color='red', marker='.',  label='Correct trials')
    axRT.semilogx(Rs[-1], rtError[-1], color='blue', marker='.')
    axRT.semilogx(Rs[-1], rtCorrect[-1], color='red', marker='.')
    axRT.set_xlabel('R')
    axRT.set_ylabel('RT (ms)')
    axRT.set_xticks(Rs)
    axRT.set_xticklabels(list(Rs[:-1])+['Novel'])
    axRT.legend()


#%%############
### Helpers ###
###############
def left_justify_w(w, preventRepeatRows=False):
    w = w.clone()
    Nh,d = w.shape
    for i in range(d):
        for j in range(i+1,d):
            overlap = False     
            for x,y in zip(w[:,i], w[:,j]):
                    if x!=0 and y!=0:
                        overlap = True            
            if not overlap:
                wTemp = w.clone()
                wTemp[:,i] = wTemp[:,i] + w[:,j]
                wTemp[:,j] = 0
                if not hasRepeatRows(wTemp) or not preventRepeatRows:
                    w = wTemp
    return w

def hasRepeatRows(w):
    Nh,d = w.shape
    for i in range(d):
        for j in range(i+1,d):
            if (w[i]==w[j]).all():
                return True
    return False


def shuffle_w(w, rows=True, cols=True):
    w = w.clone().detach()
    if rows:
        for i,row in enumerate(w):
            w[i] = row[torch.randperm(w.shape[0])]
    if cols:
        for j,col in enumerate(w.t()):
            w[:,j] = col[torch.randperm(w.shape[1])]
    return w


def clean_up_w(w, cleanSmall=True, cleanLarge=True, largeVal=5, thres=3):
    w = w.clone().detach()
    if cleanLarge:
        w[w<-thres] = -largeVal
        w[w> thres] =  largeVal   
    if cleanSmall:
        w[w.abs()<thres]=0    
    return w


def get_recog_positive_rates(net, gen_recog_data, upToR=float('-inf'), stopAtR=float('inf'), burnin=True):
    from data import recog_chance 

    testR = []
    testAcc = []
    truePosRate = []
    falsePosRate = []

    acc = float('inf')
    truePos = 1; falsePos = 0
    R = 1
    if burnin:
        change_reset_fn_to_burned_in(net, gen_recog_data, burnT=int(math.log(0.001)/math.log(net.lam.item())))
    while (truePos > falsePos or R < upToR) and R < stopAtR:
        testData = gen_recog_data(R)
        with torch.no_grad():
            db = net.evaluate_debug(testData.tensors)
            out, acc, testData = db['out'], db['acc'], db['data']
            falsePos, truePos = error_breakdown(out, testData.tensors[1], th=0.5)        
        chance = recog_chance(testData)
        fracNov = (testData.tensors[1].round()==0).float().sum()/len(testData)
        acc2 = (1-falsePos)*fracNov + truePos*(1-fracNov)
        truePosRate.append( truePos )
        falsePosRate.append( falsePos )
        testAcc.append( acc )
        testR.append( R )
        print( 'R={}, truePos={:.3f}, falsePos={:.3f}, acc={:.3f}={:.3f}, (chance={:.3f})'.format(R, truePos, falsePos, acc, acc2, chance) )
        R = int(np.ceil(R*1.3))
    
    return testR, testAcc, truePosRate, falsePosRate

        
def change_reset_fn_to_burned_in(net, gen_data, burnT=2000, burnR=0):        
    if burnT > 0:
        if burnR<=0:
            burnR=int(math.ceil(burnT/100.))

        burnData = TensorDataset(torch.tensor([]),torch.tensor([]))
        while len(burnData) < burnT:
            dataChunk = gen_data(burnR)
            x = torch.cat((burnData.tensors[0], dataChunk.tensors[0]))
            y = torch.cat((burnData.tensors[1], dataChunk.tensors[1]))
            burnData = TensorDataset(x,y)
        net.evaluate(burnData.tensors)

        
        def reset_state(self):
            self.A = net.A.detach()    
        net.reset_state = types.MethodType(reset_state, net)


def error_breakdown(out, y, th=0.5):
    y = y.round()
    
    posOutIdx = out>th

    totPos = y.sum().item()
    totNeg = len(y)-totPos
    
    falsePos = (1-y)[posOutIdx].sum().item()        
    truePos = y[posOutIdx].sum().item()
    
    falsePosRate = falsePos/totNeg 
    truePosRate = truePos/totPos 

    return falsePosRate, truePosRate

    

def sort_w(w, alg='hierarchical', metric='euclidean', preproc=lambda x: x, sort='both'):
    """Attempts to reshuffle the rows/cols of w to expose underlying structure.
    Example:
        w = torch.eye(10) - torch.roll(torch.eye(5), 1, dims=1) #two diagonals 
        wShuf = w[torch.randperm(10),:] #shuffle rows
        wShuf = w[:,torch.randperm(10)] #shuffle cols
        wSort = sort_w(wShuf)
        
    """
    
    wPrep = preproc(w)
    
    for i in range(5):
        if alg == 'hierarchical':
            import scipy.cluster as spc
            
            if sort=='both' or sort=='rows':        
                Z = spc.hierarchy.linkage(wPrep, metric=metric)
                leaves = spc.hierarchy.leaves_list(Z)
                w = w[leaves]
                
            if sort=='both' or sort=='cols':        
                Z = spc.hierarchy.linkage(np.transpose(wPrep), metric=metric)
                leaves = spc.hierarchy.leaves_list(Z)
                w = w[:,leaves]
            
        elif alg == 'mds':
            from sklearn.manifold import MDS
            embedding = MDS(n_components=2)
            
            if sort=='both' or sort=='rows':        
                wE = embedding.fit_transform(wPrep)
                atan = np.arctan2(wE[:,0], wE[:,1])
                order = np.argsort(atan)
                w = w[order]
                
            if sort=='both' or sort=='cols':                    
                wE = embedding.fit_transform(np.transpose(wPrep))
                atan = np.arctan2(wE[:,0], wE[:,1])
                order = np.argsort(atan)
                w = w[:,order]
            
        else:
            raise ValueError('Invalid alg arg')
        
        wPrep = w
        
    return w


def get_R_max(net, gen_data, chance=0.99):
    Rhi = 1
    Rlo = 0
    Rmax = float('inf')
    while Rhi > Rlo:
        data = gen_data(Rhi)
        acc = net.accuracy(data.tensors).item()
        if acc > chance:
            Rlo = Rhi
            Rhi = min(Rhi*2, Rmax)            
        else:
            Rmax = Rhi
            Rhi = (Rhi+Rlo)/2    
        
        print( 'acc(Rhi)={:.3f}, chance={:.3f}, Rhi={}, Rlo={}'.format(acc,chance,Rhi,Rlo) )
    return Rlo     


def interleave(a,b):
    '''a,b are tensors, must be of same length and dimension. The first element of a is first.'''
    if len(a) != len(b):
        raise ValueError('a and b must be same length')
    s = list(a.shape)
    s[0] = s[0]*2
    x = np.zeros(s) 
    x[range(0, 2*len(a), 2)] = a
    x[range(1, 2*len(a), 2)] = b
    return x
    

def get_all_pkl(folder='.', exclude=None):
    files = filter(lambda f: f.endswith('.pkl'), os.walk(folder).next()[2])
    if exclude is not None:
        files = filter(lambda f: f.find(exclude)<0, files)
    return files 


def maxabs(listOfTensors):
    """Get largest absolute weight in list of pytorch tensors"""
    v = -np.inf 
    for w in listOfTensors:
        m = w.abs().max().item()
        if m > v: v = m
    return v  


def maxval(listOfTensors):
    """Get largest absolute weight in list of pytorch tensors"""
    v = -np.inf 
    for w in listOfTensors:
        m = w.max().item()
        if m > v: v = m
    return v


def ticks_off(ax):
    ax.tick_params(
        axis='both',        
        which='both',
        top=False,      
        left=False,
        bottom=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelbottom=False,
        labelright=False)
