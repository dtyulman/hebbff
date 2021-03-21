import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

#%% Plotting
def plot_hebb(update_fn, pre=None, post=None, surf=False): 
    pre = torch.tensor([-1.,1]) if pre is None else pre
    post = torch.arange(1, -0.05, -0.05) if post is None else post
    delta = update_fn(pre,post)      
    
    v = delta.abs().max().item()
    delta = delta.detach().numpy()

    
    #fig,ax = plt.subplots()
    #im = ax.imshow(delta, vmin=-v, vmax=v, cmap='RdBu_r', aspect='equal')
    #ax.set_xlabel('pre') 
    #ax.set_xticklabels(['']+list(pre)+[''])
    #ax.set_ylabel('post')
    #ax.set_yticklabels(['']+list(post)+[''])
    #ax.set_title('Change in synaptic weight')
    #ax.set_aspect('equal')
    #cb = fig.colorbar(im)
    
    fig = plt.figure()
    if surf:
        raise Exception('TODO: test this to make sure axes are corretly labeled')
        _pre, _post = np.meshgrid(pre, post)            
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(_pre, _post, delta, cmap='RdBu_r')
    else:
        pre = make_label(pre.numpy())
        post = make_label(post.numpy())
        ax = sns.heatmap(delta, cmap='RdBu_r', vmin=-v, vmax=v, square=True, xticklabels=pre, yticklabels=post)
        ax.set_xlabel('pre $x_j$') 
        ax.set_ylabel('post $h_i$')
        ax.set_title('$\Delta A_{ij}$')
        fig.set_size_inches(2.1,7)
        plt.tight_layout()
    
    return fig, ax


def make_label(arr, maxTicks=10):
    arr = arr.astype(str)
    if len(arr) <= maxTicks:
        return arr
    label = []
    for i,x in enumerate(arr[:-1]):
        if i% (len(arr)/maxTicks)==0:
            label.append(arr[i])
        else:
            label.append('')
    label.append(arr[-1])
    return label


def plot_W_seq(net, data, resetHebb=False):
    '''Plot sequence of W and A matrices during evaluation of data'''
    #TODO: this needs cleanup
    
    hist = debug_net(net, data)
    y_hist, A_hist, WA_hist  = hist['y'], hist['A'], hist['WA']
    WA_hist.insert(0, net.J1.clone().detach())
         
    Av = maxabs(A_hist)
    WAv = maxabs(WA_hist)    
    fig, axs = plt.subplots(2,len(data)+1)
    for i in range(len(data)):     
        imA = axs[0,i+1].matshow(A_hist[i], cmap='RdBu_r', vmin=-Av, vmax=Av)
        imWA = axs[1,i+1].matshow(WA_hist[i+1], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 

#        axs[0,i+1].set_title('y={:.2f} ({})'.format(y_hist[i], data[i][1].int().item()), 
#                             color='Red' if y_hist[i] != data[i][1].int().item() else 'Black')
                       
    imWA = axs[1,0].matshow(WA_hist[0], cmap='RdBu_r', vmin=-WAv, vmax=WAv) 
    [ax.axis('off') for ax in axs.flatten()]
    
    #axs[1,0].set_title('$W_{1,init}$')
    fig.set_size_inches(19,5.5)
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.93, .05+.5, 0.01, 0.4])  
    fig.colorbar(imA, cax=cax)
    cax = fig.add_axes([0.93, .05, 0.01, 0.4])  
    fig.colorbar(imWA, cax=cax)  

    return A_hist, WA_hist


def plot_xhy_seq(net, data, resetHebb=False):
    '''Plot sequence of hidden activities during evaluation of data
    '''
  
    hist = run_net(net, data, resetHebb)
    y_hist, h_hist = hist['y'], hist['h']
    
    fig = plt.figure()
    left = 0.05
    width = 0.5/len(data.tensors[0])  
    hv = maxabs(h_hist)
    for i,x in enumerate(data.tensors[0]):    
        gs = gridspec.GridSpec(4, 2)
        gs.update(left=left, right=left+width)
        ax1 = plt.subplot(gs[1:3,0])
        ax2 = plt.subplot(gs[:,1])
        left += width+0.4/len(data.tensors[0])  
        
        ax1.matshow(x.numpy().reshape(-1,1), cmap='binary')
        im = ax2.matshow(h_hist[i], cmap='Reds', vmin=0, vmax=hv)
        
        ax2.set_title('y={:.2f} ({})'.format(y_hist[i], data[i][1].int().item()), 
                      color='Red' if y_hist[i] != data[i][1].int().item() else 'Black')
        
        for ax in [ax1, ax2]:
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
    
    fig.set_size_inches(19,4)
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.93, .04+.33, 0.01, 0.25])  
    fig.colorbar(im, cax=cax) 
    
    return y_hist, h_hist

def plot_xahy_seq(net, data, resetHebb=False):
    '''Plot sequence of hidden activities during evaluation of data
    '''
  
    hist = run_net(net, data, resetHebb)
    y_hist, h_hist, a1, a2 = hist['y'], hist['h'], hist['a1'], hist['a2']
    
    fig = plt.figure()
    left = 0.05
    width = 0.5/len(data.tensors[0])  
    hv = maxabs(h_hist)
    for i,x in enumerate(data.tensors[0]):    
        gs = gridspec.GridSpec(4, 3)
        gs.update(left=left, right=left+width)
        ax1 = plt.subplot(gs[1:3,0])
        ax2 = plt.subplot(gs[:,1])
        ax3 = plt.subplot(gs[:,2])
        left += width+0.4/len(data.tensors[0])  
        
        ax1.matshow(x.numpy().reshape(-1,1), cmap='binary')
        im2 = ax2.matshow(a1[i], cmap='Greens', vmin=0, vmax=hv)        
        im3 = ax3.matshow(h_hist[i], cmap='Reds', vmin=0, vmax=hv)        

        ax2.set_title('$a_1$={:.2f} ({})'.format(a2[i], data[i][1].int().item()), 
                      color='Red' if int(y_hist[i]) != data[i][1].int().item() else 'Black')

        
        for ax in [ax1, ax2, ax3]:
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
    
    fig.set_size_inches(19,4)
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.9)
    cax2 = fig.add_axes([0.93, .05+.5, 0.01, 0.25])  
    fig.colorbar(im2, cax=cax2) 
    cax3 = fig.add_axes([0.93, .05, 0.01, 0.25])  
    fig.colorbar(im3, cax=cax3) 
    

def plot_corr(vec, fname=None):     
    R = np.corrcoef(vec)
    
    cmap = 'Reds' if np.all(R>=-0.1) else 'RdBu_r'
    (vmin, vmax) = (-np.abs(R).max(), np.abs(R).max()) if cmap=='RdBu_r' else (None,None)
    plt.matshow(R, cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.colorbar()               
    plt.gcf().set_size_inches(5,4)  
    plt.title(fname)
    plt.tight_layout()
    
    if fname:
        plt.savefig(fname+'.png')
    return R


def plot_weights(net):
   pass

 
    
def plot_xhuy_seq(net, data):
    hist = debug_net(net, data)
    
    fig = plt.figure()
    left = 0.05
    width = 0.5/len(data.tensors[0])  
    for i,x in enumerate(data.tensors[0]):    
        gs = gridspec.GridSpec(4, 4)
        gs.update(left=left, right=left+width)
        ax1 = plt.subplot(gs[1:3,0])
        ax2 = plt.subplot(gs[:,1])
        ax3 = plt.subplot(gs[:,2])
        ax4 = plt.subplot(gs[1:3,3])
        left += width+0.4/len(data.tensors[0])  
        
        im1 = ax1.matshow(x.numpy().reshape(-1,1), cmap='binary')
        im2 = ax2.matshow(hist['h'][i].reshape(-1,1), cmap='Greens', vmin=0, vmax=1)        
        im3 = ax3.matshow(hist['r'][i].reshape(-1,1), cmap='Reds', vmin=0, vmax=1)        
        im4 = ax4.matshow(hist['y'][i].reshape(-1,1), cmap='binary')        

        ax2.set_title('' if (hist['y'][i].sign()==data.tensors[1][i]).all() or torch.isnan(data.tensors[1][i]).all() else 'error')
        
        if i == 0:
            ax1.set_xlabel('x')
            ax2.set_xlabel('h\nt={}'.format(i))
            ax3.set_xlabel('r')
            ax4.set_xlabel('y')
        else:
            ax2.set_xlabel('\nt={}'.format(i))


        
        for ax in [ax1, ax2, ax3, ax4]:
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
    
    fig.set_size_inches(15,4)
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.9)
    cax2 = fig.add_axes([0.93, .55, 0.01, 0.33])  
    fig.colorbar(im2, cax=cax2, label='hidden (h)') 
    cax3 = fig.add_axes([0.93, .11, 0.01, 0.33])  
    fig.colorbar(im3, cax=cax3, label='gate (r)') 


#%% Execution   
   
def debug_net(net, data):
    execHist = {k:[] for k in net(data.tensors[0][0], debug=True).keys()}    
    
    net.reset_state()
    for x in data.tensors[0]:
        out = net(x, debug=True)
        for var in out.keys():
            execHist[var].append( out[var] )
      
    return execHist
            





def run_net(net, data, resetHebb=False, zeroW1=False, zeroA=False): 
    '''Run the network on dataset, keeping track of all activities and outputs throughout'''    
    if resetHebb:
        net.reset_A() 
    
    if zeroW1:
        net.w1 = torch.nn.Parameter(torch.zeros_like(net.w1))
    
    hist = {'y':[],
            'h':[],
            'A':[],
            'WA':[net.w1.clone().detach()],
            'a1':[],
            'a2':[]}   
    
    for i,x in enumerate(data.tensors[0]):
        if zeroA:
            net.reset_A()   
            
        a1,h,a2,y = net(x, debug=True)
        hist['a1'].append( a1.clone().detach().reshape(-1,1) )
        hist['h' ].append(  h.clone().detach().reshape(-1,1) )
        hist['a2'].append( a2.item() )
        hist['y' ].append(  y.item() )
        hist['A' ].append( net.A.clone().detach() )
        hist['WA'].append( hist['A' ][-1] + net.w1.clone().detach() )
    
    return hist


                
#%% Miscellaneous               

def list2tensor(listOfTensors):
    tensor = torch.zeros(len(listOfTensors), listOfTensors[0].shape[0])                  
    for i in range(len(listOfTensors)):
        tensor[i,:] = listOfTensors[i].view(-1) 
    return tensor
