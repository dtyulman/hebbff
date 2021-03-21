import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.directory'] = '~'


from data import generate_recog_data
from net_utils import load_from_file
from plotting import plot_recog_generalization

def run_debug(net, data):
    debug = {}
    debug['a1'] = torch.empty(T,Nh)
    debug['h'] = torch.empty(T,Nh)
    debug['Wxb'] = torch.empty(T,Nh)
    debug['Ax'] = torch.empty(T,Nh)
    debug['a2'] = torch.empty_like(data.tensors[1])
    debug['out'] = torch.empty_like(data.tensors[1])
    for t,(x,y) in enumerate(data):
        debug['Ax'][t] = torch.mv(net.A, x) 
        debug['a1'][t], debug['h'][t], debug['a2'][t], debug['out'][t] = net.forward(x, debug=True)
        
        w1 = net.g1*net.w1 if not torch.isnan(net.g1) else net.w1
        debug['Wxb'][t] = torch.addmv(net.b1, w1, x) 
    debug['acc'] = net.accuracy(data.tensors)
    return debug

def plot_W_corr(w, title='', rowvar=True):
    R = np.corrcoef(w.detach(), rowvar=rowvar)
    plt.matshow(R, vmin=-1., vmax=1., cmap='RdBu_r')
    plt.title('$W_1$ {} corr $E|R|={:.3f}$'.format('row' if rowvar else 'column', np.abs(R).mean()) + ' ({})'.format(title) if title else title)
    plt.xlabel('$W_{{i,:}}$' if rowvar else '$W_{{:,i}}$')
    plt.ylabel('$W_{{j,:}}$' if rowvar else '$W_{{:,j}}$')
    plt.colorbar()  
#    plt.savefig('W1corr_{}_{}'.format(title, 'rows' if rowvar else 'cols'))
    return R
    
#%% Plot correlation matrices b/w the rows of W1 before/after shuffling 
  
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl'
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl'
#net = load_from_file(fname)
w = net.w1.clone().detach()
#gen_data = lambda R: generate_recog_data(T=R*50, d=w.shape[1], R=R, P=0.5, interleave=True, multiRep=True)
#Gax = None

#Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='base ')
R = plot_W_corr(net.w1, rowvar=True, title='base')
R = plot_W_corr(net.w1, rowvar=False, title='base')


net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 
#Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf ')
R = plot_W_corr(net.w1, rowvar=True, title='shuf')
R = plot_W_corr(net.w1, rowvar=False, title='shuf')


for i,row in enumerate(w):
    net.w1.data[i] = row[torch.randperm(w.shape[1])]
#Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf_r ')
R = plot_W_corr(net.w1, rowvar=True, title='shuf_r')
R = plot_W_corr(net.w1, rowvar=False, title='shuf_r')


for j,col in enumerate(w.t()):
    net.w1.data[:,j] = col[torch.randperm(w.shape[0])]
#Gax,_,_ = plot_recog_generalization(net, gen_data, ax=Gax, label='shuf_c ')
R = plot_W_corr(net.w1, rowvar=True, title='shuf_c')
R = plot_W_corr(net.w1, rowvar=False, title='shuf_c')


#%% Plot correlation matrices b/w h_i and h_j (i.e. vector index is time)
def plot_h_corr(h, title=''):
    R = np.corrcoef(h.detach(), rowvar=False)
    plt.matshow(R, vmin=-1., vmax=1., cmap='RdBu_r')
    plt.title('Hidden unit corr ($E[|R|]={:.2f}$)'.format(np.abs(R).mean()) + ' ({})'.format(title) if title else title)
    plt.xlabel('$h_i$')
    plt.ylabel('$h_j$')
    plt.colorbar()   
    return R


#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=scalar_w2init=scalar.pkl'
#fname = 'HebbNet[25,25,1]_w1init=randn_b1init=randn_w2init=randn.pkl'
#net = load_from_file(fname)
w = net.w1.clone().detach()
Nh,d = net.w1.shape     

R = 10
T = 10000
data = generate_recog_data(T=T, d=d, R=R, interleave=True, multiRep=True)

out = run_debug(net, data)
R_base = plot_h_corr(out['h'], title='base')

net.w1.data = w.flatten()[torch.randperm(w.shape.numel())].reshape(w.shape) 
out = run_debug(net, data)
R_shuf = plot_h_corr(out['h'], title='shuf')

for i,row in enumerate(w):
    net.w1.data[i] = row[torch.randperm(w.shape[0])]
out = run_debug(net, data)
R_shuf_r = plot_h_corr(out['h'], title='shuf_r')

for j,col in enumerate(w.t()):
    net.w1.data[:,j] = col[torch.randperm(w.shape[0])]
out = run_debug(net, data)
R_shuf_c = plot_h_corr(out['h'], title='shuf_c')
#%%
def cov(M):        
    C = np.empty([len(M),len(M)])
    for i in range(len(M)):
        for j in range(len(M)):
            C[i,j] = np.dot(M[i,:]-M[i,:].mean(), M[j,:]-M[j,:].mean())/(len(M)-1)
    return C

def plot_rows(M, ax, color='k'):
    for row in M:
        ax.arrow(0, 0, row[0], row[1], head_width=0.05, head_length=0.1, fc=color, ec=color)

def rotate(M, th):
    rot = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    M_r = np.matmul(M,rot.T)
    return M_r

def center(M):
    M_ctr = np.empty([2,2])
    for i in range(len(M)):
        M_ctr[i,:] = M[i,:]-M[i,:].mean()
    return M_ctr


_,ax = plt.subplots()
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_aspect('equal')
ax.grid()

M = rotate(np.array([[ 1, 0],
                     [ 0, 1]]), 0.000001)
#M = np.random.randn(2,2)
    
plot_rows(M, ax, 'k')
C = np.cov(M)
print('C = \n{}'.format(C))
R = np.corrcoef(M)
print('R = \n{}'.format(R))

M_ctr = center(M)
plot_rows(M_ctr, ax, 'b')

M_r = rotate(M, np.pi/4)
plot_rows(M_r, ax, 'r')

M_r_ctr = center(M_r)
plot_rows(M_r_ctr, ax, 'm')

C_r = np.cov(M_r)
print('C_r = \n{}'.format(C_r))
R_r = np.corrcoef(M_r)
print('R_r = \n{}'.format(R_r))


