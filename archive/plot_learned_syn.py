import torch
import joblib
import matplotlib.pyplot as plt
from torch_net_utils import plot_hebb
from data_utils import generate_recognition_data, recog_chance
from neural_net_utils import plot_train_perf, Synapse

#%%
d = 25
P = 0.5
intlv = True

for R in [1,2,5,10,20]:
    for i in [2,3]:#,2,3]:
    #    fname = 'LearnedSynHebb_R={}_intlv={}_({}).pkl'.format(R, intlv, i)
        fname = 'LearnedSynHebb_R={}_Ns=50_({}).pkl'.format(R, i)
        net = joblib.load(fname)
        
        testData = generate_recognition_data(T=10000, d=d, R=R, P=P, interleave=intlv)
        plot_train_perf(net, recog_chance(testData), testAcc=net.accuracy(testData).item(), 
                        title='R={}'.format(R))
        plt.savefig(fname[:-4]+'_train.png') 
    
        fig,ax = plot_hebb(net.synaptic_update)
        ax.set_title('$\Delta A_{ij}$ ' + '($\lambda={:.2}$)'.format(net.lam.item()))
        plt.savefig(fname[:-4]+'_syn.png') 
    
#%% Vanilla Hebbian rule
#for R,lam,eta in [(1, -.47, 0.106),]:#( 2, 0.584, 0.08), (7, 0.973, -0.138)]:
#    vanilla_hebb = lambda x,h: eta*torch.ger(h,x)
#    _,ax = plot_hebb(vanilla_hebb, pre=torch.arange(1.,-1.,-0.05), post=torch.arange(-1.,1.,0.05))
#    ax.set_title(ax.get_title() + ' ($\lambda={:.2}$)'.format(lam))
##    plt.savefig('vanilla_hebb_R={}_lam={}_eta={}.png'.format(R,lam,eta))
    
#%% Can Ns=5 learn multiplications?
#
#pre = torch.arange( 1., -1.01, -0.01)    
#post = torch.arange(-1. , 1.01,  0.01)
#tgt = torch.ger(post, pre)
#
#S = Synapse(Nh=5)
#opt = torch.optim.Adam(S.parameters())
#
#for i in range(10000000):
#    opt.zero_grad()
#    out = S(pre,post)
#    loss = ((tgt-out)**2).mean()
#    if i%1000 == 0:
#        print i, loss
#    loss.backward()
#    opt.step()
#    
#pre_ = pre#torch.tensor([1., -1.])
#post_ = post#torch.arange(-1, 1.05, .05)
#_,ax = plot_hebb(S, pre_, post_)
##plt.savefig('syn_N=5_rel.png')
#
#plot_hebb(lambda pre,post: torch.ger(post,pre), pre_, post_)
#plt.savefig('hebb_rel.png')


