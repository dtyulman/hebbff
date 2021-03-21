import joblib
import torch
torch.set_printoptions(precision=4, sci_mode=False, threshold=5000)
import matplotlib.pyplot as plt

from data import generate_recog_data
from net_utils import load_from_file
from plotting import plot_ROC, plot_output_distr

#%%

d = 25
T = 25

#%%
reset = True
Nh = 10
fname = 'HebbNet_R=curriculum_Nh={}.pkl'.format(Nh)
net = load_from_file(fname)

#net = joblib.load(fname)
#net.requires_grad_(False)

#%%
if reset:
    net.reset_state()

familiar = generate_recog_data(T=T, d=d, R=T, P=0.5)    
net.plastic = True    
for x in familiar.tensors[0]:
    net(x)
net.plastic = False

novel = generate_recog_data(T=T, d=d, R=T, P=0.5)
yFam = torch.empty_like(familiar.tensors[1])
yNov = torch.empty_like(novel.tensors[1])
for t, (xFam, xNov) in enumerate(zip(familiar.tensors[0], novel.tensors[0])):
    yFam[t] = net(xFam)
    yNov[t] = net(xNov)

result = torch.cat([yFam, yNov, (yFam>yNov).float()], dim=1)

acc = (yFam > yNov).float().mean()    


print result
print acc
    
#%%

idx = torch.sort(yNov.flatten(), descending=True)[1][0].item()
(((familiar.tensors[0] - novel.tensors[0][-8])/2).abs().sum(1).int())
#
#
#print familiar[-37][0]
#print novel[-5][0]
#    

#%%

ax=None
for R in [1,5,10,20,100]:
    data = generate_recog_data(T=5000, d=25, R=R)
    out = net.evaluate(data.tensors)
    y = data.tensors[1]
    acc = net.accuracy(data.tensors)
    ax,fpr,tpr,auc = plot_ROC(out,y, label='R={}, acc={:.2f}'.format(R,acc), ax=ax)


