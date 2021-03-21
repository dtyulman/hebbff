import torch
import torchviz
from minimal import Synapse

pre = torch.arange(3., requires_grad=True)
post = torch.arange(5., requires_grad=True)
syn = Synapse()
delta = syn(pre,post)
print delta

torchviz.make_dot(syn(pre,post))

#%%