import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset

from data import GenRecogClassifyData, generate_recog_data
from networks import HebbNet
from plotting import plot_train_perf


#%%
d=Nh=50
net = HebbNet([d, Nh, 2]) 

teacher = HebbNet([d, Nh, 1])
teacher.plastic = torch.tensor(False)

generator = GenRecogClassifyData(d=d, teacher=teacher)
gen_data = lambda R: generator(T=max(500, R*20), R=R, P=0.5, batchSize=-1)

#%%
net.fit('curriculum', gen_data, batchSize=None, filename=None, folder=None, R0=10)
plot_train_perf(net, chance=0.5)


#%%
gen_data_test = lambda R: generator(T=max(1000, R*20), R=R, P=0.5, batchSize=None)
data = gen_data_test(10)

out = net.evaluate(data.tensors)
acc = (out.round() == data.tensors[1]).float().mean(0)
print('recog acc={}, class acc={}'.format(*acc))

