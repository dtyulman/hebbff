import types, os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import TensorDataset

from data import prob_repeat_to_frac_novel, GenRecogClassifyData, generate_recog_data
from plotting import plot_recog_generalization, plot_recog_positive_rates
from net_utils import load_from_file
from networks import DetHebb

#%%
def forward_remove_readout(self, x): 
    #x.shape = [batchSize,3,224,224]
    x = self.conv1(x)   #shape = [batchSize,64,112,112]
    x = self.bn1(x)     #shape = [batchSize,64,112,112]
    x = self.relu(x)    #shape = [batchSize,64,112,112]
    x = self.maxpool(x) #shape = [batchSize,64,56,56]

    x = self.layer1(x) #shape=[batchSize, 64, 56, 56]
    x = self.layer2(x) #shape=[batchSize, 128, 28, 28]
    x = self.layer3(x) #shape=[batchSize, 256, 14, 14]
    x = self.layer4(x) #shape=[batchSize, 512, 7, 7]

    x = self.avgpool(x)     #shape=[batchSize, 512, 1, 1]
    x = torch.flatten(x, 1) #shape=[batchSize, 512]
#    x = self.fc(x)          #shape=[batchSize, 1000]
    
    return x 

resnet = models.resnet18(pretrained=True)
resnet._forward_impl = types.MethodType(forward_remove_readout, resnet)
resnet.eval()

#%%
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#os.chdir('BradyOliva2008_UniqueObjects')
os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/BradyOliva2008_OneOfPairs')
files = [f for f in os.listdir('.') if f.lower().endswith('.jpg')]
#assert len(files)==2400
assert len(files)==200


allImages = torch.empty(len(files), 3, 224, 224)
for i,fname in enumerate(files):
    image = Image.open(fname)
    allImages[i] = preprocess(image)

#%%
with torch.no_grad():
    sampleSpace = resnet(allImages)

#%%

downsampled = sampleSpace[:,:50]
#torch.save(downsampled, 'BradyOliva2008_OneOfPairs_ResNet18_d=50.pkl')
normalized = downsampled - downsampled.mean(dim=1).reshape(-1,1)
normalized = np.sqrt(50) * normalized / normalized.norm(dim=1).reshape(-1,1)
#torch.save(normalized, 'BradyOliva2008_OneOfPairs_ResNet18_d=50_normalize.pkl')
binarized = normalized.sign()
torch.save(binarized, 'BradyOliva2008_OneOfPairs_ResNet18_d=50_binarize.pkl')


#%%   
useResNetData = True

if useResNetData:
    os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/BradyOliva2008_UniqueObjects')
    images = torch.load('BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl')
    dummyClasses = torch.zeros(images.shape[0],1)
    sampleSpace = TensorDataset(images, dummyClasses)
    generator = GenRecogClassifyData(sampleSpace=sampleSpace)
    def generate_recog_images(T, d, R, P=0.5, batchSize=None, multiRep=False):
        x,y = generator(T, R, P, batchSize=batchSize, multiRep=multiRep).tensors
        return TensorDataset(x, y[...,0:1])
else:
    generate_recog_images = generate_recog_data
    

#%%
Rtest = np.unique(np.logspace(0, 3, 20, dtype=int))
P = 0.5 
f = prob_repeat_to_frac_novel(P, multiRep=False)

n = 4
D = 50-n
d = D+n
net = DetHebb(D, n, f=f, Ptp=0.99, Pfp=0.01)
net.evaluate(generate_recog_data(T=5000, d=d, R=5000).tensors[0]) #burn-in A to get to steady-state

Ptp = np.zeros(len(Rtest))
Pfp = np.zeros(len(Rtest))
for i,R in enumerate(Rtest):    
    X,Y = [xy.numpy() for xy in generate_recog_images(T=min(2400, max(1000, R*20)), d=d, R=R, P=P, multiRep=False).tensors]
    f = (1 - sum(Y)/len(Y))[0]
      
    Yhat = net.evaluate(X)
    Ptp[i], Pfp[i] = net.true_false_pos(Y, Yhat)
    acc = net.accuracy(Y, Yhat)
    acc2 = (1-f)*Ptp[i] + f*(1-Pfp[i]) #sanity check   
    print('R={}, f={:.3f}, Ptp={:.3f}, Pfp={:.3f}, acc={:.4f}={:.4f}'.format(R, f, Ptp[i], Pfp[i], acc, acc2))    
    
#%%
_,ax = plt.subplots()
line1 = ax.semilogx(Rtest, Ptp, label='DetHebb, '+'ResNet data' if useResNetData else 'random data')[0]
line2 = ax.semilogx(Rtest, Pfp, color=line1.get_color(), ls='--')[0]
ax.set_title('d={}, N={}, f={:.2f} \n$P_{{fp}}$={:.2f}, $P_{{tp}}$={:.2f}, $\\alpha$={:.2f}, $\gamma$={:.2f}, b={:.2f},'.format(net.d, net.N, net.f, net.Pfp, net.Ptp, net.a, net.gam, net.b))
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('True/false positive rate')
ax.legend()
ax.set_ylim(0,1)

#%%
os.chdir('/Users/danil/My/School/Columbia/Research/ongoing_plasticity/results/2020-07-31')
net = load_from_file('HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl')
Nh, d = net.w1.shape      
    
label = 'HebbFF, '+'ResNet data' if useResNetData else 'random data'
upToR = float('inf')
stopAtR = 1000
gen_data = lambda R: generate_recog_images(T=min(2400, max(1000, R*20)), d=d, R=R, P=0.5, multiRep=False)
#axGen = axTfp = None
axTfp,Rs,acc,truePos,falsePos = plot_recog_positive_rates(net, gen_data, upToR=upToR, stopAtR=stopAtR, ax=axTfp, label=label)

    



