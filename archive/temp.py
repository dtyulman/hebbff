import os
from net_utils import load_from_file
import plotting
import matplotlib.pyplot as plt

#root = '/Users/danil/Desktop/Habanero/vary_Nh_noMultiRep/'
#root = '/Users/danil/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/habaxfer.rcs.columbia.edu – SFTP/vary_Nh_noMultiRep'
root = '/Users/danil/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/habaxfer.rcs.columbia.edu – SFTP/vary_Nh_allBinInit'


for folder in next(os.walk(root))[1]:
    #LATEST: 3.3GB
    if folder == 'converged' or folder.find('backup')>=0:
        continue
    print('----', folder, '----')

    ax = None
    for fname in filter(lambda f: f.endswith('.pkl'), next(os.walk(os.path.join(root, folder)))[2]):
        label = fname[fname.find('['):fname.find('_')]
        d,N,_ = [int(x) for x in label.lstrip('[').rstrip(']').split(',')]
        print('N={}, d={}'.format(N,d))
        try:
            net = load_from_file(os.path.join(root, folder, fname))
        except (RuntimeError, EOFError) as e:
            print(e)
            continue

        if net.hist['iter'] > net.hist['increment_R'][-1][0]+2000000:
            print(label+ ' CONVERGED \n')
            if ax is None:
                _,ax = plt.subplots()
            ax.plot([], ls=None, label=label+' CONVERGED')
            continue

        iters, Rs = zip(*net.hist['increment_R'])
        iters = list(iters)
        Rs = list(Rs)
        iters.append( net.hist['iter'] )
        Rs.append( net.hist['increment_R'][-1][1] )
        ax = plotting.plot_R_curriculum(iters, Rs, label=label, ax=ax)
    ax.legend()

    # for fname in filter(lambda f: f.find('log_')>=0, next(os.walk(os.path.join(root, folder)))[2]):
    #     fname = os.path.join(root, folder, fname)
    #     # print(fname)
    #     with open(fname, 'rb') as f:
    #         f.seek(-2, os.SEEK_END)
    #         while f.read(1) != b'\n':
    #             f.seek(-2, os.SEEK_CUR)
    #         line = f.readline().decode()
    #         # print(line)
    #     if not (line.find('elapsed')>=0 or line.find('TIME LIMIT')>=0):
    #         print('ERR')
    #         print(fname)
    #         print(line)
    #         break

#%%
import numpy as np
import matplotlib.pyplot as plt
from net_utils import load_from_file
import plotting
import math

#          N= 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
converged = [[1, 1, 1, 1,  1,  1,  1,   1,   0,   0   ], #d= 25
             [1, 1, 1, 1,  1,  1,  0,   0,   0,   0   ], #   50
             [1, 1, 1, 1,  1,  0,  0,   0,   0,   0   ], #   100
             [1, 1, 1, 1,  0,  0,  0,   0,   0,   0   ], #   200
             [1, 1, 1, 0,  0,  0,  0,   0,   0,   0   ]] #   400

ds = np.array([25, 50, 100, 200, 400])
Ns = np.array([ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
Rmax = np.full((len(ds), len(Ns)), np.nan)
for i,d in enumerate(ds):
    for j,N in enumerate(Ns):
        if not converged[i][j]:

            continue
        R0 = int(math.ceil(.025*(N*d)**0.8))
        fname = '/Users/danil/Desktop/backup_20210319/d_{}/HebbNet[{},{},1]_train=cur{}_incr=plus1_forceAnti_reparamLam.pkl'.format(d,d,N, R0)
        try:
            net = load_from_file(fname)
            Rmax[i,j] = net.hist['increment_R'][-1][1]
        except IOError:
            print( N, d)
            pass
#
fig,ax = plt.subplots(2,2)
# selected nets to plot R over time
netDims = [(25,16), (50,16), (100,16), (200,16)]
Riter = []
for i,(d,N) in enumerate(netDims):
    R0 = int(math.ceil(.025*(N*d)**0.8))
    fname = '/Users/danil/Desktop/backup_20210319/d_{}/HebbNet[{},{},1]_train=cur{}_incr=plus1_forceAnti_reparamLam.pkl'.format(d,d,N, R0)
    net = load_from_file(fname)
    iters, Rs = map(list, zip(*net.hist['increment_R']))
    iters.append( net.hist['iter'] )
    Rs.append( net.hist['increment_R'][-1][1] )
    label = 'd={}, N={}'.format(d,N)
    ax[0,0] = plotting.plot_R_curriculum(iters, Rs, label=label, ax=ax[0,0])
    ax[0,0].legend()


#%%
import torch
import plotting
from net_utils import load_from_file

datasetType = 'OneOfPairs' #'UniqueObjects'
preprocType = '_d=50_binarize' #'' #'_d=50_normalize'
data = torch.load('BradyOliva2008_{}_ResNet18{}.pkl'.format(datasetType, preprocType))
feat = data

if preprocType == '':
    net = load_from_file('HebbFeatureLayer[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18.pkl_Nx=512_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl')
    with torch.no_grad():
        feat = torch.empty(len(data),50)
        for i,x in enumerate(data):
            feat[i] = net.featurizer(x)

#%%
fig,ax = plt.subplots()

feat_subset = feat[0:100]
h_il = plotting.interleave(feat_subset, torch.full(feat_subset.shape, float('nan'))).T
im = ax.matshow(h_il, cmap='Greys')

fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
fig.colorbar(im, cax=cax)

ax.set_title('{} {}'.format(datasetType, preprocType))
ax.set_ylabel('x')
ax.set_frame_on(False)
ax.tick_params(
    axis='both',
    which='both',
    top=False,
    left=False,
    bottom=False,
    right=False,
    labeltop=False,
    labelleft=False,
#        labelbottom=False,
    labelright=False)

#%%
C = np.corrcoef(feat)
plt.matshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
plt.colorbar()

#%%
flat = feat.flatten()

fig, ax = plt.subplots()
ax.hist(flat)
ax.set_title('{} {}'.format(datasetType, preprocType))
ax.set_xlabel('x_i')


#%%
from PIL import Image
import os

files = os.walk('.').next()[-1]
fig,axs = plt.subplots(10,10)
axs = axs.flatten()
for i,f in enumerate(files):
    im = Image.open(f)
    axs[i].imshow(im)
    axs[i].axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

os.chdir('pair2')

files = os.walk('.').next()[-1]
fig,axs = plt.subplots(10,10)
axs = axs.flatten()
for i,f in enumerate(files):
    im = Image.open(f)
    axs[i].imshow(im)
    axs[i].axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

os.chdir('..')
