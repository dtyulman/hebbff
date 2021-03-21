import matplotlib.pyplot as plt
import numpy as np

#%%
batchSize = [0,32,128,512]
gpuTime = np.array([443.9, 12.6, 3.4, 1.1])
cpuTime = np.array([163.6, 9.1, 6.2, 5.1])
gpuAcc = [0.79, 0.47, 0.56, 0.52] #cpu vs gpu acc should be comparable for a given batchSize since both versions saw the same number of samples and iters
cpuAcc = [0.83, 0.49, 0.48, 0.55] #difference comes from initialization and randomness in data generation
          
fig, ax = plt.subplots(2)
ax[0].plot(batchSize, cpuTime/gpuTime, marker='o')
ax[0].set_xlabel('BatchSize (proportional to Iters=CacheSize/BatchSize)')
ax[0].set_ylabel('[CPU time]/[GPU time]')
ax[0].set_title('Fixed CacheSize=2048 (D=N=25, R=10, T=200)')
ax[0].axhline(1, color='k')
ax[0].set_ylim([0,6])

#%%
batchSize = [0,32,128,512]

#gpuTime = np.array([4.6, 4.1, 4.2, 4.3])
#cpuTime = np.array([1.7, 2.9, 7.5, 24.4])
gpuAcc = [0.54, 0.38, 0.55, 0.48] #cpu vs gpu acc should be the same for a given batchSize
cpuAcc = [0.57, 0.53, 0.62, 0.48]

fig, ax = plt.subplots(2)

#gpuTime = np.array([2.7, 2.7, 2.7, 3.0])
#cpuTime = np.array([1.1, 2.0, 6.1, 20.2])

gpuTime = np.array([4.5, 4.7, 4.7, 6.5]) #D=N=100
cpuTime = np.array([2, 18, 65, 359])

line = ax[0].plot(batchSize, cpuTime, marker='o', label='cpu')
ax[0].plot(batchSize, gpuTime, ls='--', color=line[0].get_color(), marker='o', label='gpu')
ax[0].set_ylabel('Time')
ax[0].set_title('Iters=20 (D=N=100, R=10, T=200)')

ax[1].plot(batchSize, cpuTime/gpuTime, marker='o')
ax[1].set_xlabel('BatchSize')
ax[1].set_ylabel('[CPU time]/[GPU time]')
ax[1].axhline(1, color='k')
plt.tight_layout()

#%%
batchSize = [1,32,128] #512 out of memory (10GB)
lr=1e-3
time = [40, 72, 104]
acc = [0.745, 0.734, .738]

fig, ax = plt.subplots()
ax.plot(batchSize, gpuAcc, marker='o', label='lr={:.3f}'.format(lr))
ax.set_xlabel('Batch Size (proportional to CacheSize=BatchSize*Iters)')
ax.set_ylabel('Acc (GPU)')
ax.set_title('Fixed Iters=500 (D=N=25, R=10, T=200)')

lr=1e-2
batchSize = [1,32, 128]
gpuTime = [41, 71, 105] 
acc = [.880, .919, .924] #cpu vs gpu acc should be the same for a given batchSize
ax.plot(batchSize, gpuAcc, marker='o', label='lr={:.3f}'.format(lr))

lr=1e-1
batchSize = [1,32] #128 out of memory (2.5GB)
gpuTime = [41, 71, 105] 
acc = [.46, 0.98, 0.52] #cpu vs gpu acc should be the same for a given batchSize
ax.plot(batchSize, gpuAcc, marker='o', label='lr={:.3f}'.format(lr))

