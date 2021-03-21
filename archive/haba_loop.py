"""Script to loop through all the parameters and launch a batch process. This gets exectuted on the cluster
using a command sent from haba_run.py via ssh.

Example:
script = 'script.py'
argsList = [(arg1A, arg2A, arg3A), (arg1B, arg2B, arg3B)] #each entry is a tuple of args for <script>
NUM_TRIALS = 1 #number of copies of each argument configuration
"""

import sys, os, itertools, math
from haba_run import sbatch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time

script = 'script_train_recog.py'
NUM_TRIALS=1
RAM = []
useShortQueue = True


#%%
#argsList = [
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--train', 'inf', '--forceHebb', '--R0', 3,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--train', 'inf', '--forceHebb', '--R0', 6,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--train', 'inf', '--forceAnti', '--R0', 3,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--train', 'inf', '--forceAnti', '--R0', 6,'--noMultiRep', '--folder', 'final_figs/fig3'),
#
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100, '--train', 'inf', '--forceHebb', '--R0', 3,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100, '--train', 'inf', '--forceHebb', '--R0', 6,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100, '--train', 'inf', '--forceAnti', '--R0', 3,'--noMultiRep', '--folder', 'final_figs/fig3'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100, '--train', 'inf', '--forceAnti', '--R0', 6,'--noMultiRep', '--folder', 'final_figs/fig3'),

#        ('--net', 'nnLSTM', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 3,6, '--noMultiRep', '--folder', 'RNN'),
#        ]



#%%
#argsList = [
#            ('--cont', '--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--forceHebb', '--train', 'cur', '--noMultiRep',  '--folder', 'forceHebbCurriculum'),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'inf', '--R0', 1, '--noMultiRep',  '--folder', 'fig4abc'),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--Rf', 7, '--noMultiRep',  '--folder', 'fig4abc'),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--Rf', 14, '--noMultiRep',  '--folder', 'fig4abc'),
#            ('--net', 'HebbNet', '--Nh', 100, '--d', 100, '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--noMultiRep',  '--folder', 'cfBogacz'),
#            ]


#%%
#argsList = [
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'inf', '--R0', 7, '--noEarlyStop', '--folder', 'over_Rmax' ),
#
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf100.pkl', '--train', 'inf', '--R0', 7, '--folder', 'over_Rmax' ),
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf40.pkl', '--train', 'inf', '--R0', 7, '--folder', 'over_Rmax' ),
#
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf40.pkl', '--train', 'inf', '--R0', 14, '--folder', 'over_Rmax' ),
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf40.pkl', '--train', 'inf', '--R0', 100, '--folder', 'over_Rmax' ),
#
#
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf100.pkl', '--train', 'inf', '--R0', 14, '--folder', 'over_Rmax' ),
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf100.pkl', '--train', 'inf', '--R0', 40, '--folder', 'over_Rmax' ),
#
#
#            ('--load', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'inf', '--R0', 500, '--folder', 'over_Rmax' ),
#            ('--cont', 'HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'inf', '--R0', 1000, '--folder', 'over_Rmax' ),
#            ]


#%%
#argsList = [
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--train', 'inf', '--R', 5, '--folder', 'undertrained/tensorboard' )
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--train', 'cur', '--w1init', 'diag+-', '--w2init', 'randn', '--folder', 'randn_diags_lin_scal/tensorboard' ),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--train', 'cur', '--w1init', 'diag+-', '--w2init', 'scalar', '--b1init', 'scalar', '--folder', 'randn_diags_lin_scal/tensorboard' ),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--train', 'cur', '--w1init', 'randn', '--w2init', 'randn', '--folder', 'randn_diags_lin_scal/tensorboard' ),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--train', 'cur', '--w1init', 'randn', '--w2init', 'scalar', '--b1init', 'scalar', '--folder', 'randn_diags_lin_scal/tensorboard' ),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--eta0', 0.2, '--lam0', 0.8,                '--train', 'cur', '--folder', 'hebb/tensorboard' ),
#            ('--net', 'HebbNet', '--Nh', 25, '--d', 25,  '--eta0', 0.2, '--lam0', 0.8, '--forceHebb', '--train', 'cur', '--folder', 'hebb/tensorboard' )
#           ]

#%% Network with variable lambda, and a trainable parameter to choose between variable and fixed
#argsList = [('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25,                                                                     '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25, '--lam_slider0', 0, '--freeze', 'lam_slider',                       '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25, '--lam_slider0', 1, '--freeze', 'lam_slider',                       '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25,                                               '--w2init', 'scalar', '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25, '--lam_slider0', 0, '--freeze', 'lam_slider', '--w2init', 'scalar', '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ('--cont', '--net', 'HebbVariableLam', '--Nh', 25, '--d', 25, '--lam_slider0', 1, '--freeze', 'lam_slider', '--w2init', 'scalar', '--train', 'cur', '--folder', 'variable_lambda/tensorboard'),
#            ]

#%% Larry init
#argsList = [
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 64,                                                                                       '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 32,                                                                                       '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 32,                                           '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 64,                                           '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 32, '--w1init', 'all_binary', '--gain', 'w1',                                             '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 64, '--w1init', 'all_binary', '--gain', 'w1',                                             '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 32, '--w1init', 'all_binary',                                                             '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--cont', '--net', 'HebbNet', '--Nh', 64, '--d', 64, '--w1init', 'all_binary',                                                             '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 32, '--w1init', 'all_binary', '--gain', 'w1', '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 64, '--w1init', 'all_binary', '--gain', 'w1', '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 32, '--w1init', 'all_binary',                 '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#            ('--net', 'HebbNet', '--Nh', 64, '--d', 64, '--w1init', 'all_binary',                 '--b1init', 'scalar', '--w2init', 'scalar', '--train', 'cur',  '--folder', 'allBinaryInit/tensorboard'),
#           ]

#%% vary d, Nh

#useShortQueue = False
#argsList = []
#
#for i,d in enumerate([25, 50, 100, 200, 400]):
#    for startNew in [False]:
#        if startNew:
#            NhList = [[1],    #25 #comp to temp folder
#                      [1],    #50 #comp to temp folder
#                      [1],    #100  #comp to temp folder
#                      [1],    #200 #comp to temp folder
#                      [1],    #400 #comp to temp folder
#                      ]
#            R0 = [1, 2, 4, 7, 12]
#        else:
#            NhList = [[512],       #25  conv=[2,4,8,16,32,64,128,256]
#                      [256, 512],       #50  conv=[1,2,4,8,16,32,64,128,]
#                      [64, 128, 256],    #100 conv=[2,4,8,16,32, ]
#                      [64, 128],      #200 conv=[1,2,4,8,16,32,]
#                      [32, 64],     #400 conv=[1,2,4,8,16,]
#                      ]
#
#        for Nh in NhList[i]:
#            if Nh*d > 5*10**4:
#                continue
#
#            if startNew:
#                argsList.append(( '--net', 'HebbNet', '--Nh', Nh, '--d', d, '--forceAnti', '--w1init', 'randn', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', R0[i], '--noMultiRep', '--folder', 'vary_Nh_noMultiRep/d_{}/'.format(d)))
#            else:
#                fname = 'HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,Nh)
#                argsList.append(('--filename', fname, '--cont', '--train', 'cur', '--noMultiRep', '--folder', 'vary_Nh_noMultiRep/d_{}'.format(d)))
#
#            if Nh*d >= 400*64:
#                RAM.append(32)
#            else:
#                RAM.append(16)

#%% vary d, Nh (allBinInit, simple avg readout)

useShortQueue = False
argsList = []

#          N= 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
converged = [[1, 1, 1, 1,  1,  1,  1,   1,   0,   0   ], #d= 25
             [1, 1, 1, 1,  1,  1,  0,   0,   0,   0   ], #   50
             [1, 1, 1, 1,  1,  0,  0,   0,   0,   0   ], #   100
             [1, 1, 1, 1,  0,  0,  0,   0,   0,   0   ], #   200
             [1, 1, 1, 0,  0,  0,  0,   0,   0,   0   ]] #   400

startNew = False
for i,d in enumerate([25, 50, 100, 200, 400]):
    for j, Nh in enumerate([2,4,8,16,32,64,128,256,512,1024]):
        if Nh*d > 10**5:
            print('Too large: {}x{}'.format(d, Nh))
            continue
        if converged[i][j]:
            print('Converged: {}x{}'.format(d, Nh))
            continue

        R0 = int(math.ceil(.025*(Nh*d)**0.8))
        if startNew:
            argsList.append(( '--net', 'HebbNet', '--Nh', Nh, '--d', d, '--forceAnti', '--init', 'analytic', '--reparamLam', '--train', 'cur', '--R0', R0, '--noMultiRep', '--folder', 'vary_Nh_allBinInit/d_{}/'.format(d)))
        else:
            fname = 'HebbNet[{},{},1]_train=cur{}_incr=plus1_forceAnti_reparamLam.pkl'.format(d,Nh,R0)
            argsList.append(('--filename', fname, '--cont', '--train', 'cur', '--noMultiRep', '--folder', 'vary_Nh_allBinInit/d_{}/'.format(d)))

        if Nh*d >= 400*128:
            RAM.append(64)
        elif Nh*d >= 400*64:
            RAM.append(32)
        else:
            RAM.append(16)


#%%
#argsList = [
#            ('--net', 'nnLSTM', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 3,   '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'nnLSTM', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 6,   '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'nnLSTM', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 3,6, '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'nnLSTM', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 1,2,3,4,5,6,7,8,9, '--noMultiRep', '--folder', 'RNN'),
#
#            ('--net', 'VanillaRNN', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 3,   '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'VanillaRNN', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 6,   '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'VanillaRNN', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 3,6, '--noMultiRep', '--folder', 'RNN'),
#            ('--net', 'VanillaRNN', '--Nh', 625, '--d', 25,  '--train', 'inf', '--R0', 1,2,3,4,5,6,7,8,9, '--noMultiRep', '--folder', 'RNN'),
#            ]

#%%
#argsList = [
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.84', '--eta0', -0.2, '--freeze', '_lam', '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.97', '--eta0', -0.2, '--freeze', '_lam', '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.84', '--eta0', 0.1, '--freeze', '_lam', '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.97', '--eta0', 0.1, '--freeze', '_lam', '--folder', 'fixLam'),
#
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.84', '--eta0', -0.2, '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.97', '--eta0', -0.2, '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.84', '--eta0', 0.1, '--folder', 'fixLam'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--lam0', '0.97', '--eta0', 0.1, '--folder', 'fixLam'),
#        ]

#%%
#argsList = [
#        #Control: does GTP naturally give preference to Hebb or Anti? What decay/learning rates does it pick?
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.84, '--eta0', -0.1, '--forceAnti',  '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.97, '--eta0', -0.1, '--forceAnti',  '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.84, '--eta0',  0.1, '--forceHebb',  '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.97, '--eta0',  0.1, '--forceHebb',  '--folder', 'fixLamGTP'),
#
#        #is it possible to fix lam and eta to the same magnitudes and get same perf without GTP? with?
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep',                       '--lam0', 0.9, '--eta0', -0.07,                '--freeze', '_lam','_eta', '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep',                       '--lam0', 0.9, '--eta0',  0.07,                '--freeze', '_lam','_eta', '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.9, '--eta0', -0.07,                '--freeze', '_lam','_eta', '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.9, '--eta0',  0.07,                '--freeze', '_lam','_eta', '--folder', 'fixLamGTP'),
#
#        #fix lam. Does including GTP find the same eta for both Anti and Hebb?
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.84, '--eta0', -0.1, '--forceAnti', '--freeze', '_lam',  '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.97, '--eta0', -0.1, '--forceAnti', '--freeze', '_lam',  '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.84, '--eta0',  0.1, '--forceHebb', '--freeze', '_lam', '--folder', 'fixLamGTP'),
#        ('--net', 'HebbNet', '--Nh', 100, '--d', 100,  '--train', 'inf', '--R0', 5,  '--noMultiRep', '--groundTruthPlast', '--lam0', 0.97, '--eta0',  0.1, '--forceHebb', '--freeze', '_lam', '--folder', 'fixLamGTP'),
#        ]


#%% Recognition and classification

# useShortQueue = False
# RAM = [32]
# argsList = [
#        ('--net', 'HebbClassify', '--Nh', 50, '--d', 25,                                               '--sampleSpace', 'sample_space.pkl', '--train', 'inf', '--R0', 1, '--noMultiRep', '--folder', 'recogAndClass'),
#        ('--net', 'HebbClassify', '--Nh', 50, '--d', 25,                                               '--sampleSpace', 'sample_space.pkl', '--train', 'inf', '--R0', 5, '--noMultiRep', '--folder', 'recogAndClass'),
#        ('--net', 'HebbClassify', '--Nh', 50, '--d', 25,                                               '--sampleSpace', 'sample_space.pkl', '--train', 'inf', '--R0', 10,'--noMultiRep', '--folder', 'recogAndClass'),
#        ('--net', 'HebbClassify', '--Nh', 50, '--d', 25,                                               '--sampleSpace', 'sample_space.pkl', '--train', 'cur', '--R0', 1, '--noMultiRep', '--folder', 'recogAndClass'),
#        ( '--net', 'HebbClassify', '--Nh', 50, '--d', 25, '--w2init', 'scalar', '--b1init', 'scalar',  '--sampleSpace', 'sample_space.pkl', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'recogAndClass')
        #  ('--net', 'HebbClassify', '--Nh', 128, '--d', 25,                                               '--sampleSpace', int(3e4), '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'recogAndClass'),
        # ]

#%% Split synapses in HebbFF like the DetHebb analytics does
#argsList = [
##        #n=4, D=96
##        #Control: HebbFF with d=D or d=D+n (should be the same)
#        ( '--net', 'HebbNet',      '--Nh', 16, '--d', 96,                 '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbNet',      '--Nh', 16, '--d', 96,                 '--forceAnti', '--w1init', 'randn',      '--w2init', 'scalar',    '--b2init', 'scalar',   '--b1init', 'scalar',    '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbNet',      '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbNet',      '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'randn',      '--w2init', 'scalar',    '--b2init', 'scalar',   '--b1init', 'scalar',    '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
##
##        #Expt: SplitSyn, d=100 ==> D=d-n=96, three different W1 (randn should approach allBin, allBin should not change, allBin+gain should do as well as control)
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100, '--gain','w1', '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'randn',      '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar',    '--b2init', 'scalar',   '--b1init', 'scalar',    '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100, '--gain','w1', '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar',    '--b2init', 'scalar',   '--b1init', 'scalar',    '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ( '--net', 'HebbSplitSyn', '--Nh', 16, '--d', 100,                '--forceAnti', '--w1init', 'randn',      '--w2init', 'scalar',    '--b2init', 'scalar',   '--b1init', 'scalar',    '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSyn'),
#        ]

#%% Train HebbFF on output of ResNet
#argsList = [
#             #train from scratch, simple readout
##            ('--cont', '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 2, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#            ( '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--forceAnti', '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 2, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 1, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#
#             #train from scratch, weighted readout
##            ('--cont', '--net', 'HebbNet', '--Nh', 16, '--d', 50,  '--train', 'cur', '--R0', 2, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ('--cont', '--net', 'HebbNet', '--Nh', 16, '--d', 50,  '--train', 'cur', '--R0', 2, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#            ('--net', 'HebbNet', '--Nh', 16, '--d', 50,  '--forceAnti', '--train', 'cur', '--R0', 1, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#
#             #train on random, fine-tune on ResNet
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 2, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 50, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#
#            #train on random, freeze W1. Can we get it to perform? Compare to re-trained W1?
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--freeze', 'w1', '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--freeze', 'w1', '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50_normalize.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
##            ( '--load', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--train', 'cur', '--R0', 15, '--freeze', 'w1', '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18_d=50.pkl', '--noMultiRep', '--folder', 'ResNetOutput'),
#
#            #prepend extra layer to dim-reduce the input from 512 to d=50
##             ( '--net', 'HebbFeatureLayer', '--Nx', 512, '--d', 50, '--Nh', 16, '--b1init', 'scalar-10', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18.pkl', '--train', 'cur', '--R0', 1, '--noMultiRep', '--folder', 'ResNetOutput'),
##             ( '--net', 'HebbFeatureLayer', '--Nx', 512, '--d', 50, '--Nh', 16,                                              '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18.pkl', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'ResNetOutput'),
#              ( '--net', 'HebbFeatureLayer', '--Nx', 512, '--d', 50, '--Nh', 16, '--HebbFeatInit', 'HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '--freeze', 'w1', '--sampleSpace', 'BradyOliva2008_UniqueObjects_ResNet18.pkl', '--train', 'cur', '--R0', 1, '--noMultiRep', '--folder', 'ResNetOutput'),
#              ]

#%% Soft labels to see if distribution will stay the same width across R
#argsList = [
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 1, '--noMultiRep', '--softLabels', 0.1, '--folder', 'softLabels'),
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 1, '--noMultiRep', '--softLabels', 0.01, '--folder', 'softLabels'),
#        ('--net', 'HebbNet', '--Nh', 25, '--d', 25, '--w2init', 'scalar', '--b1init', 'scalar', '--train', 'cur', '--R0', 1, '--noMultiRep', '--softLabels', 0.001, '--folder', 'softLabels'),
#        ]


#%% Empirically get scaling for allBinInit and SplitSyn
#useShortQueue = False
#argsList = []
#d = 100
#for Nh in [2,4,8,16,32,64,128,256,512]:
#    argsList.append(( '--net', 'HebbNet',      '--d', d, '--Nh', Nh, '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'allBinInitScaling'))
#    argsList.append(( '--net', 'HebbSplitSyn', '--d', d, '--Nh', Nh, '--forceAnti', '--w1init', 'all_binary', '--w2init', 'scalar-10', '--b2init', 'scalar10', '--b1init', 'scalar-10', '--train', 'cur', '--R0', 2, '--noMultiRep', '--folder', 'splitSynScaling'))
#    if Nh*d >= 400*64:
#        RAM.append(32)
#        RAM.append(32)
#    else:
#        RAM.append(16)
#        RAM.append(16)

#%% Control for ConvHebb - train on uncorrelated but non-binary random vectors
#argsList = [
#        ( '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--forceAnti', '--w2init', 'scalar-10', '--b1init', 'scalar-10', '--b2init', 'scalar10', '--train', 'cur', '--R0', 1, '--xDataVals', 'uniform', '--noMultiRep', '--folder', 'UncorrNonBin'),
#        ( '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--forceAnti', '--w2init', 'scalar-10', '--b1init', 'scalar-10', '--b2init', 'scalar10', '--train', 'cur', '--R0', 1, '--xDataVals', 'normal', '--noMultiRep', '--folder', 'UncorrNonBin'),
#        ( '--net', 'HebbNet', '--Nh', 16, '--d', 50, '--forceAnti', '--w2init', 'scalar-10', '--b1init', 'scalar-10', '--b2init', 'scalar10', '--train', 'cur', '--R0', 1, '--xDataVals', '01', '--noMultiRep', '--folder', 'UncorrNonBin'),
#        ]

#%%
if len(RAM) == 0:
    RAM = [16 for _ in range(len(argsList))]

for _ in range(NUM_TRIALS):
    print('[{}] Looping over args...'.format(sys.platform))
    for i,args in enumerate(argsList):
        argstr = ' '.join([str(a) for a in args])
        gpu = True if '--gpu' in args else False
        folder = args[args.index('--folder')+1] if '--folder' in args else None
        sbatch(script, argstr, output=None, ram=RAM[i], cpus=1 if gpu else 2, gpu=gpu, folder=folder, short=useShortQueue)
#    time.sleep(10) #time for the logfile to be created to allow auto-renaming of subsequent ones if necessary TODO: this hack only works if logs are created (i.e. jobs start) w/in 10 secs of submission
