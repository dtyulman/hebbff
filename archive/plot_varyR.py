import joblib
import matplotlib.pyplot as plt

from plastic import PlasticNet
from neural_net_utils import plot_train_perf, plot_W, plot_B
from data_utils import generate_recognition_data, recog_chance

from torch.nn import functional as F
#%%
d = 25           #length of input vector
P = 0.5           #probability of repeat

#%% Plot training performance for each network
fnameTemplate = 'antiHebb_recog_R={}_std' #UPDATE       
#fnameTemplate = 'PlasticNet_R={}_eta=-0.5_lam=0.5' #UPDATE       
    
for R in [5]: #range(1,11)+[15,30]:  #UPDATE  
    testData = generate_recognition_data(T=10000, d=d, R=R, P=P, interleave=True, astensor=True)
    
#    net = joblib.load(fnameTemplate.format(R)+'.pkl') 
    plot_train_perf(net, recog_chance(testData), 
                    title = 'R={} ({:.4}%)'.format(R, net.accuracy(testData)*100))    
#    plt.savefig(fnameTemplate.format(R)+'.png') 

#%% Test each network on R from 0 to 200
from joblib import Parallel, delayed
R_test_list = range(1,15)#+range(70,200,10)#+range(200,301,50) #UPDATE (decrease range for small R)
R_net_list = [5] #UPDATE  
test_acc = {} #UPDATE (comment out if appending nets)
test_acc.update( {R:[] for R in R_net_list} ) #UPDATE (comment out if appending Rs)

for R_test in R_test_list:
#    testData = generate_recognition_data(T=10000, d=d, R=R_test, P=P, interleave=True, astensor=True)  
    testData = generate_delayed_recall_data(T=10000, d=d, R=R_test)
    def load_and_test(R_net):
        print('testR={}, netR={}'.format(R_test, R_net))
#        net = joblib.load(fnameTemplate.format(R_net)+'.pkl')
        acc = net.accuracy(testData).item()
        test_acc[R_net].append( acc )
    Parallel(n_jobs=1)(delayed(load_and_test)(R_net) for R_net in R_net_list) 
    

#%% Plot generalization results   
plt.figure()
for R_net in sorted(test_acc.keys()):
    plt.plot(R_test_list, test_acc[R_net], '-', label='R={}'.format(R_net))
    plt.xlabel('R_test')
    plt.ylabel('Generalization accuracy')
plt.legend()
plt.gcf().set_size_inches(13,5)
plt.tight_layout()

#plt.savefig('{}_generalization.png'.format(fnameTemplate).format(R))   #UPDATE
#joblib.dump(test_acc, '{}_generalization.pkl'.format(fnameTemplate).format(R)) #UPDATE

#%%
for R in range(1,11)+[15,30]:
    net = joblib.load(fnameTemplate.format(R)+'.pkl') 
    plot_W([net.w1.detach().numpy(), net.w2.detach().numpy()])
    plt.gcf().set_size_inches(3.5,5)
    plt.savefig('W_R={}.png'.format(R))   
    

    plot_B([net.b1.detach().numpy(), net.b2.detach().numpy()])
    plt.gcf().set_size_inches(1.75,5)
    plt.savefig('B_R={}.png'.format(R))  


