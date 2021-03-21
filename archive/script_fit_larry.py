from scipy.special import erfc, erfcinv
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
from plotting import get_recog_positive_rates
from data import generate_recog_data
from net_utils import load_from_file


#%% Human data
testR = np.array([1,2,4,8,16,32,64,128,256,512,1024])
Ptp = np.array([1., .99, .99, .99, .98, .96, .96, .92, .88, .83, .79])
Pfp = 0.013*np.ones(len(testR))
fracNov = 7./8


#%% HebbFF data
multiRep = True

fname, label = ('HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl', '$R_{train}=[1-14]$')
#fname, label = ('HebbNet[25,25,1]_train=cur2_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar_train=inf100.pkl', '$R_{train}=[1-14,100]$')
net = load_from_file(fname)
d,Nh = net.w1.shape
gen_data = lambda R: generate_recog_data(T=max(R*20, 1000), d=d, R=R, P=0.5, interleave=True, multiRep=multiRep)
testR, testAcc, Ptp, Pfp = get_recog_positive_rates(net, gen_data, upToR=10000 if label.find('100')>0 else 1000)
testR, testAcc, Ptp, Pfp = np.array(testR), np.array(testAcc), np.array(Ptp), np.array(Pfp)

data = generate_recog_data(T=20000, d=d, R=20, P=0.5, interleave=True, multiRep=multiRep)
fracNov = 1 - data.tensors[1].sum().item()/len(data)


#%% Fit acc, fix Pfp
def acc_fn(R, alpha, D, n):
    N = 2**n
    gamma = 1 - (alpha**2 * fracNov)/(2*D*N)
    Pfp = 0.01 #= (1./2)*erfc( alpha*(n+b)/np.sqrt(2) )
    b = erfcinv(2*Pfp)*np.sqrt(2)/a - n
    Ptp = (1./2)*erfc( alpha*(n-gamma**(R-1)+b)/np.sqrt(2.) )
    acc = (1-fracNov)*Ptp + fracNov*(1-Pfp)
    return acc


fig,ax = plt.subplots()
for _ in range(10):
    a0 = 1+np.random.rand()*20 # 7.67
    D0 = np.random.randint(20,500) #50
    n0 = np.random.randint(2,10) #8
    popt, pcov = curve_fit(acc_fn, xdata=Rtest, ydata=acc, p0=[a0,D0,n0], bounds=([1,2,2],np.inf))
        
    a,D,n = popt
    N = 2**n
    
    accFit = acc_fn(Rtest, *popt) 
    ax.semilogx(Rtest, accFit, label='$\\alpha$={:.3f} D={:.1f} N={:.1f} (DN={:.1f})'.format(a,D,N,D*N))
ax.semilogx(Rtest, acc, label='human')
ax.legend()
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('Accuracy')


#%% Fit both Pfp and Ptp
def fp_tp_fn(R, a,b,D,n ):
    R = R[:len(R)/2]
    N = 2**n
    g = 1 - (a**2 * fracNov)/(2*D*N)
    Pfp = (1./2)*erfc( a*(n+b)/np.sqrt(2) )
    Pfp = Pfp*np.ones(len(R))
    Ptp = (1./2)*erfc( a*(n-g**(R-1)+b)/np.sqrt(2.) )            
    return np.concatenate([Pfp,Ptp])
    

def fit_fp_tp(R, Pfp, Ptp):                   
    a0 = 1+np.random.rand()*20 # 7.67
    D0 = np.random.randint(20,500) #50
    n0 = np.random.randint(2,10) #8
    b0 = 2.326/a0 - n0    
    popt, pcov = curve_fit(fp_tp_fn, xdata=np.concatenate([R,R]), ydata=np.concatenate([Pfp,Ptp]), p0=[a0,b0,D0,n0], bounds=([1,-np.inf,2,2],np.inf))
    a,b,D,n = popt
    return a,b,D,n


def fit_fp_tp_fixDN(R, Pfp, Ptp, D=25, n=np.log2(25)):  
    f = lambda R,a,b: fp_tp_fn(R, a,b,D,n)                
    a0 = 1+np.random.rand()*20 # 7.67
    b0 = 2.326/a0 - n    
    popt, pcov = curve_fit(f, xdata=np.concatenate([R,R]), ydata=np.concatenate([Pfp,Ptp]), p0=[a0,b0], bounds=([1,-np.inf],np.inf))
    a,b = popt
    return a,b


fig,ax = plt.subplots()
for _ in range(1):
    #fit
    a,b,D,n = fit_fp_tp(testR, Pfp, Ptp)
#    D=25
#    n=np.log2(25)
#    a,b = fit_fp_tp_fixDN(testR, Pfp, Ptp, D=D, n=n)
    N = 2**n
    g = 1 - a**2*fracNov/(2*D*N)
    #plot fitted params
    fitR = testR #np.logspace(0,4)
    P = fp_tp_fn(np.concatenate([fitR,fitR]), a,b,D,n)
    lineLabel='$\\alpha$={:.2f}, b={:.2f}, D={:.1f} N={:.1f} (DN={:.1f}, $\gamma={:.2f}$)'.format(a,b,D,N,D*N,g)
    line = ax.semilogx(fitR, P[:len(P)/2], ls='--')[0]
    ax.semilogx(fitR, P[len(P)/2:], color=line.get_color(), label=lineLabel)

#plot data
lineLabel='{} $\lambda=${:.2f}, $\eta=${:.2f}, D={}, N={}'.format(label, net.lam.item(), net.eta.item(), d, Nh)
#label = 'Human'
line = ax.semilogx(testR, Pfp, ls='--')[0]
ax.semilogx(testR, Ptp, color=line.get_color(), label=lineLabel)
ax.legend()
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('True/false positive rate')

fig.set_size_inches(6,3)
fig.tight_layout()

#%% Fit Ptp, fix Pfp
Pfp_fit = Pfp.mean()
def fp_tp_fn(R, a,b,D,n):
    N = 2**n
    g = 1 - (a**2 * fracNov)/(2*D*N)
    b = erfcinv(2*Pfp_fit)*np.sqrt(2)/a - n
    Ptp = (1./2)*erfc( a*(n-g**(R-1)+b)/np.sqrt(2.) )            
    return Ptp
    

def fit_tp_fixDN(R, Ptp, D=25, n=np.log2(25)):  
    f = lambda R,a,b: fp_tp_fn(R, a,b,D,n)                
    a0 = 1+np.random.rand()*20 # 7.67
    b0 = 2.326/a0 - n    
    popt, pcov = curve_fit(f, xdata=R, ydata=Ptp, p0=[a0,b0], bounds=([1,-np.inf],np.inf))
    a,b = popt
    return a,b


fig,ax = plt.subplots()
for _ in range(1):
    #fit
    D=25
    n=np.log2(25)
    a,b = fit_tp_fixDN(testR, Ptp, D=D, n=n)
    
    #plot fitted params
    N = 2**n
    g = 1 - a**2*fracNov/(2*D*N)
    plotR = testR #np.logspace(0,4)
    Ptp_fit = fp_tp_fn(plotR, a,b,D,n)
    lineLabel='$\\alpha$={:.2f}, b={:.2f}, D={:.1f} N={:.1f} (DN={:.1f}, $\gamma={:.2f}$)'.format(a,b,D,N,D*N,g)
    line = ax.semilogx(plotR, Pfp_fit*np.ones(len(plotR)), ls='--')[0]
    ax.semilogx(plotR, Ptp_fit, color=line.get_color(), label=lineLabel)

#plot data
lineLabel='{} $\lambda=${:.2f}, $\eta=${:.2f}, D={}, N={}'.format(label, net.lam.item(), net.eta.item(), d, Nh)
#label = 'Human'
line = ax.semilogx(testR, Pfp, ls='--')[0]
ax.semilogx(testR, Ptp, color=line.get_color(), label=lineLabel)
ax.legend()
ax.set_xlabel('$R_{test}$')
ax.set_ylabel('True/false positive rate')

fig.set_size_inches(6,3)
fig.tight_layout()