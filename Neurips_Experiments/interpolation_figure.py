import seaborn as sns
import matplotlib.pyplot as plt
from experiment_functions import *

sns.set_theme()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

n = 200
k = 10
beta = 2
r = 500

G = sbm(n,k,0.25,0.05)
Cl = [list(range((n//k)*i,(n//k)*(i+1))) for i in range(k)]
Clb = [[i] for i in range(n)]

h = homophily_effects(G)
fY = pom_ugander_yin(G,h,2)

p = 0.1
P = [0,p/2,p]
q = 0.5
Q = [0,q/2,q]

l0 = lambda P: lambda x: (x-P[1])*(x-P[2])/((P[0]-P[1])*(P[0]-P[2]))
l1 = lambda P: lambda x: (x-P[0])*(x-P[2])/((P[1]-P[0])*(P[1]-P[2]))
l2 = lambda P: lambda x: (x-P[0])*(x-P[1])/((P[2]-P[0])*(P[2]-P[1]))
f = lambda Y: lambda P: lambda x : Y[0]*l0(P)(x) + Y[1]*l1(P)(x) + Y[2]*l2(P)(x)



fig,ax = plt.subplots(1,2,sharey=True,sharex=True)
ax[0].yaxis.label.set(rotation='horizontal',ha="left")
ax[0].set_ylabel("$\\widehat{F}(p)$")
ax[0].set_title("Bern$([0,0.05,0.1])$ Rollout")
ax[1].set_title("Bern$(\\Pi,0.1,[0,0.25,0.5])$ Rollout")

plt.setp(ax,xlim=(0,1))
plt.setp(ax,xlabel="$p$")

x = np.linspace(0,1,10000)

Z,_ = staggered_rollout_two_stage(n,Cl,p,P,r)
Y = 1/n*np.sum(fY(Z),axis=2)

for i in range(r):
    f_hat = f(Y[:,i])(P)
    ax[0].plot(x,f_hat(x),color="tab:blue",alpha=0.05)


Z,_ = staggered_rollout_two_stage(n,Cl,p,Q,r)
Y = 1/n*np.sum(fY(Z),axis=2)

for i in range(r):
    f_hat = f(Y[:,i])(Q)
    ax[1].plot(x,q/p*(f_hat(x)-f_hat(0))+f_hat(0),color="tab:blue",alpha=0.05)

true_y = [1/n*np.sum(fY(np.ones(n)*P[0])),1/n*np.sum(fY(np.ones(n)*P[1])),1/n*np.sum(fY(np.ones(n)*P[2]))]
ax[0].plot(x,f(true_y)(P)(x),"k",linewidth=2)
ax[0].axvline(x=0.1,color="k",linewidth=0.5)
ax[1].plot(x,f(true_y)(P)(x),"k",linewidth=2)
ax[1].axvline(x=0.1,color="k",linewidth=0.5)

plt.show()



