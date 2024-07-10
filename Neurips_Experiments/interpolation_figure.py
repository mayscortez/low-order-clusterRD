import matplotlib.pyplot as plt
from experiment_functions import *

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

n = 200
k = 10
beta = 2
r = 200

G = sbm(n,k,0.25,0.05)
Cl = [list(range((n//k)*i,(n//k)*(i+1))) for i in range(k)]
Clb = [[i] for i in range(n)]

h = homophily_effects(G)
fY = pom_ugander_yin(G,h,2)

p = 0.2
P = [0,p/2,p]
q = 0.5
Q = [0,q/2,q]

l0 = lambda P: lambda x: (x-P[1])*(x-P[2])/((P[0]-P[1])*(P[0]-P[2]))
l1 = lambda P: lambda x: (x-P[0])*(x-P[2])/((P[1]-P[0])*(P[1]-P[2]))
l2 = lambda P: lambda x: (x-P[0])*(x-P[1])/((P[2]-P[0])*(P[2]-P[1]))
f = lambda Y: lambda P: lambda x : Y[0]*l0(P)(x) + Y[1]*l1(P)(x) + Y[2]*l2(P)(x)


fig,ax = plt.subplots(1,2,sharey=True,sharex=True)
ax[0].set_title("Bern$(2,0.2)$ Rollout",fontsize=16)
ax[1].set_title("Bern$(2,\\Pi,0.2,0.5)$ Two-Stage Rollout",fontsize=16)
ax[1].yaxis.set_tick_params(which='both', labelleft=True)

plt.setp(ax,xlim=(0,1))
plt.setp(ax,ylim=(-1,7))
plt.setp(ax,xlabel="$x$")

x = np.linspace(0,1,10000)

Z,_ = staggered_rollout_two_stage(n,Cl,p,P,r)
Y = 1/n*np.sum(fY(Z),axis=2)

for i in range(r):
    f_hat = f(Y[:,i])(P)
    ax[0].plot(x,f_hat(x),color="tab:blue",alpha=0.1)


Z,_ = staggered_rollout_two_stage(n,Cl,p,Q,r)
Y = 1/n*np.sum(fY(Z),axis=2)

for i in range(r):
    f_hat = f(Y[:,i])(Q)
    ax[1].plot(x,q/p*(f_hat(x)-f_hat(0))+f_hat(0),color="tab:blue",alpha=0.1)

true_y = [1/n*np.sum(fY(np.ones(n)*P[0])),1/n*np.sum(fY(np.ones(n)*P[1])),1/n*np.sum(fY(np.ones(n)*P[2]))]

for axis in ax:
    axis.set_ylabel("$\\widehat{F}(x)$", labelpad=20, rotation='horizontal', fontsize=16)
   #axis.yaxis.label.set(rotation='horizontal',fontsize=14,labelpad=10)
    axis.xaxis.label.set(fontsize=16)
    axis.plot(x,f(true_y)(P)(x),"k",linewidth=2)
    axis.axvline(x=0.1,color="k",linewidth=0.5,linestyle='--')
    axis.axvline(x=0.2,color="k",linewidth=0.5,linestyle='--')
    axis.text(0.01,6,'$t=0$',fontsize=14)
    axis.text(0.11,6,'$t=1$',fontsize=14)
    axis.text(0.21,6,'$t=2$',fontsize=14)

fig.subplots_adjust(bottom=0.25)
plt.show()