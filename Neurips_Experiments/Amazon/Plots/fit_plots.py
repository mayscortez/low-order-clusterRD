import seaborn.objects as so
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations

file = open('../Experiments/fit_poly.pkl', 'rb')
data,L,Lj,Ljjp,Cl = pickle.load(file)
file.close()

df = pd.DataFrame(data)

def aggregate_stats(x):
    d = {}
    d['bias^2'] = x['tte_hat'].mean()**2
    d['variance'] = x['tte_hat'].var()
    return pd.Series(d, index=['bias^2', 'variance'])

df = df[df['nc'] == 100].drop(['nc'],axis=1) 

df = (
    df.groupby(['q','est'])                              # consider each set of parameters separately
    .apply(aggregate_stats,include_groups=False)                     # calculate bias and variance
    .reset_index()
    .pivot(index=['q'], columns=['est'])                 # columns for overall / conditional variance
    .pipe(lambda s: s.set_axis(s.columns.map('_'.join), axis=1))     # fix column naming
    .reset_index()
    .drop(['bias^2_exp'],axis=1)                                     # remove redundant bias column
    .rename(columns={'bias^2_real': 'Bias$^2$', 
                     'variance_exp': 'Sampling Variance', 
                     'variance_real': 'Extrapolation Variance'})
)

df['Extrapolation Variance'] = (df['Extrapolation Variance'] - df['Sampling Variance']) #* df['q']**2
#df = df.drop(['Sampling Variance'], axis=1)
df = df.melt(                                             # rows for each stat to allow for stack plot 
    id_vars=['q'], 
    value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
    #value_vars=['Bias$^2$','Extrapolation Variance'],
    var_name='stat', 
    value_name='value')

plot = (
    so.Plot(df, x='q', y='value', color='stat')
    .add(so.Area(), so.Stack())
    .layout(size=(6,4))
)

f, ax = plt.subplots()

n = 14436
nc = 100
p = 0.2

e1 = sum([L[frozenset([i])]**2 for i in range(nc)])
e2 = 2*sum([L[frozenset([i,ip])]**2 for (i,ip) in combinations(list(range(nc)),2)])
e3 = sum([(L[frozenset([i])] + L[frozenset([ip])]) * L[frozenset([i,ip])] for (i,ip) in combinations(list(range(nc)),2)])
e4 = 2*sum([sum([L[frozenset([i,ip])] for ip in range(nc) if ip != i])**2 for i in range(nc)])

a1 = e1
a2 = -e1 + e2 + 2*e3
a3 = -2*e2 - 2*e3 + e4
a4 = -e4

b1 = sum([x**2 for x in Lj])
print("b1",b1)
b2 = sum([x**2 for _,x in Ljjp.items()])
print("b2",b2)

b3 = 0
for C in Cl:
    for j in C:
        b3 += Lj[j] * sum([Ljjp[frozenset([j,jp])] for jp in C if jp != j and frozenset([j,jp]) in Ljjp])
print("b3",b3)

b4 = 0
for C in Cl:
    for j in C:
        b4 += sum([Ljjp[frozenset([j,jp])] for jp in C if jp != j and frozenset([j,jp]) in Ljjp])**2
print("b4",b4)

c1 = 4*b1
print("c1",c1)
c2 = -8*b1+4*b2+4*b3
print("c2",c2)
c2p = -8*b1+4*b3
print("c2p",c2p)
c3 = 5*b1-8*b2-6*b3+2*b4
print("c3",c3)
c4 = -1*b1+5*b2+2*b3-3*b4
print("c4",c4)
c5 = b4-b2
print("c5",c5)

d = 0
for (i,ip) in combinations(list(range(nc)),2):
    d += L[frozenset([i,ip])]
print("d",d)

x = np.linspace(p,1,1000)
y = ((p/x - 1) / n * d)**2
#y1 = a1*x + a2 + y
y1 = 1/n**2*(x/p * a1 + a2 + p/x * a3 + p**2/x**2 * a4) + y
#y1 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2 + c3 + x * c4 + x**2 * c5) + y
#y2 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2) + y
y3 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2p) + y1

ax.plot(x,y,'k')
ax.plot(x,y1,'r')
#ax.plot(x,y2,'b')
ax.plot(x,y3,'b')
plot.on(ax).show()
f.savefig("fit_plot.png")