import seaborn.objects as so
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np

file = open('../Experiments/fit_poly3.pkl', 'rb')
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
df = df.drop(['Sampling Variance'], axis=1)
df = df.melt(                                                       # rows for each stat to allow for stack plot 
    id_vars=['q'], 
    #value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
    value_vars=['Bias$^2$','Extrapolation Variance'],
    var_name='stat', 
    value_name='value')

plot = (
    so.Plot(df, x='q', y='value', color='stat')
    .add(so.Area(), so.Stack())
    .layout(size=(6,4))
)

f, ax = plt.subplots()

n = 16716
nc = 100
p = 0.2

a = sum([L[frozenset([i])]**2 for i in range(nc)])/n**2

b1 = sum([x**2 for x in Lj])
print("b1",b1)
b2 = sum([x**2 for _,x in Ljjp.items()])
print("b2",b2)

b3 = 0
for C in Cl:
    for j in C:
        for jp in C:
            if jp != j and frozenset([j,jp]) in Ljjp: 
                b3 += Lj[j] * Ljjp[frozenset([j,jp])]
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
c3 = 5*b1-8*b2-6*b3+2*b4
print("c3",c3)
c4 = -1*b1+5*b2+2*b3-3*b4
print("c4",c4)
c5 = b4-b2
print("c5",c5)

x = np.linspace(0.2,1,1000)
#y = a*x/0.2 - a
y1 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2 + c3 + x * c4 + x**2 * c5) #+ y
#y2 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2 + c3 + x * c4) #+ y
#y3 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2 + c3) #+ y
y4 = 1/(p*n**2) * (1/x**2 * c1 + 1/x * c2) #+ y

#ax.plot(x,y,'k')
ax.plot(x,y1,'r')
ax.plot(x,y4,'k')
plot.on(ax).show()

# p = (
#     so.Plot(df, x='q', y='Extrapolation Variance')#, color='stat')
#     .facet(row='beta', col='nc')
#     .add(so.Line())
#     .layout(size=(10,8))
# )

#p.show()