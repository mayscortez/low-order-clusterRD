import seaborn.objects as so
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np

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

df = df.melt(                                                       # rows for each stat to allow for stack plot 
    id_vars=['q'], 
    value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
    var_name='stat', 
    value_name='value')

p = (
    so.Plot(df, x='q', y='value', color='stat')
    .add(so.Area(), so.Stack())
    .layout(size=(6,4))
)

f, ax = plt.subplots()

n = 16716
nc = 100

a = sum([L[frozenset([i])]**2 for i in range(nc)])/n**2

c1 = sum([x**2 for x in Lj])
c2 = sum([x**2 for _,x in Ljjp.items()])
c3 = 0
for C in Cl:
    for j in C:
        for jp in C:
            if jp == j: continue
            c3 += Lj[j] * Ljjp[frozenset([j,jp])]
c4 = 0
for C in Cl:
    for j in C:
        c4 += sum([Ljjp[frozenset([j,jp])] for jp in C if jp != j]) 

print(c1)
print(c2)
print(c3)
print(c4)


x = np.linspace(0.2,1,1000)
y = a*x/0.2 - a
y2 = (1-x)*(2-x)/(x**2*0.2*n**2) * ((2-x)*c1 + x*(2-x)*c2 + 2*x*c3 + x**2*c4) + y
ax.plot(x,y,'k')
ax.plot(x,y2,'r')

p.on(ax).show()

# p = (
#     so.Plot(df, x='q', y='Extrapolation Variance')#, color='stat')
#     .facet(row='beta', col='nc')
#     .add(so.Line())
#     .layout(size=(10,8))
# )

#p.show()