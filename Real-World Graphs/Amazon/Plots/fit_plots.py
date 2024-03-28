import seaborn.objects as so
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np

file = open('../Experiments/fit_poly.pkl', 'rb')
data,L = pickle.load(file)
file.close()

L = L[100]

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

a1 = sum([L[frozenset([i])]**2 for i in range(nc)])/n**2/0.2
print(a1)

a0 = -0.2*a1
for i in range(nc):
    for j in range(nc):
        if j == i: continue
        a0 += (2*L[frozenset([i])] + L[frozenset([i,j])]) * L[frozenset([i,j])]/n**2
print(a0)

x = np.linspace(0.2,1,1000)
y = a1*x + a0
ax.plot(x,y,'k')

p.on(ax).show()

# p = (
#     so.Plot(df, x='q', y='Extrapolation Variance')#, color='stat')
#     .facet(row='beta', col='nc')
#     .add(so.Line())
#     .layout(size=(10,8))
# )

#p.show()