import seaborn.objects as so
import pandas as pd
import pickle

file = open('../Experiments/compare_clusterings_data.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

def aggregate_stats(x):
    d = {}
    d['bias^2'] = x['tte_hat'].mean()**2
    d['variance'] = x['tte_hat'].var()
    return pd.Series(d, index=['bias^2', 'variance'])

df = (
    df.groupby(['q','nc','clustering','est'])                        # consider each set of parameters separately
    .apply(aggregate_stats,include_groups=False)                     # calculate bias and variance
    .reset_index()
    .pivot(index=['q','nc','clustering'], columns=['est'])           # columns for overall / conditional variance
    .pipe(lambda s: s.set_axis(s.columns.map('_'.join), axis=1))     # fix column naming
    .reset_index()
    .drop(['bias^2_exp'],axis=1)                                     # remove redundant bias column
    .rename(columns={'bias^2_real': 'Bias$^2$', 
                     'variance_exp': 'Sampling Variance', 
                     'variance_real': 'Extrapolation Variance'})
)

df['Extrapolation Variance'] = (df['Extrapolation Variance'] - df['Sampling Variance']) #* df['q']**2

df = df.melt(                                                       # rows for each stat to allow for stack plot 
    id_vars=['q','nc','clustering'], 
    value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
    var_name='stat', 
    value_name='value')

p = (
    so.Plot(df, x='q', y='value', color='stat')
    .facet(row='nc', col='clustering')
    .add(so.Area(), so.Stack())
    .layout(size=(6,8))
)

p.show()
p.save("mse_compare_clusterings.png")