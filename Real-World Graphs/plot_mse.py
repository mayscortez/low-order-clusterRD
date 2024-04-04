import seaborn.objects as so
import pandas as pd
import pickle
import sys

# basic mse:            col = nc              row = beta
# compare clustering:   col = clustering      row = nc
# vary p:               col = p               row = nc 

def main():
    args = sys.argv[1:]
    input_file = args[0]    # location of experiment data pkl file
    output_file = args[1]   # location to save plot image

    group_vars = ['q']      # columns to group by when manipulating data frame
    col_var = None
    row_var = None
    if len(args) > 2:
        col_var = args[2]   # variable facet plots by column
        group_vars.append(col_var)
    if len(args) > 3:
        row_var = args[3]   # variable to facet plots by row
        group_vars.append(row_var)

    data_file = open(input_file, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    df = pd.DataFrame(data)

    def aggregate_stats(x):
        d = {}
        d['bias^2'] = x['tte_hat'].mean()**2
        d['variance'] = x['tte_hat'].var()
        return pd.Series(d, index=['bias^2', 'variance'])

    df = (
        df.groupby(group_vars+['est'])                     # consider each set of parameters separately
        .apply(aggregate_stats,include_groups=False)                     # calculate bias and variance
        .reset_index()
        .pivot(index=group_vars, columns=['est'])                 # columns for overall / conditional variance
        .pipe(lambda s: s.set_axis(s.columns.map('_'.join), axis=1))     # fix column naming
        .reset_index()
        .drop(['bias^2_exp'],axis=1)                                     # remove redundant bias column
        .rename(columns={'bias^2_real': 'Bias$^2$', 
                        'variance_exp': 'Sampling Variance', 
                        'variance_real': 'Extrapolation Variance'})
    )

    df['Extrapolation Variance'] = (df['Extrapolation Variance'] - df['Sampling Variance']) #* df['q']**2

    df = df.melt(                                                       # rows for each stat to allow for stack plot 
        id_vars=group_vars, 
        value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
        var_name='stat', 
        value_name='value')

    p = (
        so.Plot(df, x='q', y='value', color='stat')
        .facet(row=row_var, col=col_var)
        .add(so.Area(), so.Stack())
        .layout(size=(10,8))
    )

    p.show()
    p.save(output_file)

if __name__ == '__main__':
    main()