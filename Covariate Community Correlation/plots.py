# plots
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
pd.options.mode.chained_assignment = None  # default='warn'

def main(beta=1, B=0.05, p=1): 
    graph = "SBM"
    n = 1000
    nc = 50
    p_in = 10/(n/nc) 
    p_out = 0

    Bstr = str(np.round(B,3)).replace('.','')
    pstr = str(np.round(p,3)).replace('.','')
    
    experiment = 'correlation'
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + pstr # naming convention
    x_label = [fixed + '_B' + Bstr + '_' + experiment]
    x_var = ['Phi']
    x_plot = ['$\phi$']
    title = ['$\\beta={}, n={}, n_c={}, B={}, p={}$'.format(beta, n, nc, B, np.round(p,3))]
    for ind in range(len(x_var)):
        plot(graph,x_var[ind],x_label[ind],beta,x_plot[ind],title[ind])

def plot(graph,x_var,x_label,model,x_plot,title):
    load_path = 'output/' + 'deg' + str(model) + '/'
    save_path = 'plots/' + 'deg' + str(model) + '/'
    deg_str = '_deg' + str(model)
    
    # All possible estiamtors: ['PI-$n$($p$)', 'PI-$n$($B$)', 'HT, 'PI-$\mathcal{U}$($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    # All possible designs: ['Cluster', 'Bernoulli'] - note that this only matters for 'LS-Prop', 'LS-Num','DM', and 'DM($0.75$)'
    estimators = ['PI-$n$($p$)', 'PI-$n$($B$)', 'PI-$\mathcal{U}$($p$)', 'HT'] 
    estimators_str = '_nUHT' # other options: 'n', 'U', 'all' -- basically, which estimators show up in the plots

    experiment = x_label
    print(experiment+deg_str+'_'+x_var+estimators_str)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data' + deg_str+ '.csv')
    newData = df.loc[df['Estimator'].isin(estimators)]

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, errorbar='sd', legend='brief', markers=True)

    #ax.set_xlim(0,0.001)
    ax.set_xlabel(x_plot, fontsize = 18)
    #ax.set_ylim(-0.75,0.75)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=18)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path+graph+'_'+x_var+deg_str+experiment+estimators_str+'.png')
    plt.close()

    # Create and save MSE plots
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', data=newData, legend='brief', markers=True)

    ax2.set_xlabel(x_plot, fontsize = 18)
    ax2.set_ylabel("MSE", fontsize = 18)
    ax2.set_title(title, fontsize=20)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path+graph+'_'+x_var+deg_str+experiment+estimators_str+'_MSE.png')
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    beta = [1,2,3]
    B = [0.06, 0.5, 0.5]
    probs = [[0.06, 0.25, 1/3, 2/3, 1],     # K in [50, 12, 9, 6, 3]
            [0.5, 0.625, 25/33, 25/29, 1],  # K in [50, 40, 33, 29, 25]
            [0.5, 0.625, 25/33, 25/29, 1]]
for b in range(len(beta)):
    print('Plotting degree: {}'.format(b+1))
    for p in probs[b]:
        print()
        main(beta[b], B[b], p)
