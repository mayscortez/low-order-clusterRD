# plots
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def main():
    graph = "SBM" 
    n = 1000
    nc = 50
    B = 0.25
    Bstr = str(B).replace('.','')
    p_in = 10/(n/nc) 
    p_out = 0
    p = .25 
    K = int(np.floor(B * nc / p))
    q_or_K_st = '_K' + str(K)
    experiment = 'correlation'
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + str(p).replace('.','') # naming convention
    x_label = [fixed + '_' + experiment]
    x_var = ['Phi']
    x_plot = ['$\phi$']
    beta = [1,2,3]
    for b in beta:
        title = ['$\\beta={}, n={}, n_c={}, E[d_i]=10, p={}$'.format(b, n, nc, p)]
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],b)


def plot(graph,x_var,x_label,model,x_plot,title,beta=1):
    load_path = 'output/' + 'deg' + str(beta) + '/'
    save_path = 'plots/' + 'deg' + str(beta) + '/'
    deg_str = '_deg' + str(beta)
    
    #estimators = ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    #estimators = ['PI-$n$($p$)'] 
    #estimators = ['PI-$\mathcal{U}$($p$)'] 
    estimators = ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)'] 
    estimators_str = '_nU' # other options: 'n', 'U', 'all' -- basically, which estimators show up in the plots

    experiment = x_label
    print(experiment+deg_str+'_'+x_var+estimators_str)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data' + deg_str+ '.csv')

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    newData = df.loc[df['Estimator'].isin(estimators)]

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, errorbar='sd', legend='brief', markers=True)
    #sns.lineplot(x=x_var, y='Bias', data=df, errorbar='sd', legend='brief', markers=True)

    #ax.set_xlim(0,0.001)
    ax.set_xlabel(x_plot, fontsize = 18)
    #ax.set_ylim(-0.75,0.75)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=18)
    #plt.xticks([0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path+graph+'_'+x_var+deg_str+experiment+estimators_str+'.pdf')
    plt.close()

    #TODO: Create and save MSE plots


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
