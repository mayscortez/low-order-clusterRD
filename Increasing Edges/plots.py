from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def main():
    graph = "SBM" 
    n = 1000
    nc = 50
    q = 0.06 
    B = 0.06
    K = int(np.floor(q*nc))
    p = B/q
    #q_or_K_st = '_K' + str(K)
    Pii = 10/(n/nc) 
    
    fixed =  '_n' + str(n) + '_nc' + str(nc) + '_' + 'B' + str(B).replace('.','') + '_p' + str(np.round(p,3)).replace('.','')  # naming convention
    x_label = [fixed + "_incrEdges", fixed + "_incrEdges", fixed + "_incrEdges"]
    x_var = ['Pij', 'global', 'average']
    x_plot = ['Probability of an edge between 2 communities $P_{out}$',
              'Global clustering coefficient $C$',
              'Network Average Clustering Coefficient $C$']
    beta = [1,2,3]
    for b in beta:
        title = ['$\\beta={}, n={}, B={}, n_c={}, K={}, E[d_i]=10$'.format(b, n, B, nc, K),
                 '$\\beta={}, n={}, B={}, n_c={}, K={}, E[d_i]=10$'.format(b, n, B, nc, K),
                 '$\\beta={}, n={}, B={}, n_c={}, K={}, E[d_i]=10$'.format(b, n, B, nc, K)]
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
    plt.savefig(save_path+graph+'_'+x_var+deg_str+experiment+estimators_str+'.pdf')
    plt.close()

    #Create and save MSE plots
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    newData['RelBias_sq'] = df['Bias']**2
    sns.lineplot(x=x_var, y='RelBias_sq', hue='Estimator', style='Estimator', data=newData, legend='brief', markers=True)

    ax2.set_xlabel(x_plot, fontsize = 18)
    ax2.set_ylabel("MSE", fontsize = 18)
    ax2.set_title(title, fontsize=20)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path+graph+'_'+x_var+deg_str+experiment+estimators_str+'_MSE.pdf')
    plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
