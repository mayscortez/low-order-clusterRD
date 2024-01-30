from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def main(beta=[1,2,3], p=0.06, q=1, estimators = ['PI-$n$($p$)', 'HT', 'LS-Prop', 'DM','DM($0.75$)']):
    n = 1000
    nc = 50
    
    fixed =  '_n' + str(n) + '_nc' + str(nc) + '_' + 'p' + str(p).replace('.','') + '_q' + str(np.round(q,3)).replace('.','')  # naming convention
    x_label = "incrEdges" + fixed
    x_var = ['Pij'] # x_var = ['Pij', 'global', 'average']
    x_plot = ['probability of edge between 2 communities $p_{\mathrm{out}}$'] # , 'Global clustering coefficient $\mathcal{C}_\mathrm{global}$', 'Network Average Clustering Coefficient $\mathcal{C}_\mathrm{avg}$']
    for b in beta:
        title = ['$\\beta={}, q={}$'.format(b, np.round(q,3))]#, '$\\beta={}, n={}, n_c={}, B={}, p={}$'.format(b, n, nc, B, np.round(p,3)), '$\\beta={}, n={}, n_c={}, B={}, p={}$'.format(b, n, nc, B, np.round(p,3))]
        for ind in range(len(x_var)):
            plot(x_var[ind],x_label,b,x_plot[ind],title[ind],estimators)


def plot(x_var,x_label,beta,x_plot,title,estimators):
    load_path = 'output/' + 'deg' + str(beta) + '/'
    save_path = 'plots/' + 'deg' + str(beta) + '/'
    deg_str = '_deg' + str(beta)
    
    color_map = {'PI($q$)': '#0296fb', 
                 'PI-$\mathcal{U}$($q$)': '#0d7901',
                 'HT': '#fc00ff', 
                 'DM-C': '#ff7c00', 
                 'DM-C($0.75$)': '#03ff00',
                 'PI($p$)': '#0296fb',
                 'LS-Prop': '#fb6702',
                 'LS-Num': '#15c902',
                 'DM': '#ff6200',
                 'DM($0.75$)':'#ff891a'}
    color_pal = [color_map[est] for est in estimators]

    experiment = x_label
    print(experiment+deg_str+'_'+x_var)

    # Create and save plots
    df = pd.read_csv(load_path + experiment + deg_str + '_bernoulli-full.csv')
    df = df.assign(Estimator = lambda df: df.Estimator.replace({'PI-$n$($p$)':'PI($q$)', 'PI-$\mathcal{U}$($p$)': 'PI-$\mathcal{U}$($q$)', 'PI-$n$($B$)':'PI($p$)'}))
    newData = df.loc[df['Estimator'].isin(estimators)]

    plt.rc('text', usetex=True)

    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, errorbar='sd', legend='brief', markers=True, palette=color_pal)

    #ax.set_xlim(0,0.001)
    ax.set_xlabel(x_plot, fontsize = 18)
    #ax.set_ylim(-0.75,0.75)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=18)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path + experiment + deg_str + '_' + x_var + '_bern.png')
    plt.close()

    #Create and save MSE plots
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', data=newData, errorbar=None, legend='brief', markers=True, palette=color_pal)

    #ax.set_xlim(0,0.001)
    ax2.set_xlabel(x_plot, fontsize = 18)
    ax2.set_ylim(0,2)
    ax2.set_ylabel("MSE", fontsize = 18)
    ax2.set_title(title, fontsize=20)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels, loc='upper left', fontsize = 14)
    #plt.grid()
    plt.tight_layout()

    plt.savefig(save_path + experiment + deg_str + '_' + x_var +'_bern_MSE.png')
    plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ''' 
    beta = [1,2,3]
    B = 0.06
    probs = [0.06, 0.25, 1/3, 2/3, 1]    # K in [50, 12, 9, 6, 3] 
    '''
    beta = [1,2,3,4]
    budget = 0.06
    probs = [1,3/5]
    
    # All possible estimators: ['PI-$n$($p$)', 'PI-$\mathcal{U}$($p$)', 'HT', 'DM-C', 'DM-C($0.75$)', 'PI-$n$($B$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    # Note: for colors to match in each plot, the estimator names should be in the same relative order as above
    estimators = ['PI($q$)', 'HT', 'DM-C', 'DM-C($0.75$)' ]

    for p in probs:
        print('Plotting p: {} (bernoulli design)'.format(p))
        print()
        main(beta, budget, p, estimators)
