# plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def main(beta=1, B=0.06, p=1, cluster_selection = "bernoulli"): 
    graph = "SBM"
    n = 1000
    nc = 50
    p_in = 0.5 #10/(n/nc) 
    p_out = (0.5-p_in)/49

    load_path = 'output/' + 'deg' + str(beta) + '/' + cluster_selection + '/'

    Bstr = str(np.round(B,3)).replace('.','')
    pstr = str(np.round(p,3)).replace('.','')
    
    experiment = 'correlation'
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + pstr + '_B' + Bstr # naming convention
    x_label = [experiment + fixed]
    x_var = ['Phi']
    x_plot = ['$\phi$']
    title = ['$\\beta={}, SBM({},{},{},{}), B={}, p={}$'.format(beta, n, nc, p_in, p_out, B, np.round(p,3))]
    for ind in range(len(x_var)):
        plot(load_path,cluster_selection,x_var[ind],x_label[ind],beta,x_plot[ind],title[ind])

def plot(load_path,cluster_selection_RD,x_var,x_label,model,x_plot,title):
    save_path = 'plots/' + 'deg' + str(model) + '/' + cluster_selection_RD + '/'
    deg_str = '_deg' + str(model)
    
    # All possible estiamtors: ['PI-$n$($p$)', 'PI-$n$($B$)', 'HT', 'PI-$\mathcal{U}$($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    # All possible designs: ['Cluster', 'Bernoulli'] - note that this only matters for 'LS-Prop', 'LS-Num','DM', and 'DM($0.75$)'
    estimators = ['PI-$n$($p$)', 'HT', 'LS-Prop', 'DM','DM($0.75$)'] 
    estimators_str = '_n' # other options: 'n', 'U', 'all', 'nUHT' -- basically, which estimators show up in the plots

    experiment = x_label
    print(experiment+deg_str+'_'+x_var+estimators_str)

    # Create and save plots
    full_path = load_path + experiment + deg_str + '_' + cluster_selection_RD + '-full.csv'
    df = pd.read_csv(full_path)
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
    ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path + experiment + deg_str + '_' + cluster_selection_RD + '.png')
    plt.close()

    # Create and save MSE plots
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', data=newData, legend='brief', markers=True, palette=color_pal)

    ax2.set_ylim(-1,10)
    ax2.set_xlabel(x_plot, fontsize = 18)
    ax2.set_ylabel("MSE", fontsize = 18)
    ax2.set_title(title, fontsize=20)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path + experiment + deg_str + '_' + cluster_selection_RD + '_MSE.png')
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ''' 
    beta = [1,2]
    B = [0.06, 0.5]
    probs = [[0.06, 0.25, 1/3, 2/3, 1],    # K in [50, 12, 9, 6, 3] #[0.06, 0.25, 1/3, 2/3, 1]
            [0.5, 0.625, 25/33, 25/29, 1],] # K in [50, 40, 33, 29, 25]
            [0.02, 0.1, 0.2, 0.3, 1]
    '''
    beta = [2,2]
    B = [0.02,0.06] 
    probs = [[0.02], [0.06]]
    design = "bernoulli"  # bernoulli   complete
for b in range(len(beta)):
    print('Plotting degree: {} ({} design)'.format(beta[b], design))
    for p in probs[b]:
        print()
        main(beta[b], B[b], p, cluster_selection=design)
