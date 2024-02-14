# plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
})
plt.rcParams["mathtext.fontset"]

def main(beta=1, p=0.06, q=1, p_in = 0.5, cluster_selection = "bernoulli", estimators = ['PI-$n$($p$)', 'HT', 'LS-Prop', 'DM','DM($0.75$)'], plot_type="MSE"): 
    n = 1000
    nc = 50
    p_out = (0.5-p_in)/49

    load_path = 'output/' + 'deg' + str(beta) + '/' + cluster_selection + '/'

    pstr = str(np.round(p,3)).replace('.','')
    qstr = str(np.round(q,3)).replace('.','')
    
    experiment = 'correlation'
    
    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_q' + qstr + '_p' + pstr # naming convention
    x_label = [experiment + fixed]
    x_var = ['Phi']
    x_plot = ['covariate balance $\phi$']
    title = ['$\\beta={}, q = {}$'.format(beta, q)]
    for ind in range(len(x_var)):
        plot(load_path,cluster_selection,x_var[ind],x_label[ind],beta,x_plot[ind],title[ind],estimators,plot_type)

def plot(load_path,cluster_selection_RD,x_var,x_label,model,x_plot,title,estimators,plot_type):
    save_path = 'plots/' + 'deg' + str(model) + '/' + cluster_selection_RD + '/'
    deg_str = '_deg' + str(model)
    
    color_map = {'PI($q$)': '#0296fb', 
                 'PI-$\mathcal{U}$($q$)': '#0d7901',
                 'HT': '#1100ff', 
                 'DM-C': '#990002', 
                 'DM-C($0.75$)': '#5f0099',
                 'E[PI$(q)|\mathcal{U}$]': '#e20287', #e20287
                 'E[PI-$\mathcal{U}(q)|\mathcal{U}$]': '#ffc900',
                 'PI($p$)': '#0296fb',
                 'LS-Prop': '#fb6702',
                 'LS-Num': '#15c902',
                 'DM': '#ff0004',
                 'DM($0.75$)':'#9e00ff'}
    color_pal = [color_map[est] for est in estimators]
    
    experiment = x_label
    print(experiment+deg_str+'_'+x_var)

    # Create and save plots
    full_path = load_path + experiment + deg_str + '_' + cluster_selection_RD + '-full.csv'
    df = pd.read_csv(full_path)
    newData = df.loc[df['Estimator'].isin(estimators)]

    plt.rc('text', usetex=True)


    if (plot_type == 'Bias' or plot_type == 'both'):
        # Plot with all the estimators
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, errorbar='sd', legend='brief', markers=True, palette=color_pal)

        #ax.set_xlim(0,0.001)
        ax.set_xlabel(x_plot, fontsize = 18)
        #ax.set_ylim(0,1)
        ax.set_ylabel("Relative Bias", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        plt.savefig(save_path + experiment + deg_str + '_' + cluster_selection_RD + '.png')
        plt.close()

    if (plot_type == 'MSE' or plot_type == 'both'):
        # Create and save MSE plots
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', data=newData, errorbar=None, legend='brief', markers=True, palette=color_pal)

        ax2.set_ylim(0,1)
        ax2.set_xlabel(x_plot, fontsize = 18)
        ax2.set_ylabel("MSE", fontsize = 18)
        ax2.set_title(title, fontsize=20)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles, labels=labels, loc='upper left', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        plt.savefig(save_path + experiment + deg_str + '_' + cluster_selection_RD + '_MSE.png')
        plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    '''
    beta = [1,2,3]
    B = 0.06
    probs = [0.06, 0.25, 1/3, 2/3, 1]      # K in [50, 12, 9, 6, 3] for B = 0.06
            [0.5, 0.625, 25/33, 25/29, 1]  # K in [50, 40, 33, 29, 25] for B = 0.5
            [0.02, 0.1, 0.2, 0.3, 1]       # K in [1, 5, 10, 15, 50] for B = 0.02
    '''
    beta = [3]
    budget = 0.06 
    probs = [0.25, 3/5, 1]
    p_in = 0.4
    design = "bernoulli"  # bernoulli   complete
    
    # All possible estimators: [ 'E[PI$(q)|\mathcal{U}$]', 'PI($q$)', 'PI-$\mathcal{U}$($q$)', 'HT', 'DM-C', 'DM-C($0.75$)', 'PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    # Note: for colors to match in each plot, the estimator names should be in the same relative order as above
    estimators = ['E[PI$(q)|\mathcal{U}$]', 'PI($q$)']

    plot_type = "Bias"  #option: MSE, Bias, both

    for b in range(len(beta)):
        print('Plotting degree: {} ({} design)'.format(beta[b], design))
        for q in probs:
            print()
            main(beta[b], budget, q, p_in, design, estimators, plot_type)