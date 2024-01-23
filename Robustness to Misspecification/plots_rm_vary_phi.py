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

def main(model, B=0.06, p_in=0.5, p=0, cluster_selection = "bernoulli", type='both', estimators=['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)']): 
    model_name = model['name']
    degree = model['degree']
    experiment = 'vary_phi'
    load_path = 'output/' + experiment + '/'  + model_name + '-ppom' + str(degree) + '/' + cluster_selection + '/'              

    n = 1000
    nc = 50
    p_out = (0.5-p_in)/49

    if model["type"] == 'threshold' and model["params"][1] == "prop":
        name = '$\mathrm{Prop}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'threshold' and model["params"][1] == "num":
        name = '$\mathrm{Num}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'saturation':
        name = '$\mathrm{Sat}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'ppom':
        name = 'Low-Order Polynomial'
    else:
        raise ValueError("Model type is invalid.")

    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_B' + str(B).replace('.','') + '_p' + str(np.round(p,3)).replace('.','') # naming convention
    
    x_label = [experiment + '-' + model_name + fixed + '_' +  cluster_selection]
    x_var = ['Phi']
    x_plot = ['$\phi$']
    title = ['True Model: {} with $\\beta={}$ \n SBM$({},{},{},{}), B={}, p={}$'.format(name, degree, n, nc, np.round(p_in,3), np.round(p_out,3), B, np.round(p,3))]
    for ind in range(len(x_var)):
        plot(load_path, degree, x_var[ind],x_label[ind],model_name,x_plot[ind],title[ind], cluster_selection, estimators, type)

def plot(load_path, degree, x_var, experiment_label, model, x_plot, title, cluster_selection, estimators, type='both'):
    save_path = 'plots/' + 'vary_phi' + '/'  + model + '-ppom' + str(degree) + '/' + cluster_selection + '/'    

    color_map = {'PI-$n(p;1)$': '#6a9f00', 'PI-$n(p;2)$':'#b2ce02', 'PI-$n(p;3)$': '#feba01',
                 'PI-$\mathcal{U}(p;1)$': '#1b45a6', 'PI-$\mathcal{U}(p;2)$': '#019cca', 'PI-$\mathcal{U}(p;3)$': '#009d9d', 
                 'HT': '#e51e31',
                 'DM-C': '#e35610', 'DM-C($0.75$)': '#ff7787',  #'DM-C': '#fb5082'
                 'PI-$n(B;1)$': '#e0d100', 'PI-$n(B;2)$': '#ffa706', 'PI-$n(B;3)$': '#ed5902',
                 'LS-Prop(1)': '#009633', 'LS-Prop(2)': '#95c413', 'LS-Prop(3)': '#f5c003',
                 'LS-Num(1)': '#46c1c1', 'LS-Num(2)': '#3e66c9', 'LS-Num(3)': '#c42796', 
                 'DM': '#9e35af', 'DM($0.75$)': '#c069c9'}
    
    color_pal = [color_map[est] for est in estimators]

    print('\n'+experiment_label+'_'+'vary-'+x_var)
    
    df = pd.read_csv(load_path + experiment_label + '-full.csv')
    newData = df.loc[df['Estimator'].isin(estimators)]

    plt.rc('text', usetex=True)
    
    if (type == 'MSE' or type == 'both'):
        # MSE plots
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', errorbar=None, data=newData, legend='brief', markers=True, palette=color_pal)

        #ax.set_xlim(0,0.001)
        #ax.set_ylim(-0.25,0.75)
        ax.set_xlabel(x_plot, fontsize = 18)
        ax.set_ylabel("MSE", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        save_str = save_path + experiment_label + '_MSE.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()

    if (type == 'Bias' or type == 'both'):
        # Plot with all the estimators
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', errorbar='sd', data=newData, legend='brief', markers=True, palette=color_pal)

        #ax.set_xlim(0,0.001)
        #ax.set_ylim(-2,4)
        ax.set_xlabel(x_plot, fontsize = 18)
        ax.set_ylabel("Relative Bias", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)
        plt.grid()
        plt.tight_layout()

        save_str = save_path + experiment_label + '.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()
    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    models = [{'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []},
            {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]
    B = 0.06
    Piis = [0.5]        # edge probability within clusters        
    probs = [0.06, 0.25, 1]   # covariate balance parameter (phi = 0 is exact homophily, phi = 0.5 is no homophily)
    cluster_selection = "bernoulli" # other option: "complete"
    type = "both" # other options:  "Bias"   "MSE"   "both"
    lin_nonpara_ests = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'HT', 'DM-C', 'DM-C($0.75$)', 'LS-Prop(1)', 'LS-Num(1)'] # which estimators to plot
    linear_ests = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)'] # which estimators to plot
    cluster_ests = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'DM-C', 'DM-C($0.75$)'] # which estimators to plot
for i in range(len(models)):
    print('Plotting for true model: {} ({} design)'.format(models[i]['name'],cluster_selection))
    for j in range(len(Piis)):
        for p in probs:
            main(models[i], B, Piis[j], p, cluster_selection, type, estimators=linear_ests)
    print() 

'''
models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []},
            {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]
Piis = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.01]
Pijs = [(0.5-p)/49 for p in Piis]
probs = [0.06, 0.12, 0.25, 1/3, 2/3, 1]

Possible estimators: 
    names_ClRD = ['PI-$n(p;1)$', 'PI-$\mathcal{U}(p;1)$', 'PI-$n(p;2)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$n(p;3)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'DM-C', 'DM-C($0.75$)']
    names_BRD = ['PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)', 'PI-$n(B;2)$', 'LS-Prop(2)', 'LS-Num(2)', 'PI-$n(B;3)$', 'LS-Prop(3)', 'LS-Num(3)','DM', 'DM($0.75$)']
NOTE: for the colors to be consistent, the estimators have to be in a specific order, regardless of which are included
NOTE (cont): they are in order in names_ClRD and names_BRD, so as long as that relative order is kept and cluster design-based ones come before bernoulli design-based ones, it should be fine

Examples:
    estimators = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'PI-$n(B;1)$', 'LS-Prop(1)', 'DM($0.75$)'] 
    estimators = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)']
    estimators = ['PI-$\mathcal{U}(p;1)$', 'HT', 'DM-C', 'DM-C($0.75$)', 'PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)']  
    estimators = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'PI-$n(B;1)$', 'PI-$n(B;2)$'] 
'''
