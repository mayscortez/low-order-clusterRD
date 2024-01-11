The folder **Covariate Community** Correlation has the files related to the experiment varying $\phi$ aka homophily, aka cluster quality with respect to covariate balance.
- `main_correlation.py` is the experiment (fix all parameters, but vary $\phi$ and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
    - the subfolders `deg1`, `deg2`, `deg3` correspond to different models (with  $\beta=1$, $\beta=2$, and $\beta=3$, respectively.)
    - the subsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py` (there is a seperate folder for each model degee)
    - folder structure is the same as output


# Robustness to Misspecification
This folder has the files for the experiments comparing the performance of different estimators and designs under model misspecification.

## Main experiment files
- `main_vary_p.py` fixes all parameters except the treatment probability within clusters $p$, and computes different, possibly misspecifed, estimators
    - the idea is to compare the robustness of the different designs+estimators as we reduce or increase the extrapolation error
- `execute_main-vary_p.py` is the file to actually run the experiment that varies $p$
- `main_vary_phi.py` fixes all parameters except the balance (w.r.t. covariates) level $\phi$, and computes different, possibly misspecifed, estimators
    - When $\phi=0$, there is high/perfect homophily. When $\phi=0.5$ there is no homophily. The idea is to compare the robustness of the different designs+estimators as we reduce or increase the stregnth of the homophily
- `execute_main-vary_phi.py` is the file to actually run the experiment that varies $\phi$
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as the file above, but with some name changes inside the file

## Plotting
`plots_rm_vary_p.py` is a script that plots the output from the experiment `main_vary_p`
- To run this file, scroll down to the bottom. 
- The parameters to change, depending on what you'd like to plot, are `models`, `B`, `Piis`, `Pijs`, `phis`, `cluster_selection`, `type`, and `estimators`
- For example, to plot the MSE as a function of $p$ for the Polynomial Interpolation estimators under cluster staggered rollout and Bernoulli staggered rollout assuming a linear ($\beta=1$) model when the true underlying model is degree 4 polynomial ($\beta=4$), the treatment budget is $B=0.06$, cluster selection is Bernoulli design, clusters are balanced across covariates ($\phi=0.5$), and the graph is made up of disjoint clusters, the parameters would be:
    - `models = [{'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]`
    - `B = 0.06`
    - `Piis = [0.5]`
    - `Pijs = [0]`
    - `phis = [0.5]`
    - `cluster_selection = "bernoulli"`
    - `type = "MSE"`
    - `estimators = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$']`
- To plot for several different graph structures and balance/homophily levels, note that you can add these all to the lists `Piis`, `Pijs`, `phis`. The code is already set up to iterate over the values in there.
- To see all the possible estimators and models (currently implemented), see the block comment at the bottom of the file `plots_rm_vary_p.py`.

`plots_rm_vary_phi.py` is a script that plots the output from the experiment `main_vary_phi`. 
- The structure is the same as above, except instead of the variable `phis`, you have the variable `probs` corresponding to the treatment probabilities (within clusters) you want to plot for.
- You can plot MSE as a function of $\phi$, Relative Bias as a function of $\phi$, or both. 

## Saving experiment output and plots
- `output` is a folder containing the results from the experiments
    - the subfolders `vary_p` and `vary_phi` contain the results from the respective experiments
    - the subsubfolders `ppom1`, `ppom2`, and `ppom3` correspond to results when the *true* model is polynomial with degree $\beta=1$, $\beta=2$, and $\beta=3$, respectively. If we choose to try other models, such as threshold models, we should create a new folder for those results.
    - the subsubsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py`
    - folder structure is the same as `output`


The folder **Increasing Edges** has the files related to the experiment varying the cluster quality with respect to edges crossing clusters.
- `main_increasing_edges.py` is the experiment (fix all parameters, but vary $p_\text{in}$ and $p_\text{out}$, keeping average degree fixed, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `myFunctions_edges.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network; yes, essentially same file as the corresponding two with the same function in the other folders
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py`(there is a seperate folder for each model degee)


# Misc/Other Notes
- Ignore the folders `Lattice_experiments` and `old-code` in the main file directory.