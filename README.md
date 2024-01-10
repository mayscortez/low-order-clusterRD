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

The folder **Robustness to Misspecification** has the files related to the experiments that compare the performance of different estimators and designs under model misspecification.
- `main_vary_p.py` fixes all parameters except the treatment probability within clusters $p$, and computes different, possibly misspecifed, estimators
    - the idea is to compare the robustness of the different designs+estimators as we reduce or increase the extrapolation error
- `execute_main-vary_p.py` is the file to actually run the experiment that varies $p$
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as `experiment_functions.py` above, but with some name changes inside the file
- `plots_rm_vary_p.py` is a script that plots the output from the experiment `main_vary_p`
- `plots_rm_vary_phi.py` is a script that plots the output from the experiment `main_vary_phi`
- `output` is a folder containing the results from the experiments
    - the subfolders `vary_p` and `vary_phi` contain the results from the respective experiments
    - the subsubfolders `ppom1`, `ppom2`, and `ppom3` correspond to results when the *true* model is polynomial with degree $\beta=1$, $\beta=2$, and $\beta=3$, respectively. If we choose to try other models, such as threshold models, we should create a new folder for those results.
    - the subsubsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py`
    - folder structure is the same as `output`

The folder **Increasing Edges** has the files related to the experiment varying the cluster quality with respect to edges crossing clusters.
- `main_increasing_edges.py` is the experiment (fix all parameters, but vary $p_\text{in}$ and $p_\text{out}$, keeping average degree fixed, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `myFunctions_edges.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as `experiment_functions.py` above
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py`(there is a seperate folder for each model degee)