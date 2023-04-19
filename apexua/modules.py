import os
# import spotpy
import pandas as pd
import numpy as np
from apexua.models import APEX_setup
from apexua.likelihoods import gaussianLikelihoodMeasErrorOut as GLMEOUT
from apexua.likelihoods import gaussianLikelihoodHomoHeteroDataError as GLHHDE
from apexua.algorithms import dream_ac, fast_ac
from apexua import analyzer


def run_dream(info, 
        dbname="DREAM_apex", dbformat="csv", parallel='mpc', obj_func=GLHHDE):
    # spot_setup = single_setup(GausianLike)
    delete_old_files(info)
    # Bayesian algorithms should be run with a likelihood function
    # obj_func = ua.likelihoods.gaussianLikelihoodHomoHeteroDataError
    # obj_func = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut
    apex_model = APEX_setup(info, parallel=parallel, obj_func=obj_func)
    # Select seven chains and set the Gelman-Rubin convergence limit
    delta = 3
    convergence_limit = 1.2

    # Other possible settings to modify the DREAM algorithm, for details see Vrugt (2016)
    c = 0.1
    nCr = 3
    runs_after_convergence = 1
    acceptance_test_option = 6
    eps=10e-6

    # sampler = spotpy.algorithms.dream(
    #     apex_model, dbname=dbname, dbformat=dbformat, parallel=parallel,
    #     # dbappend=True
    #     )
    sampler = dream_ac.dream(
        apex_model, dbname=dbname, dbformat=dbformat, parallel=parallel,
        dbappend=True
        )
    r_hat = sampler.sample(
        int(info.loc["NumberRuns", "val"]),
        int(info.loc["NumberChains", "val"]),
        nCr,
        delta,
        c,
        eps,
        convergence_limit,
        runs_after_convergence,
        acceptance_test_option,
    )
    # if dbformat == 'csv':
    #     results = pd.DataFrame(sampler.getdata())
    #     results.to_csv(f"{dbname}.csv", index=False)
    #     #########################################################
    #     # Example plot to show the convergence #################
    #     results02 = analyzer.load_csv_results(f"{dbname}")
    #     analyzer.plot_gelman_rubin(results02, r_hat, fig_name="DREAM_r_hat.png")
    np.savetxt("r_hat.csv", r_hat, delimiter=",")
    if dbformat == "ram":
        results = pd.DataFrame(sampler.getdata())
        results.to_csv(f"{dbname}.csv", index=False)
        #########################################################
        # Example plot to show the convergence #################
        results02 = analyzer.load_csv_results(f"{dbname}")
        analyzer.plot_gelman_rubin(results02, r_hat, fig_name="DREAM_r_hat.png")
        
## it is going to be interesting
def run_fast(
        info, 
        dbname="FAST_apex", dbformat="csv", parallel='mpc', obj_func=None):
    apex_model = APEX_setup(
        info, parallel=parallel, obj_func=obj_func)
    # Select number of maximum allowed repetitions
    sampler = fast_ac.fast(
            apex_model, dbname=dbname, 
            dbformat=dbformat, parallel=parallel
            )
    sampler.sample(int(info.loc["NumberRuns", "val"]))


def delete_old_files(info):
    if os.path.isfile(os.path.join(info.loc["WD", "val"], "DREAM_apex.csv")):
        print("found obsolete outputs ...")
        os.remove(os.path.join(info.loc["WD", "val"], "DREAM_apex.csv"))
        print("...deleted ...")


    

