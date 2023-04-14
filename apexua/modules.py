import os
# import spotpy
import pandas as pd
import numpy as np
from apexua.models import APEX_setup
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GLMEOUT
from spotpy.likelihoods import gaussianLikelihoodHomoHeteroDataError as GLHHDE
import spotpy


# def run_dream(ui):
#     APEX_setup(ui)

def run_dream(info, eps=10e-6, nChains=10, 
        dbname="DREAM_apex", dbformat="csv", parallel='mpc', obj_func=GLHHDE):
    # spot_setup = single_setup(GausianLike)
    # os.chdir(os.path.join(info.loc["WD", "val"], info.loc["Mode", "val"]))
    spot_setup = APEX_setup(info, parallel=parallel, obj_func=obj_func)
    # Select seven chains and set the Gelman-Rubin convergence limit
    delta = 3
    convergence_limit = 1.2

    # Other possible settings to modify the DREAM algorithm, for details see Vrugt (2016)
    c = 0.1
    nCr = 3
    runs_after_convergence = 1
    acceptance_test_option = 6

    # sampler = spotpy.algorithms.dream(
    #     spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,
    #     # dbappend=True
    #     )
    sampler = spotpy.algorithms.dream(
        spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,
        dbappend=True
        )
    r_hat = sampler.sample(
        1000,
        nChains,
        nCr,
        delta,
        c,
        eps,
        convergence_limit,
        runs_after_convergence,
        acceptance_test_option,
    )
    if dbformat == 'ram':
        results = pd.DataFrame(sampler.getdata())
        results.to_csv(f"{dbname}.csv", index=False)
        #########################################################
        # Example plot to show the convergence #################
        results02 = spotpy.analyser.load_csv_results(f"{dbname}")
        spotpy.analyser.plot_gelman_rubin(results02, r_hat, fig_name="DREAM_r_hat.png")