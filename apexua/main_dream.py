
import modules
from spotpy.likelihoods import gaussianLikelihoodHomoHeteroDataError as GLHHDE
import multiprocessing

if __name__ == "__main__":
    # print('test')
    multiprocessing.freeze_support()
    parallel = "mpc"
    proj_dir = "D:/Projects/Tools/APEX-CUTE/Analysis/test01"

    # Initialize the Hymod example (will only work on Windows systems)
    # spot_setup=spot_setup(parallel=parallel)
    modules.run_dream(proj_dir, eps=10e-6, nChains=12, 
        dbname="DREAM_apex", dbformat="csv", parallel='mpc', obj_func=GLHHDE)