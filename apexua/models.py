import os
import pandas as pd
import numpy as np
from apexua.pars import updatePars
from distutils.dir_util import copy_tree, remove_tree
import subprocess
from apexua import parameter
import datetime
from apexua.objectivefunctions import rmse
from apexua.read_output import create_ua_sim_obd


# FORM_CLASS,_=loadUiType(find_data_file("main.ui"))

class APEX_setup(object):
    def __init__(
        self, info,  parallel="seq", obj_func=None
        ):
        self.info = info
        self.obj_func = obj_func
        self.curdir = os.getcwd()
        self.ua_dir = info.loc["WD", "val"]
        self.mod_folder = info.loc["Mode", "val"]
        self.main_dir = os.path.join(self.ua_dir, self.mod_folder)
        os.chdir(self.main_dir)
        self.params = []
        pars_df = self.load_ua_pars()
        for i in range(len(pars_df)):
            self.params.append(
                parameter.Uniform(
                    name=pars_df.iloc[i, 0],
                    low=pars_df.iloc[i, 3],
                    high=pars_df.iloc[i, 4],
                    optguess=np.mean(
                        [float(pars_df.iloc[i, 3]), float(pars_df.iloc[i, 4])]
                    )            
                )
            )
        self.pars_df = pars_df
        self.parallel = parallel
        if self.parallel == "seq":
            pass
        # NOTE & TODO: mpi4py is for linux and ios () 
        if self.parallel == "mpi":
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.mpi_size = comm.Get_size()
            self.mpi_rank = comm.Get_rank()

        # NOTE: read all ui setting here then use as initiates
        # if ui.rb_user_obs_day.isChecked():
        #     self.time_step = "day"
        # if ui.rb_user_obs_mon.isChecked():
        #     self.time_step = "month"
        # self.rchnum = ui.txt_sub_1.text()
        # if ui.rb_user_obs_type_rch.isChecked():
        #     self.obs_type = "rch"
        # if ui.txt_apex_out_1.currentText().upper()=="RCH-FLOW":
        #     self.obs_nam = "Flow(m3/s)"
        # self.obs_path = ui.txt_user_obs_save_path.toPlainText()
        # self.ui = ui
        # APEXCUTE_path_dict = self.dirs_and_paths()
        # os.chdir(APEXCUTE_path_dict['apexcute'])
        # print(inspect.getmodule(ui).__dir__)
        self.time_step = "day"


        
    # def load_sim(self, wd):
    #     stdate_, eddate_, ptcode = self.get_start_end_step()
    #     if ptcode == 6 and self.time_step == "day":
    #         sim_df = read_output.extract_day_stf()
    #     if ptcode == 6 and self.time_step == "month":
    #         sim_df = read_output.extract_day_stf(wd)
    #         print('nope!')
    #         sim_df = sim_df.resample('M').mean()
    #         sim_df['year'] = sim_df.index.year
    #         sim_df['month'] = sim_df.index.month
    #         sim_df['time'] = sim_df['year'].astype(str) + "-" + sim_df['month'].astype(str)
    #     if ptcode == 3 and self.time_step == "month":
    #         sim_df = read_output.extract_mon_stf(wd)
    #     print(sim_df)
    #     return sim_df
    


    def load_ua_pars(self):
        pars_df = pd.read_csv(os.path.join(self.ua_dir, "ua_sel_pars.csv"))
        return pars_df

    def parameters(self):
        return parameter.generate(self.params)
    
    def update_apex_pars(self, parameters):
        # print(f"this iteration's parameters:")
        # print(parameters)
        apex_pars_df = self.pars_df   
        apex_pars_df['val'] = parameters
        self.update_parm_pars(apex_pars_df)


    def update_parm_pars(self, updated_pars, parval_len=8):
        """update parm pars

        Args:
            updated_pars (dataframe):  
            parval_len (int, optional): _description_. Defaults to 8.
        """
        new_pars_df = updated_pars.loc[updated_pars['type']=='parm']
        with open("parms.dat", "r") as f:
            content = f.readlines()
        upper_pars = [x.rstrip() for x in content[:35]] 
        core_pars = [x.rstrip() for x in content[35:46]]
        lower_pars = [x.rstrip() for x in content[46:]]
        n_core_pars = []
        for line in core_pars:
            n_core_pars += [
                str(line[i:i+parval_len]) for i in range(0, len(line), parval_len)
                ]
        parnams = [f"PARM{i}" for i in range(1, len(n_core_pars)+1)]
        core_pars_df = pd.DataFrame({"parnam":parnams, "val":n_core_pars})
        for pnam in core_pars_df["parnam"]:
            if pnam in new_pars_df.loc[:, "name"].tolist():
                new_val = "{:8.4f}".format(float(new_pars_df.loc[new_pars_df["name"] == pnam, "val"].tolist()[0]))
                core_pars_df.loc[
                    core_pars_df["parnam"]==pnam, "val"
                    ] = "{:>8}".format(new_val)
        newdata = core_pars_df.loc[:, "val"].values.reshape(11, 10)

        with open("parms.dat", 'w') as f:
            for urow in upper_pars:
                f.write(urow + '\n')
            for row in newdata:
                f.write("".join(row) + '\n')
            for lrow in lower_pars:
                f.write(lrow + '\n')

    # Simulation function must not return values besides for which evaluation values/observed data are available
    def simulation(self, parameters):     
        if self.parallel == "seq":
            call = ""
        elif self.parallel == "mpi":
            # Running n parallel, care has to be taken when files are read or written
            # Therefor we check the ID of the current computer core
            call = str(int(os.environ["OMPI_COMM_WORLD_RANK"]) + 2)
            # And generate a new folder with all underlying files
            copy_tree(self.main_dir, self.main_dir + call)

        elif self.parallel == "mpc":
            # Running n parallel, care has to be taken when files are read or written
            # Therefor we check the ID of the current computer core
            call = str(os.getpid())
            # And generate a new folder with all underlying files
            # os.chdir(self.wd)
            copy_tree(self.main_dir, self.main_dir + call)
            
        else:
            raise "No call variable was assigned"
        self.main_dir_call =self.main_dir + call
        os.chdir(self.main_dir_call)
        try:
            self.update_apex_pars(parameters)

            comline = "APEX1501.exe"
            run_model = subprocess.Popen(comline, cwd=".", stdout=subprocess.DEVNULL)
            # run_model = subprocess.Popen(comline, cwd=".")
            run_model.wait()
            # all_df = self.all_sim_obd(self.main_dir_call) # read all sim_obd from all 
            all_df = create_ua_sim_obd(self.info)
        except Exception as e:
            raise Exception("Model has failed")
        
        os.chdir(self.curdir)
        # os.chdir(self.main_dir)
        # os.chdir("d:/Projects/Tools/DayCent-CUTE/tools")
        if self.parallel == "mpi" or self.parallel == "mpc":
            remove_tree(self.main_dir + call)
        return all_df['sim'].tolist()



    def evaluation(self):
        # os.chdir(self.main_dir_call)
        all_df = create_ua_sim_obd(self.info)
        return all_df["obd"].tolist()
        # return all_df[self.obs_nam].tolist()


    def objectivefunction(self, simulation, evaluation):
        if not self.obj_func:
            like = rmse(evaluation, simulation)
        else:
            like = self.obj_func(evaluation, simulation)
        return like






# if __name__ == "__main__":
#     APEX_setup.update_apex_pars()

