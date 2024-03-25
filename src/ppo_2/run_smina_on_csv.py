

import sys
import os
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4")
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4/src")
sys.path.append("/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4/src/smina")
import numpy as np
import pandas as pd
import time
import warnings
import shutil
from datetime import datetime
import hydra
from config.config import cs
from omegaconf import DictConfig

from smina.runsmina_parallel import SMINA_data

@hydra.main(config_path="../config/", config_name="config")
def run_smina_on_df(config):
    """
    Reads a CSV file containing molecular structures and performs docking analysis on each structure
    using SMINA. The results are stored in a new DataFrame.

    Parameters:
    - config: Configuration object provided by Hydra based on the specified config file.

    The function iterates through the rows of the input DataFrame, filtering out certain rows based
    on specified criteria. For each selected molecule, it initiates a SMINA_data object to perform
    docking simulations. The results, including scores and interaction types, are stored in a new DataFrame
    which is then saved to a CSV file.
    """
    
    df = pd.read_csv("/Users/ruhichoudhary/code/gitrepos/rlgmcts/summary1.csv")
    output_df=pd.DataFrame(columns=['SMILES','smina_MDP','smina_PEP','smina_MCT1','hydrophobicContacts_NOD2'
        ,'hydrogenBonds_NOD2','halogenBonds_NOD2','piPiStackingInteractions_NOD2','tStackingInteractions_NOD2'
        ,'cationPiInteractions_NOD2','saltBridges_NOD2','electrostaticEnergies_NOD2','ASN_PEP','GLU_PEP','hydrophobicContacts_MCT'])
    output_base_dir="/Users/ruhichoudhary/runs_data/output"
    for index in range(df.shape[0]):
        row=df[index]
        if (row['qed'] == -1) & (row['pains'] != 1) & (row['macro'] != 1):
            continue
        smina_data_object=SMINA_data(df['SMILES'][index], index, output_base_dir, "/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4")
        smina_data_object.run_smina_processes()
        smina_result=smina_data_object.get_data()

        output_df.at[index,'smina_MDP'] = smina_result['NOD2_score']
        output_df.at[index,'smina_PEP'] = smina_result['7pn1_score']
        output_df.at[index,'smina_MCT1'] = smina_result['MCT1_score']
        output_df.at[index,'hydrophobicContacts_NOD2'] = smina_result['NOD2_hydrophobicContacts']
        output_df.at[index,'hydrogenBonds_NOD2'] =smina_result['NOD2_hydrogenBonds']
        output_df.at[index,'halogenBonds_NOD2'] =smina_result['NOD2_halogenBonds']
        output_df.at[index,'piPiStackingInteractions_NOD2'] =smina_result['NOD2_piPiStackingInteractions']
        output_df.at[index,'tStackingInteractions_NOD2'] =smina_result['NOD2_tStackingInteractions']
        output_df.at[index,'cationPiInteractions_NOD2'] =smina_result['NOD2_cationPiInteractions']
        output_df.at[index,'saltBridges_NOD2'] =smina_result['NOD2_saltBridges'] 
        output_df.at[index,'electrostaticEnergies_NOD2'] = smina_result['NOD2_electrostaticEnergies']
        output_df.at[index, 'ASN_PEP'] = smina_result['PEP_hydrogenBonds']
        output_df.at[index, 'GLU_PEP'] = smina_result['PEP_closeContacts']
        output_df.at[index, 'LYS_MCT'] = smina_result['MCT1_hydrogenBonds']
        output_df.at[index, 'hydrophobicContacts_MCT'] = smina_result['MCT1_hydrophobicContacts']

    output_df.to_csv(output_base_dir+"/output_df.csv")

if __name__ == "__main__":
    run_smina_on_df()
