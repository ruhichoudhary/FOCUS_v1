import os
import sys
import pandas as pd
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import math
from gym.spaces import Dict, Discrete, MultiBinary, MultiDiscrete
import gym 
from gym import spaces


import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle
from rdkit.Chem import AllChem, QED, DataStructs, Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from unicodedata import decimal
import numpy as np
 

from Utils.sascore import calculateScore

#check how many invalids.
#Make sure average QED, logP, SAscore, MlWt is within desirable range
#Smina score should be at least -5.8 
#Interactions

import pandas as pd
import os
import sys
import glob
 

def assignOuterReward(df):
    temp = df
    temp['OutReward'] = 0
    point = []
    data = temp.iloc[:6]
    df = pd.DataFrame(data)
    grade = 0
    for i in df['smina_MDP']:
        if i <= -5.8:
            point.append(0.1)
        else:
            point.append(0)
    for i in df['smina_PEP']:
        if (i <= -5.8) & (i >= -7.5):
            point.append(0.1)
        else:
            point.append(0)
    for i in df['smina_MCT1']:
        if (i <= -2.0) & (i >= -6.0):
            point.append(0.1)
        else:
            point.append(0)
    for i in df['hydrophobicContacts_NOD2']:
        if i >= 18:
            grade = grade + (i*0.01)
            point.append(grade)
        else:
            point.append(0)
    for i in df['hydrogenBonds_NOD2']:
        if i >= 2:
            grade = grade + (i*0.2)
            point.append(grade)
        else:
            point.append(0)
    for i in df['saltBridges_NOD2']:
        if i >= 1:
            grade = grade + (i*0.05)
            point.append(grade)
        else:
            point.append(0)
    for i in df['piPiStackingInteractions_NOD2']:
        if i >= 1:
            grade = grade + (i*0.05)
            point.append(grade)
        else:
            point.append(0)
    for i in df['tStackingInteractions_NOD2']:
        if i >= 1:
            grade = grade + (i*0.05)
            point.append(grade)
        else:
            point.append(0)
    for i in df['cationPiInteractions_NOD2']:
        if i >= 1:
            grade = grade + (i*0.05)
            point.append(grade)
        else:
            point.append(0)
    for i in df['electrostaticEnergies_NOD2']:
        if i >= 8:
            grade = grade + (i*0.05)
            point.append(grade)
        else:
            point.append(0)
    for i in df['ASN_PEP']:
        if i >= 1:
            grade = grade + (i*0.1)
            point.append(grade)
        else:
            point.append(0)
    for i in df['GLU_PEP']:
        if i >= 1:
            grade = grade + (i*0.1)
            point.append(grade)
        else:
            point.append(0)
    for i in df['LYS_MCT']:
        if i >= 1:
            grade = grade + (i*0.1)
            point.append(grade)
        else:
            point.append(0)
    for i in df['hydrophobicContacts_MCT']:
        if i >= 1:
            grade = grade + (i*0.1)
            point.append(grade)
        else:
            point.append(0)
        final = [point[i::6] for i in range(6)]
        points = [sum(i) for i in final]
    for i in range(6):
        temp.at[i, 'OutReward'] = points[i]

    return


def outerreward(df,current_reward):
    assignOuterReward(df)
    score = 0 
    valid_df = df.loc[(df['qed'] != -1) & (df['pains'] == 1)& (df['macro'] == 1)]
 
    valid_df = valid_df.fillna(0)
    smina_valid_df = valid_df[(valid_df.smina_MDP < 0)&(valid_df.smina_PEP < 0)&(valid_df.smina_MCT1 < 0)]
    
    best_smina_MDP = smina_valid_df['smina_MDP'].min()
    best_smina_PEP = smina_valid_df['smina_PEP'].min()
    best_smina_MCT1 = smina_valid_df['smina_MCT1'].min()

    if best_smina_MDP <= -5.8:
        score = score + 0.3
    if ((best_smina_PEP <= -5.8)&(best_smina_PEP >= -7.5)):
        score = score + 0.3
    if ((best_smina_MCT1 <= -2)&(best_smina_MCT1 >= -6.0)):
        score = score + 0.1
    
    MDP_fplist = [20, 3, 0, 0, 0, 0, 1, 10]
    if smina_valid_df['hydrophobicContacts_NOD2'].max() >= 18:
        score = score + (smina_valid_df['hydrophobicContacts_NOD2'].max()*0.01)
    if smina_valid_df['hydrogenBonds_NOD2'].max() >= 2:
        score = score + (smina_valid_df['hydrogenBonds_NOD2'].max()*0.2)
    if smina_valid_df['piPiStackingInteractions_NOD2'].max() >= 0:
        score = score + 0.05
    if smina_valid_df['tStackingInteractions_NOD2'].max() >= 0 :
        score = score + 0.05
    if smina_valid_df['cationPiInteractions_NOD2'].max() >= 0:
        score = score + 0.05
    if smina_valid_df['saltBridges_NOD2'].max() >= 1:
        score = score + 0.1
    if smina_valid_df['electrostaticEnergies_NOD2'].max() >= 8:
        score = score + 0.05

    Pep_fplist = [1,8]
    if (smina_valid_df['ASN_PEP'].max() > 0) & (smina_valid_df['ASN_PEP'].max() < 3):
        score = score + 0.1
    if (smina_valid_df['GLU_PEP'].max() >= 2) & (smina_valid_df['GLU_PEP'].max() < 11) : 
        score = score + 0.1

    mc_fplist = [9, 4, 0, 9, 0, 9, 2, 8]
    if (smina_valid_df['LYS_MCT'].max() >= 1) & (smina_valid_df['LYS_MCT'].max() < 5):
        score = score + 0.1
    if (smina_valid_df['hydrophobicContacts_MCT'].max() <=25) &  (smina_valid_df['hydrophobicContacts_MCT'].max() >= 7):
        score = score + 0.1
    score = score+(current_reward/10)

    values = [score, best_smina_MDP,best_smina_PEP,best_smina_MCT1,smina_valid_df['hydrophobicContacts_NOD2'].max(),
                smina_valid_df['hydrogenBonds_NOD2'].max(),smina_valid_df['piPiStackingInteractions_NOD2'].max(),
                smina_valid_df['tStackingInteractions_NOD2'].max(),smina_valid_df['cationPiInteractions_NOD2'].max(),
                smina_valid_df['saltBridges_NOD2'].max(),smina_valid_df['electrostaticEnergies_NOD2'].max(),smina_valid_df['ASN_PEP'].max(),
                smina_valid_df['GLU_PEP'].max(), smina_valid_df['LYS_MCT'].max(), smina_valid_df['hydrophobicContacts_MCT'].max()]

    retValueList = [0 if math.isnan(x) else x for x in values]
    return retValueList
        
def currentstate(df,current_reward):
    #state =Dict({"inner_loop_reward":0, "smina_NOD2": 0,"smina_PEP": 0,"smina_MCT": 0, "hydrophobicContacts_NOD2":0,"hydrogenBonds_NOD2": 0, "piPiStackingInteractions_NOD2":0,"tStackingInteractions_NOD2":0,"cationPiInteractions_NOD2":0,"saltBridges_NOD2":0,"electrostaticEnergies_NOD2":0,"ASN_PEP":0,"GLU_PEP":0,"LYS_MCT":0,"hydrophobicContacts_MCT":0})  
    
    state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)  

    
    valid_df = df.loc[(df['qed'] != -1) & (df['pains'] == 1)& (df['macro'] == 1)]

    valid_df = valid_df.fillna(0)
    smina_valid_df = valid_df[(valid_df.smina_MDP < 0)&(valid_df.smina_PEP < 0)&(valid_df.smina_MCT1 < 0)]
    print(smina_valid_df)
   
    state[1] = smina_valid_df['smina_MDP'].min()
    state[2] = smina_valid_df['smina_PEP'].min()
    state[3] = smina_valid_df['smina_MCT1'].min()
  

    MDP_fplist = [20, 3, 0, 1, 2, 0, 1, 10]
    state[4] = smina_valid_df['hydrophobicContacts_NOD2'].max()
    state[5] = smina_valid_df['hydrogenBonds_NOD2'].max()    
    state[6] = smina_valid_df['piPiStackingInteractions_NOD2'].max() 
    state[7] = smina_valid_df['tStackingInteractions_NOD2'].max() 
    state[8] = smina_valid_df['cationPiInteractions_NOD2'].max()   
    state[9] = smina_valid_df['saltBridges_NOD2'].max()
    state[10] = smina_valid_df['electrostaticEnergies_NOD2'].max() 
    
    Pep_fplist = [1,8] #[2,19]
    state[11] = smina_valid_df['ASN_PEP'].max()
    state[12] = smina_valid_df['GLU_PEP'].max() 
    
    mc_fplist = [9, 4, 0, 9, 0, 9, 2, 8]
    state[13] = smina_valid_df['LYS_MCT'].max()
    state[14] = smina_valid_df['hydrophobicContacts_MCT'].max()
    state[0] = current_reward
    state[np.isnan(state)] = 0

    return state        
    



    
    

