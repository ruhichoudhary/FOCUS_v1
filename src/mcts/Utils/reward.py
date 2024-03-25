import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import math
import hydra

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle
from rdkit.Chem import AllChem, QED, DataStructs, Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from unicodedata import decimal
 
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from tdc.benchmark_group import admet_group
from config import config

from Utils.sascore import calculateScore
from deepchem import deepchem
from tdc.benchmark_group import admet_group
import numpy as np
from Utils.caco2 import caco2



def getReward(name, init_smiles):
    if name == "QED":
        return QEDReward()
    elif name == "PLogP":
        return PenalizedLogPReward()
    elif name =="MultiReward":
        return MultiReward()

def piecewisevalue(x, molwt):
    value = np.piecewise(float(x)
                , [x < molwt*0.4
                , ((x >= molwt*0.4) & (x < molwt*0.7))
                , ((x >= molwt*0.7) & (x < molwt*1.3))
                , ((x >= molwt*1.3) & (x < molwt*2.0))
                , x >= molwt*2.0]
                , [0,0.5,1.0,0.5,-1]) 
    return value

class Reward:
    weight = [1,1,1,1]

    def __init__(self):
        self.vmin = -100
        self.template = "CC(C(=O)NC(CCC(=O)O)C(=O)N)NC(=O)C(C)OC1C(C(OC(C1O)CO)O)NC(=O)C"
        self.templatemol = Chem.MolFromSmiles(self.template)
        self.templatemolwt = Chem.Descriptors.MolWt(self.templatemol)
        self.templatelen = len(self.template)
        self.tempuplimit = 1.3*self.templatelen
        self.templowlimit = 0.7*self.templatelen
        self.printindex = 100


        self.max_r = -10000
        return

    def reward(self, smiles):
        raise NotImplementedError()

    def setweight(self,weight):
        self.weight = weight

    def setquerymolecule(self,smiles):
        self.querymolecule = smiles
        self.querymol = Chem.MolFromSmiles(smiles)
        self.querymolwt = Chem.Descriptors.MolWt(self.querymol)
        self.querymollen = len(smiles)
        self.querylenuplimit = 1.3*self.querymollen
        self.querylenlowlimit = 0.7*self.querymollen
        len_desirability = [{"x": 0, "y": 0}, {"x": self.querylenlowlimit/4, "y": 0.4}, {"x": self.querylenlowlimit/3, "y": 0.3}, 
            {"x": self.querylenlowlimit, "y": 0.8}, {"x": self.querymollen, "y": 1.0}, {"x": self.querylenuplimit , "y": 0.8,}, {"x": self.querylenuplimit*1.4, "y": 0.4}, {"x": self.querylenuplimit*1.8, "y": 0.0}]
        

class PenalizedLogPReward(Reward):
    def __init__(self, *args, **kwargs):
        super(PenalizedLogPReward, self).__init__(*args, **kwargs)
        self.vmin = -100
        return

    def reward(self, mol):
        """
            This code is obtained from https://drive.google.com/drive/folders/1FmYWcT8jDrwZlzPbmMpRhulb9OKTDWJL
            , which is a part of GraphAF program done by Chence Shi.
            Reward that consists of log p penalized by SA and # long cycles,
            as described in (Kusner et al. 2017). Scores are normalized based on the
            statistics of 250k_rndm_zinc_drugs_clean.smi dataset
            :param mol: rdkit mol object
            :return: float
            """
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455
        
        plogpscore = -50
        if mol is not None:
            try:
                log_p = Descriptors.MolLogP(mol)
                SA = -calculateScore(mol)
                
                # cycle score
                cycle_list = nx.cycle_basis(nx.Graph(
                    Chem.rdmolops.GetAdjacencyMatrix(mol)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([len(j) for j in cycle_list])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                cycle_score = -cycle_length

                normalized_log_p = (log_p - logP_mean) / logP_std
                normalized_SA = (SA - SA_mean) / SA_std
                normalized_cycle = (cycle_score - cycle_mean) / cycle_std
                pre_score = normalized_log_p + normalized_SA + normalized_cycle
                #score = 1/(1+math.exp(-pre_score))
                plogpscore = (pre_score-(-50))/(50 - (-50))
                scoredict = {'nlogp': normalized_log_p, 'nsa': normalized_SA, 'ncycle':normalized_cycle, 'plogpscore':plogpscore}
                
                
            except ValueError:
                score = self.vmin
                scoredict = {'nlogp': 0, 'nsa': 0, 'ncycle':0, 'plogpscore':plogpscore}
        else:
            score = self.vmin
            scoredict = {'nlogp': 0, 'nsa': 0, 'ncycle':0, 'plogpscore':plogpscore}

        return scoredict


class QEDReward(Reward):
    def __init__(self, *args, **kwargs):
        super(QEDReward, self).__init__(*args, **kwargs)
        self.vmin = 0
        
    def reward(self, mol):
        try:
            if mol is not None:
                r_value = AllChem.EmbedMolecule( mol, useExpTorsionAnglePrefs=True , useBasicKnowledge=True )
                if r_value == 0:
                    try:
                        Chem.SanitizeMol(mol)
                        score = QED.qed(mol)
                    except:
                        print("not sanitizable")
                        score=-1
                else: 
                    score = -1
            else:
                score = -1
        except:
            score = -1

        return score

class MultiReward(Reward):

    """
    maccskeys = deepchem.feat.MACCSKeysFingerprint()
    circular = deepchem.feat.CircularFingerprint()
    mol2vec = deepchem.feat.Mol2VecFingerprint()
    mordred = deepchem.feat.MordredDescriptors(ignore_3D=True)
    rdkit = deepchem.feat.RDKitDescriptors()
    pubchem = deepchem.feat.PubChemFingerprint()
    """
    
    clf=[None] * 5

    def __init__(self, *args, **kwargs):
        super(MultiReward, self).__init__(*args, **kwargs)
        self.vmin = -100
        self.QEDRewardobj = QEDReward()
        self.PenalizedLogPRewardobj = PenalizedLogPReward()
        for index in range(5):
            self.clf[index] = xgb.XGBRegressor()
            self.clf[index].load_model(hydra.utils.get_original_cwd()+"/data/models/featurize/model"+str(index)+".json")
        self.caco2_obj=caco2()

    def MolWt(self,mol):
        molwt = 492.5

        if mol is not None:
            x = 500.66 #Chem.Descriptors.MolWt(mol)
            #x = float(wt)
    
            score = float(np.piecewise(float(x)
                            , [x < molwt*0.4
                            , ((x >= molwt*0.4) & (x < molwt*0.7))
                            , ((x >= molwt*0.7) & (x < molwt*1.3))
                            , ((x >= molwt*1.3) & (x < molwt*2.0))
                            , x >= molwt*2.0]
                            , [0,0.5,1.0,0.5,-1]))
        else:
            score = -1 

        return score

    def penalize_macrocycles(self,mol):
        if mol is not None:
            score = 1
            ri = mol.GetRingInfo()
            for x in ri.AtomRings():
                if len(x) > 8:
                    score = 0
                    break
        else:
            score = -1 
        return score

    def pains(self,mol):
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        if mol is not None: 
            entry = catalog.GetFirstMatch(mol)
            if entry is None:
                score = 1
            else:
                score = 0 
        else:
            score = -1
        return score

    def tan_sim(self,mol,ref_mol):
        if mol is not None: 
            fp_mol = AllChem.GetMorganFingerprint(mol, 2)
            fp_ref = AllChem.GetMorganFingerprint(ref_mol, 2)
            score = DataStructs.TanimotoSimilarity(fp_ref, fp_mol)
        
        else:
            score = -1
        return score


    def caco2_2(self, mol):

        ret_status, caco2_score_val=self.caco2_obj.evaluate_caco2(mol)
        cacoscore=1
        if ret_status==False or  caco2_score_val < -5.15:
            cacoscore=-1

        scoredict = {'cacoscore': cacoscore, 'caco': caco2_score_val}
        return scoredict


    def caco2_1(self,mol):

        name = "caco2_wang"
        if mol is not None: 
            maccskeys_test=None
            circular_test=None
            mol2vec_test=None
            mordred_test=None
            rdkit_test=None
            pubchem_test=None
            try:
                maccskeys_test = self.maccskeys._featurize(mol)
                circular_test = self.circular._featurize(mol)
                mol2vec_test = self.mol2vec._featurize(mol)
                mordred_test = self.mordred._featurize(mol)
                rdkit_test = self.rdkit._featurize(mol)
                pubchem_test=None
                pubchem_test = np.array([0] * 881)

                #try:
                #    pubchem_test = self.pubchem._featurize(mol)
                #except:
                #    pubchem_test = np.array([0] * 881)
            except:
                scoredict = {'cacoscore': -1, 'caco': -100}
                print("caco2 exception")
                return scoredict

            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(maccskeys_test, arr)

            maccskeys_test=np.array([arr])
            circular_test=np.array([circular_test])
            mol2vec_test=np.array([mol2vec_test])
            mordred_test=np.array([mordred_test])
            rdkit_test=np.array([rdkit_test])
            pubchem_test=np.array([pubchem_test])            

            # combine features
            fp_test = np.concatenate(
                (
                    maccskeys_test, circular_test, mol2vec_test,
                    rdkit_test, mordred_test, pubchem_test
                ), axis=1
            )
            print(maccskeys_test.shape)
            print(circular_test.shape)
            print(mol2vec_test.shape)
            print(rdkit_test.shape)
            print(mordred_test.shape)
            print(pubchem_test.shape)
            print(fp_test.shape)
            # convert nan to 0
            fp_test = np.nan_to_num(fp_test, nan=0, posinf=0)
            #if fp_test.shape[1] < 5217:
            #    print("fp_test.shape[1] < 5217 {}", fp_test.shape[1])
            #    scoredict = {'cacoscore': -1, 'caco': -100}
            #    return scoredict
        
            # save to npy
            #np.save(open("./smiles.npy", "wb"), fp_test)
            #fp_test = np.load(open("./smiles.npy", "rb"))

            predictions_list = []
            feature_imp_list = []

            for index in range(5):       
                pred_xgb = self.clf[index].predict(fp_test)
                # add to predicitons dict
                predictions = {}
                predictions[name] = pred_xgb
                predictions_list.append(predictions)
                # get feature importance
                feature_imp_list.append(self.clf[index].feature_importances_)

            caco = (predictions_list[0][name][0]+ predictions_list[1][name][0]+predictions_list[2][name][0]+predictions_list[3][name][0]+predictions_list[4][name][0])/5
            cacofail = -1
            cacopass = 1
            if caco < -5.15:
                scoredict = {'cacoscore': cacofail, 'caco': caco}
            else:
                scoredict = {'cacoscore': cacopass, 'caco': caco}
        else:
            print("cocoa mol is None")

            scoredict = {'cacoscore': -1, 'caco': -100}

        return scoredict


    def reward(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scoredict = {'score': -1, 'qed': -1, 'plogp':0, 'n_sa':0, 'n_cycle': 0, 'nlogp': -100, 'pains': -1, 'tan_sim': -1, 'tan_sim_MDP': -1, 
                    'molwt':-1, 'macro':-1, 'cacoscore':-1, 'caco': -100, 'w0':-1, 'w1':-1, 'w2':-1,'w3':-1}
            return scoredict    
        qed = self.QEDRewardobj.reward(mol)
        plogpdict = self.PenalizedLogPRewardobj.reward(mol)
        molwt = self.MolWt(mol)
        macro = self.penalize_macrocycles(mol)
        pains = self.pains(mol)
        tan_sim = self.tan_sim(mol,self.querymol)        
        tan_sim_MDP = self.tan_sim(mol,self.templatemol)
        cacodict= self.caco2_2(mol)
        #caco2=1
        score = -1
        if (qed >= 0) & (molwt >= 0):
            score = ((qed*self.weight[0])+(plogpdict['plogpscore']*self.weight[1])+(molwt* self.weight[2])+(cacodict['cacoscore']* self.weight[3]))/sum(self.weight)
            if qed >= 0.5:
                score = score + 1
            
             
        scoredict = {'score': score, 'qed': qed, 'plogp':plogpdict['plogpscore'], 'n_sa': plogpdict['nsa'], 'n_cycle': plogpdict['ncycle'], 'nlogp': plogpdict['nlogp'], 'pains': pains, 'tan_sim': tan_sim, 'tan_sim_MDP': tan_sim_MDP,
                    'molwt':molwt, 'macro': macro, 'cacoscore':cacodict['cacoscore'], 'caco': cacodict['caco'], 'w0':self.weight[0], 'w1':self.weight[1], 'w2':self.weight[2],'w3':self.weight[3]}
        
        if self.printindex == 0:
            print(scoredict)
            self.printindex = 100
        self.printindex = self.printindex -1 
        return scoredict


        