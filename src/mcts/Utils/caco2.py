import os
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import DataStructs
from deepchem import deepchem
import xgboost as xgb
import hydra



class caco2(object):

    clf=[None] * 5

    def __init__(self):
        self.maccskeys = deepchem.feat.MACCSKeysFingerprint()
        self.circular = deepchem.feat.CircularFingerprint()
        data_path=hydra.utils.get_original_cwd()
        #data_path="/Users/ruhichoudhary/code/gitrepos/DD_Reinforcement4"
        self.mol2vec = deepchem.feat.Mol2VecFingerprint(data_path+"/data/models/featurize/mol2vec_model_300dim.pkl")
        self.mordred = deepchem.feat.MordredDescriptors(ignore_3D=True)
        self.rdkit = deepchem.feat.RDKitDescriptors()
        self.pubchem = deepchem.feat.PubChemFingerprint()
        for index in range(5):
            self.clf[index] = xgb.XGBRegressor()
            model_path=data_path+"/data/models/featurize/model"+str(index)+".json"
            #print(model_path)
            self.clf[index].load_model(model_path)


    def evaluate_caco2(self, mol):
        name = "caco2_wang"
        maccskeys_test=None
        circular_test=None
        mol2vec_test=None
        mordred_test=None
        rdkit_test=None
        pubchem_test=None
        ret_status=True
        caco2_score=0
        try:
            maccskeys_test = self.maccskeys._featurize(mol)
            circular_test = self.circular._featurize(mol)
            mol2vec_test = self.mol2vec._featurize(mol)
            mordred_test = self.mordred._featurize(mol)
            rdkit_test = self.rdkit._featurize(mol)
            try:
                arr=self.pubchem._featurize(mol)
                if arr.shape[0] == 0:
                    arr = np.array([0] * 881)
                pubchem_test=arr
            except Exception as e:
                print(e)
                pubchem_test=np.array([0]*881)
            #try:
            #    pubchem_test = self.pubchem._featurize(mol)
            #except:
            #    pubchem_test = np.array([0] * 881)
        except Exception as e:
            print(e)
            ret_status=False
            return ret_status,caco2_score

        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccskeys_test, arr)

        maccskeys_test=np.array([arr])
        circular_test=np.array([circular_test])
        mol2vec_test=np.array([mol2vec_test])
        mordred_test=np.array([mordred_test])
        rdkit_test=np.array([rdkit_test])
        pubchem_test=np.array([pubchem_test])    

        """
        print(maccskeys_test.shape)
        print(circular_test.shape)
        print(mol2vec_test.shape)
        print(mordred_test.shape)
        print(rdkit_test.shape)
        """

        # combine features
        fp_test = np.concatenate(
            (
                maccskeys_test, circular_test, mol2vec_test,
                rdkit_test, mordred_test, pubchem_test
            ), axis=1
        )
        #print(fp_test.shape)
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

        caco2_score = (predictions_list[0][name][0]+ predictions_list[1][name][0]+predictions_list[2][name][0]+predictions_list[3][name][0]+predictions_list[4][name][0])/5
        return ret_status, caco2_score


if __name__ == "__main__":
    caco2_object=caco2()
    mol = Chem.MolFromSmiles("CCCC(C(=O)C1=CC=C(C=C1)C)NC")
    print(caco2_object.evaluate_caco2(mol))

