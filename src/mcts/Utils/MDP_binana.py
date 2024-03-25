import pandas as pd
import json
import pandas as pd
import MDAnalysis as mda
import prolif as plf
import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from rdkit import Chem
import math
from rdkit import Geometry
from prolif.plotting.network import LigNetwork
from prolif.utils import get_residues_near_ligand, to_bitvectors, to_dataframe

if __name__ == '__main__':
        freeze_support()
        with open('/Users/ruhic/code/mermaid_RL_1/Data/binana_c1.json', 'r') as f:
            data = json.load(f)
            del data['closestContacts']
            del data['closeContacts']
            del data['activeSiteFlexibility']
            del data['electrostaticEnergies']
            del data['ligandAtomTypes']
            del data['metalCoordinations']
            df = pd.DataFrame()
            for key, values in data.items():
                values = data[key]
                if values is not None:
                    ligindex = []
                    resID = []
                    resname = []
                    dist = []
                    interaction = []
                    df1 = pd.DataFrame(values)
                    if df1.empty:
                        del(df1)
                    else:
                        for i in df1['receptorAtoms']:
                            resID.append(i[0]['resID'])
                            resname.append(i[0]['resName'])
                            mykey = key
                            if mykey == 'hydrophobicContacts':
                                mykey = 'Hydrophobic'
                            elif mykey == 'hydrogenBonds':
                                mykey = 'HBDonor'
                            elif mykey == 'saltBridges':
                                mykey = 'Cationic'
                            elif mykey == 'cationPiInteractions':
                                mykey = 'CationPi'
                            elif mykey == 'piPiStackingInteractions':
                                mykey = 'PiStacking'
                            elif mykey == 'tStackingInteractions':
                                mykey = 'EdgeToFace'
                            interaction.append(mykey)
                        for j in df1['ligandAtoms']:
                            ligindex.append(j[0]['atomIndex'])
                        for k in df1['metrics']:
                            dist.append(k['distance'])
                        df1['ligindex'] = ligindex
                        df1['resID'] = resID
                        df1['resname'] = resname
                        df1['interaction'] = interaction
                        df1['distance'] = dist
                        df1['atomindex'] =0 
                        df1['ligandID'] = 'UNL1'
                        df1 = df1.drop(columns = ['ligandAtoms', 'receptorAtoms', 'metrics'])
                        df1 = df1.sort_values(['distance', 'resname'])
                        df1 = df1.drop_duplicates(subset='resID')
                        df1['Residue'] = df1['resname'] + df1['resID'].astype(str) + '.A'
                        df1 = df1.drop(columns=['resID', 'resname', 'distance'])
                        #print(df)
                        df1['Frame'] = list(zip(df1.ligindex, df1.atomindex))
                        df1 = df1.drop(columns = ['ligindex', 'atomindex'])
                        res_list = df1['Residue'].tolist()
                        int_list = df1['interaction'].tolist()
                        UNL_list = df1['ligandID'].tolist()
                        header = [UNL_list, res_list,int_list]
                        ligand = df1['Frame'].tolist()
                        df2 = pd.DataFrame([ligand])
                        df2.columns=header
                        df2.index.names = ['Frame']
                        df2.columns.names = ['ligand', 'protein', 'interaction']
                        # load ligands
                        print(df2)
                        if df.empty:
                            df = df2
                        else:
                            df = df.join(df2)
            # load ligands
            #path = str("/Users/ruhic/code/mermaid_RL_1/Data/docking_poses_1.sdf")
            #lig_suppl = plf.sdf_supplier(path)
            #lig = lig_suppl[0]
            print(df)
            #net = LigNetwork.from_ifp(masterdf, lig,kind="aggregate", threshold=.3,rotation=270)
            #net.show("check.html")
            
            


                   
                    
                    

                    


