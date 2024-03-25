
import numpy as np
import pandas as pd
import json
import pandas as pd
#import prolif as plf
import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from rdkit import Chem
import math
from rdkit import Geometry
#from prolif.plotting.network import LigNetwork
#from prolif.utils import get_residues_near_ligand, to_bitvectors, to_dataframe

def getnod2list(jsonfile):
        with open( jsonfile, 'r') as f:
                data = json.load(f)
                del data['closestContacts']
                del data['closeContacts']
                del data['activeSiteFlexibility']
                del data['ligandAtomTypes']
                del data['metalCoordinations']
                df = pd.DataFrame()
                interactions = ['hydrophobicContacts', 'hydrogenBonds', 'halogenBonds', 'piPiStackingInteractions', 
                                                        'tStackingInteractions', 'cationPiInteractions', 'saltBridges', 'electrostaticEnergies']
                count = []
                result_dict={}
                df['interactions'] = interactions

                d1 = data['hydrophobicContacts']
                count1 = len(d1)
                count.append(count1)
                result_dict['NOD2_hydrophobicContacts']=count1
                #d2 = data['hydrogenBonds']
                #count2 = len(d2)
                #count.append(count2)


                d2 = data['hydrogenBonds']
                count2 = 0
                for i in d2:
                        d = (i['receptorAtoms'][0]['resName'])
                        print(d)
                        n = (i['receptorAtoms'][0]['resID'])
                        print(n)
                        print(type(n))
                        if (d == 'ARG') & (n == 857):
                                count2 = count2 + 1
                        if (d == 'ARG') & (n == 803):
                                count2 = count2 + 1 
                        if (d == 'TRP') & (n == 911):
                                count2 =  count2 + 1
                        if (d == 'SER')& (n == 913):
                                count2 = count2 +1
                count.append(count2)
                result_dict['NOD2_hydrogenBonds']=count2


                d3 = data['halogenBonds']
                count3 = len(d3)
                count.append(count3)
                result_dict['NOD2_halogenBonds']=count3


                d4 = data['piPiStackingInteractions']
                count4 = len(d4)
                count.append(count4)
                result_dict['NOD2_piPiStackingInteractions']=count4


                d5 = data['tStackingInteractions'] #with TRP911
                count5 = len(d5)
                count.append(count5)
                result_dict['NOD2_tStackingInteractions']=count5


                d6 = data['cationPiInteractions'] 
                count6 = len(d6)
                count.append(count6)
                result_dict['NOD2_cationPiInteractions']=count6

                d7 = data['saltBridges']
                count7 = len(d7)
                count.append(count7)
                result_dict['NOD2_saltBridges']=count7


                d8 = data['electrostaticEnergies']
                count8 = len(d8)
                count.append(count8)
                result_dict['NOD2_electrostaticEnergies']=count8

                print(count)
                print(result_dict)

                return result_dict

if __name__ == "__main__":
        print(getnod2list(""))
        





            
                
                
                   