#import MDAnalysis as mda
#import prolif as plf
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

def getfplist(jsonfile):
        """
        Processes a JSON file containing molecular interaction data and counts the occurrences
        of specific interaction types.

        The function reads a JSON file, removes certain keys, and then counts the occurrences
        of various interaction types, including hydrophobic contacts, hydrogen bonds, and more.

        Parameters:
        - jsonfile (str): The path to the JSON file containing the interaction data.

        Returns:
        - list: A list of counts corresponding to each interaction type.
        """
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
                df['interactions'] = interactions

                d1 = data['hydrophobicContacts']
                count1 = len(d1)
                count.append(count1)
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
                d3 = data['halogenBonds']
                count3 = len(d3)
                count.append(count3)
                d4 = data['piPiStackingInteractions']
                count4 = len(d4)
                count.append(count4)
                d5 = data['tStackingInteractions'] #with TRP911
                count5 = len(d5)
                count.append(count5)
                d6 = data['cationPiInteractions'] 
                count6 = len(d6)
                count.append(count6)
                d7 = data['saltBridges']
                count7 = len(d7)
                count.append(count7)
                d8 = data['electrostaticEnergies']
                count8 = len(d8)
                count.append(count8)
                #print(count)

                return count

if __name__ == "__main__":
        print(getfplist(""))
        





            
                
                
                   