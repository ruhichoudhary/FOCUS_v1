#import MDAnalysis as mda
#import prolif as plf
import numpy as np
import pandas as pd
import json
import pandas as pd
import pandas as pd
import numpy as np
from multiprocessing import freeze_support
from rdkit import Chem
import math
from rdkit import Geometry
#from prolif.plotting.network import LigNetwork
#from prolif.utils import get_residues_near_ligand, to_bitvectors, to_dataframe

def getmctlist(jsonfile):
    with open(jsonfile, 'r') as f:
        data = json.load(f)
        count = []
        count1 = 0
        count2 = 0 
        d1 = data['hydrogenBonds']
        for i in d1:
            d_n = (i['receptorAtoms'][0]['resName'])
            d_i = (i['receptorAtoms'][0]['resID'])
            if (d_n == 'LYS'):
                count1 = count1 + 1
            if (d_n == 'SER'):
                count1 = count1 + 1
            if (d_n == 'ARG'):
                count1 = count1 + 1
        count.append(count1)
        d2 = data['hydrophobicContacts']
        count2 = len(d2)
        count.append(count2)
        #print(count)       
        return count