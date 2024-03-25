
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


def getpeplist(jsonfile):
    with open( jsonfile, 'r') as f:
        data = json.load(f)
        count = []
        result_dict={}

        count1 = 0
        d1 = data['hydrogenBonds']
        for i in d1:
            d = (i['receptorAtoms'][0]['resName'])
            if d == 'ASN':
                count1 = count1 + 1
        count.append(count1)
        result_dict['PEP_hydrogenBonds']=count1


        count2 = 0
        d2 = data['closeContacts']
        for i in d2:
            d = (i['receptorAtoms'][0]['resName'])
            n = (i['receptorAtoms'][0]['resID'])
            if (d == 'GLU')  & (n == 595):
                count2 = count2 + 1
        count.append(count2)
        result_dict['PEP_closeContacts']=count2

        print(count)
        print(result_dict)
        return result_dict




