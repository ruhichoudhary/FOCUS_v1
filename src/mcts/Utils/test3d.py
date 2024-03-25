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
 
#smiles = 'C1(=C(C=CC(=C1[C@H](OC2=C(N=CC(=C2)C3=C[N](N=C3)C4CCNCC4)N)C)Cl)F)Cl'
#smiles = 'c1nn(Cc2ccccc2)cc1-c3ccccc3'
#smiles = 'CC(=O)N[C@H]1[C@H](c2)[C@@H]([C@H]2C)[C@@H]1O[C@H](C(c1)C1(CC)CCC(=O)O)C(N)=O'
#smiles = 'CC(=O)N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H](C)C(=O)N[C@@H](C)C(=O)N[C@H](CCC(=O)O)C(N)=O'
smiles = 'CCCC1O[C@@H]2C[C@H]3[C@@H]4CCC5=CC(=O)C=C[C@]5(C)[C@H]4[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1'
#smiles = 'CC(C(=O)NC(CCC(=O)O)C(=O)N)NC(=O)C(C)OC1C(C(OC(C1O)CO)O)NC(=O)C'
#smiles  ='CC(=O)N[C@H]1[C@H](c2ccc(Cl)cc2)[C@@H]1O[C@H](/[C@])C(=O)N([C@])C(N)=O'


mol = Chem.MolFromSmiles(smiles)

try:

    if mol is not None:
        r_value = AllChem.EmbedMolecule( mol)#useExpTorsionAnglePrefs=True , useBasicKnowledge=True

        if r_value == 0:
            print("Good Conformer ID")
            print(AllChem.ComputeMolVolume(mol))
        else:
            print('Bad Conformer ID')
            print(AllChem.ComputeMolVolume(mol))
    else:
        print("fail")

except:
    print("either molecule is bad or conformer is bad")
