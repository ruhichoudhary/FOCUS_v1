
from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import rdMolAlign
#from rdkit.Chem import rdMolDescriptors
#import py3Dmol
#import ipywidgets
#from ipywidgets import interact, interactive, fixed # For interactive display of conformers
 
#from espsim import EmbedAlignConstrainedScore, EmbedAlignScore, ConstrainedEmbedMultipleConfs, GetEspSim, GetShapeSim
from espsim import EmbedAlignScore
#from espsim.helpers import mlCharges
 
#import ehreact
#from ehreact.train import calculate_diagram
#from ehreact.predict.make_prediction import find_highest_template
 
#import urllib.request
#import gzip
#from copy import deepcopy
#from sklearn import metrics
#import numpy as np

refSmiles=['C1=CC=C(C=C1)C(C(=O)O)O','CCC(C(=O)O)O','OC(C(O)=O)c1ccc(Cl)cc1','C1=CC(=CC=C1C(C(=O)O)O)O','COc1ccc(cc1)C(O)C(O)=O','OC(C(O)=O)c1ccc(cc1)[N+]([O-])=O','CCCC(C(=O)O)O','CCC(C)C(C(=O)O)O','CC(C(=O)O)O']
refMols=[Chem.AddHs(Chem.MolFromSmiles(x)) for x in refSmiles]
prbSmile='C(C(C(=O)O)O)O'
prbMol=Chem.AddHs(Chem.MolFromSmiles(prbSmile))
simShape, simEsp = EmbedAlignScore(prbMol, refMols, renormalize=True)
print('%35s %8s %8s %8s' % ("Reference","Shape","ESP", "Shape*ESP"))
for i in range(len(refSmiles)):
    print('%35s %8.2f %8.2f %8.2f' % (refSmiles[i],simShape[i],simEsp[i],simShape[i]*simEsp[i]))