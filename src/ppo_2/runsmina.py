import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import DataStructs
from rdkit.Chem import RDConfig
from rdkit.Chem import rdBase
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from pathlib import Path
import subprocess
from openbabel import openbabel
from openbabel import pybel
#from opencadd.structure.core import Structure
import pandas as pd
import sys
import argparse
import pandas as pd
from rdkit import Chem, RDLogger
import rdkit.Chem.PropertyMol
import hydra

def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4):
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.
 
    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return
    # convert protein to PDBQT format
 

""" If you use MMFF94 you should cite the appropriate papers:

Thomas A. Halgren, J. Comput. Chem., 17, 490-519 (1996).
Thomas A. Halgren, J. Comput. Chem., 17, 520-552 (1996).
Thomas A. Halgren, J. Comput. Chem., 17, 553-586 (1996).
Thomas A. Halgren and Robert B. Nachbar, J. Comput. Chem., 17, 587-615 (1996).
Thomas A. Halgren, J. Comput. Chem., 17, 616-641 (1996). """

def smiles_to_pdbqt(smiles, pdbqt_path, pH=7.4):
    """
    Convert a SMILES string to a PDBQT file needed by docking programs of the AutoDock family.
 
    Parameters
    ----------
    smiles: str
        SMILES string.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    molecule = pybel.readstring("smi", smiles)
    # add hydrogens at given pH (7.4)
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # generate 3D coordinates
    print("Generating 3D coordinates")
    #molecule.make3D(forcefield="mmff94s", steps=1000)  #gives segmentation fault  #https://github.com/openbabel/openbabel/issues/2108
    gen3d = openbabel.OBOp.FindType("gen3D")
    gen3d.Do(molecule.OBMol, "--best")
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return
 
def run_smina(
    ligand_path, protein_path, out_path, pocket_center, pocket_size, num_poses=10, exhaustiveness=10
):
    """
    Perform docking with Smina.
 
    Parameters
    ----------
    ligand_path: str or pathlib.Path
        Path to ligand PDBQT file that should be docked.
    protein_path: str or pathlib.Path
        Path to protein PDBQT file that should be docked to.
    out_path: str or pathlib.Path
        Path to which docking poses should be saved, SDF or PDB format.
    pocket_center: iterable of float or int
        Coordinates defining the center of the binding site.
    pocket_size: iterable of float or int
        Lengths of edges defining the binding site.
    num_poses: int
        Maximum number of poses to generate.
    exhaustiveness: int
        Accuracy of docking calculations.
 
    Returns
    -------
    output_text: str
        The output of the Smina calculation.
    """
    output_text = subprocess.check_output(
        [
            #hydra.utils.get_original_cwd()+"/smina/smina.osx",
            hydra.utils.get_original_cwd()+"/smina/smina.osx",
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--center_x",
            str(pocket_center[0]),
            "--center_y",
            str(pocket_center[1]),
            "--center_z",
            str(pocket_center[2]),
            "--size_x",
            str(pocket_size[0]),
            "--size_y",
            str(pocket_size[1]),
            "--size_z",
            str(pocket_size[2]),
            "--num_modes",
            str(num_poses),
            "--exhaustiveness",
            str(exhaustiveness),
            "--atom_term_data",
            "--atom_terms",
            "atom_terms.csv"
        ],
        universal_newlines=True,  # needed to capture output text
    )
    return output_text
 
""""

def split_sdf_file(sdf_path):
    
    Split an SDF file into seperate files for each molecule.
    Each file is named with consecutive numbers.
 
    Parameters
    ----------
    sdf_path: str or pathlib.Path
        Path to SDF file that should be split.
    
    sdf_path = Path(sdf_path)
    stem = sdf_path.stem
    parent = sdf_path.parent
    molecules = pybel.readfile("sdf", str(sdf_path))
    for i, molecule in enumerate(molecules, 1):
        molecule.write("sdf", str(parent / f"{stem}_{i}.sdf"), overwrite=True)
    return
"""""

 
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)
 
def proteindata():
    # retrieve structure from the Protein Data Bank
    pdb_id = "5irn"
    structure = Structure.from_pdbid(pdb_id)
    # element information maybe missing, but important for subsequent PDBQT conversion
    if not hasattr(structure.atoms, "elements"):
        structure.add_TopologyAttr("elements", structure.atoms.types)
    structure
    # NBVAL_CHECK_OUTPUT

    # write the protein file to disk
    protein = structure.select_atoms("protein")
    protein.write(DATA + "/protein.pdb")
    
    pdb_to_pdbqt(DATA + "/protein.pdb", DATA + "/protein.pdbqt")

def split_sdf_file(sdf_path):
    """
    Split an SDF file into seperate files for each molecule.
    Each file is named with consecutive numbers.

    Parameters
    ----------
    sdf_path: str or pathlib.Path
        Path to SDF file that should be split.
    """
    sdf_path = Path(sdf_path)
    stem = sdf_path.stem
    parent = sdf_path.parent
    molecules = pybel.readfile("sdf", str(sdf_path))
    for i, molecule in enumerate(molecules, 1):
        molecule.write("sdf", str(parent / f"{stem}_{i}.sdf"), overwrite=True)
    return


def write_sdf(m):
    # convert our mols to .sdf files 
    pm = Chem.PropertyMol.PropertyMol(m)
    w = Chem.SDWriter(DATA + '/ligand.sdf')
    w.write(m)


def smiles_to_pdbqt_v2(smiles, pdbqt_path):
    mol = Chem.MolFromSmiles(smiles)
    write_sdf(mol)
    print('Done creating SDF files')
    os.system('obabel '+DATA+'/ligand.sdf -opdb -h -m')
    os.system('obabel ligand.pdb -opdbqt -xh -m')


def getsminadata(config, smiles,DATA = "./docking" ):
    # define ligand SMILES for protein-ligand complex of interest

    # convert the ligand into PDBQT format
    smiles_to_pdbqt(smiles, DATA + "/ligand.pdbqt")

    pdbqtfile=DATA + "/NOD2.pdbqt"

    if not os.path.exists(pdbqtfile) and (config != None):
        pdbfile=hydra.utils.get_original_cwd()+config["smina"]["NOD2"]
        os.system('obabel {} -O {} -xh'.format(pdbfile, pdbqtfile))
        #pdb_to_pdbqt(hydra.utils.get_original_cwd()+config["smina"]["NOD2"], DATA + "/NOD2.pdbqt")

    #pocket_center = ['48.688555823432075', '58.76977835761176', '111.3519999186198']
    #pocket_center = ['44.688555823432075', '58.76977835761176', '111.3519999186198']
    pocket_center = ['48.688555823432075', '58.76977835761176', '111.3519999186198']

    pocket_size = ['20', '20', '20']
    
    try:
        output_text = run_smina(
            DATA + "/ligand.pdbqt",
            DATA + "/NOD2.pdbqt",
            DATA + "/docking_poses.sdf",
            pocket_center,
            pocket_size,
        )
        print(output_text)
        data = output_text.splitlines(True)
        dockcount = data[-3:-2][0].split('       ')
        dockcount = dockcount[0].split('      ')
        dockcount = int(dockcount[0])
        if dockcount < 1:
            return 0.0
        l = data[-1*(dockcount+2):-1*(dockcount+1)][0].split('       ')
        # l = data[-12:-11][0].split('       ')
        split_sdf_file(DATA + "/docking_poses.sdf")
        return float(l[1])
    except Exception as e: 
        print(e)
        return 0

if __name__ == "__main__":
    DATA = "~/runs_data/single"
    #pdb_to_pdbqt(DATA + "/R857A.pdb", DATA + "/R857A.pdbqt")

    print(getsminadata(None, "CC(C(\C)(-c1ccc(C)cc1))([C@])OC1C(C(OC(C1O)([S@](=O)(=O))))C", "/Users/ruhic/runs_data/single"))  #mdp
    #print(getsminadata("CC(C(c1cc(\C)c(C)cc1)([C@H]([NH3+])))1C(/[C@H]([S@@])(OC(C1(C)[S@](=O)[C@H])))"))
    split_sdf_file(DATA + "/docking_poses.sdf")




#python /Users/ruhic/code/gitrepos/DD_Reinforcement/run_binana.py -receptor NOD2.pdbqt -ligand docking_poses_1.pdbqt -output_dir . > outputNOD2.txt 
#python /Users/ruhic/code/gitrepos/DD_Reinforcement/src/docking/vina_ligands.py
    


 