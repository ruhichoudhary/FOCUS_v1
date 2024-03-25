    
    
import sys
import os

from pathlib import Path
import subprocess
from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem
import pandas as pd
import json
import shutil
    

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
    #molecule = Chem.rdmolops.AssignStereochemistry(molecule, force=True) - convert fron pybel
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

if __name__ == "__main__":
    df = pd.read_csv("/Users/ruhichoudhary/ruhi_docking/ruhi_smile_list.csv")
    print(df['SMILES'])
    for i in range(len(df['SMILES'])):
        dir_name="/Users/ruhichoudhary/ruhi_docking/{}".format(i)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        dir_name_NOD2=dir_name+"/NOD2"
        if not os.path.exists(dir_name_NOD2):
            os.makedirs(dir_name_NOD2)
        dir_name_PEP=dir_name+"/PEP"
        if not os.path.exists(dir_name_PEP):
            os.makedirs(dir_name_PEP)
        dir_name_MCT1=dir_name+"/MCT1"
        if not os.path.exists(dir_name_MCT1):
            os.makedirs(dir_name_MCT1)

        print("\n\n----------------- "+str(i)+" -----------------")
        smiles_to_pdbqt(df.iloc[i]['SMILES'], dir_name+"/ligand.pdbqt")
        #os.system("/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/smina/smina.osx --ligand {}/ligand.pdbqt --receptor /Users/ruhichoudhary/ruhi_docking/NOD2.pdbqt --out {}/docking_poses.sdf --log {}/smina.log --center_x 48.688555823432075 --center_y 58.76977835761176 --center_z 111.3519999186198 --size_x 20 --size_y 20 --size_z 20 --num_modes 10 --exhaustiveness 10 --atom_term_data --atom_terms atom_terms.csv --cpu 4".format(dir_name, dir_name_NOD2, dir_name_NOD2))
        #split_sdf_file("{}/docking_poses.sdf".format(dir_name_NOD2))
        os.system("/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/smina/smina.osx --ligand {}/ligand.pdbqt --receptor /Users/ruhichoudhary/ruhi_docking/MCT1.pdbqt --out {}/docking_poses.sdf --log {}/smina.log --center_x 108.029499 --center_y 112.36800 --center_z 106.02998 --size_x 25 --size_y 25 --size_z 25 --num_modes 10 --exhaustiveness 10 --atom_term_data --atom_terms atom_terms.csv --cpu 4".format(dir_name, dir_name_PEP, dir_name_PEP))
        split_sdf_file("{}/docking_poses.sdf".format(dir_name_MCT1))
        os.system("/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/smina/smina.osx --ligand {}/ligand.pdbqt --receptor /Users/ruhichoudhary/ruhi_docking/7pn1.pdbqt --out {}/docking_poses.sdf --log {}/smina.log --center_x 100.366 --center_y 102.7385 --center_z 109.2055 --size_x 25 --size_y 25 --size_z 25 --num_modes 10 --exhaustiveness 10 --atom_term_data --atom_terms atom_terms.csv --cpu 4".format(dir_name, dir_name_MCT1, dir_name_MCT1))
        split_sdf_file("{}/docking_poses.sdf".format(dir_name_PEP))

        #split file
        #os.system("python /Users/ruhic/code/gitrepos/DD_Reinforcement/run_binana.py -receptor /Users/ruhichoudhary/ruhi_docking/NOD2.pdbqt -ligand {}/docking_poses_1.pdbqt -output_dir {} > {}/outputNOD2.txt".format(dir_name, dir_name_NOD2, dir_name_NOD2)) 
        os.system("python /Users/ruhic/code/gitrepos/DD_Reinforcement/run_binana.py -receptor /Users/ruhichoudhary/ruhi_docking/7pn1.pdbqt -ligand {}/docking_poses_1.pdbqt -output_dir {} > {}/output7pn1.txt".format(dir_name, dir_name_PEP, dir_name_PEP)) 
        os.system("python /Users/ruhic/code/gitrepos/DD_Reinforcement/run_binana.py -receptor /Users/ruhichoudhary/ruhi_docking/MCT1.pdbqt -ligand {}/docking_poses_1.pdbqt -output_dir {} > {}/outputMCT1.txt".format(dir_name, dir_name_MCT1, dir_name_MCT1)) 

