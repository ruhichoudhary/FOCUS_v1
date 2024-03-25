
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join('/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/src'))

from pathlib import Path
import subprocess
from openbabel import openbabel
from openbabel import pybel
from rdkit import Chem

import hydra
from omegaconf import DictConfig
from config.config import cs
import pandas as pd
import json
import shutil


class SMINA_Docker(object):

    def __init__(self, ligand_smile, receptor_name, output_dir, originaL_dir, pocket_center, pocket_size) -> None:
        self.ligand_smile=ligand_smile
        self.receptor_name=receptor_name
        self.output_dir=output_dir
        self.original_dir=originaL_dir
        self.pocket_center=pocket_center
        self.pocket_size=pocket_size
        receptor_pdb = receptor_name+'.pdb'
        receptor_pdbqt = receptor_name+'.pdbqt'
        self.pdb_dir = originaL_dir+'/data/pdb'
        self.pdbqt_dir = originaL_dir+'/data/pdbqt'
        self.receptor_pdb_path = self.pdb_dir + '/' + receptor_pdb
        self.receptor_pdbqt_path = self.pdbqt_dir + '/' + receptor_pdbqt
        self.ligand_pdbqt_path = output_dir+"/ligand.pdbqt"
        os.makedirs(self.pdbqt_dir,exist_ok=True)
        os.makedirs(self.output_dir,exist_ok=True)
        self.docking_poses_file = self.output_dir + '/docking_poses.sdf'



    def run_binana(self):
        vina_ligands_path=self.original_dir+'/smina/vina_ligands.py'
        cmd = 'cd '+self.output_dir+';python '+vina_ligands_path  #run the file in output dir
        print(cmd)
        os.system(cmd)
        ligand_path='./ligand.pdbqt'
        binana_path=self.original_dir+'/smina/run_binana.py'
        cmdstr='cd '+self.output_dir+';python3 '+binana_path+' -receptor '+self.receptor_pdbqt_path+'  -ligand docking_poses_1.pdbqt -output_dir . > output'+self.receptor_name+'.txt'
        print(cmdstr)
        os.system(cmdstr)

    def pdb_to_pdbqt(self, pdb_path, pdbqt_path, pH=7.4):
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

    # obabel method - not being used
    def smiles_to_pdbqt_obabel(self, smiles, sdf_path, pdbqt_path):
        mol = Chem.MolFromSmiles(smiles)
        pm = Chem.PropertyMol.PropertyMol(mol)
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        print('Done creating SDF files')
        os.system('obabel '+sdf_path+'/ligand.sdf -opdb -h -m')
        os.system('obabel ligand.pdb -opdbqt -xh -m')

    def smiles_to_pdbqt(self, smiles, pdbqt_path, pH=7.4):
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
    
    def split_sdf_file(self, sdf_path):
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



    def run_smina_process(self, 
        ligand_path, protein_path, sdf_path, num_poses=10, exhaustiveness=10
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
        output_text=""
        smina_commandline = [
                    self.original_dir+"/smina/smina.osx",
                    "--ligand",
                    str(ligand_path),
                    "--receptor",
                    str(protein_path),
                    "--out",
                    str(sdf_path),
                    "--log",
                    str(self.output_dir+'/smina.log'),
                    "--center_x",
                    str(self.pocket_center[0]),
                    "--center_y",
                    str(self.pocket_center[1]),
                    "--center_z",
                    str(self.pocket_center[2]),
                    "--size_x",
                    str(self.pocket_size[0]),
                    "--size_y",
                    str(self.pocket_size[1]),
                    "--size_z",
                    str(self.pocket_size[2]),
                    "--num_modes",
                    str(num_poses),
                    "--exhaustiveness",
                    str(exhaustiveness),
                    "--cpu",
                    str(8),
                    "--atom_term_data",
                    "--atom_terms",
                    "atom_terms.csv" 
                ]

        try:
            print(smina_commandline)
            output_text = subprocess.check_output(
                smina_commandline,
                universal_newlines=True,  # needed to capture output text
                cwd=self.output_dir,
                shell=False
            )
        except subprocess.CalledProcessError as e:
            print("Oops... returncode: " + e.returncode + ", output:\n" + e.output)
            output_text=e.output
        else:
            print("Everything ok:\n" + output_text)
        return output_text
    
    def getsminadata(self):

        smina_run_dir=self.output_dir


        docking_factor=99999
        try:
            output_text = self.run_smina_process(
                self.ligand_pdbqt_path,
                self.receptor_pdbqt_path,
                self.docking_poses_file,
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
            self.split_sdf_file(self.docking_poses_file)
            docking_factor=float(l[1])
            print("================>>>> "+str(docking_factor))
        except Exception as e: 
            print(e)
            return docking_factor
        docking_result = {
            "docking_factor": docking_factor
        }
        json_str=json.dumps(docking_result)
        print(json_str)
        with open(self.output_dir+'/docking_result.json', 'w', encoding='utf-8') as f:
            json.dump(docking_result, f, ensure_ascii=False, indent=4)
        self.run_binana()  
        #os.chdir(current_dir)
        return docking_factor

def run_smina(ligand_smile, receptor_name, output_dir, originaL_dir, pocket_center, pocket_size):

    #print("create the object and setup all paths required for the process")
    smina_docker = SMINA_Docker(ligand_smile, receptor_name, output_dir, originaL_dir, pocket_center, pocket_size)
    
    """
    pdbqt files are already placed. 
    WARNING -These commands are generrating empty pdbqt - need to be checked out
    # convert the pdb into PDBQT format if it does not exist
    if not os.path.exists(smina_docker.receptor_pdbqt_path):
        obabel_cmd_str='obabel {} -O {} -xh'.format(smina_docker.receptor_pdb_path, smina_docker.receptor_pdbqt_path)
        print(obabel_cmd_str)
        #smina_docker.pdb_to_pdbqt(receptor_pdb_path, receptor_pdbqt_path)
        os.system(obabel_cmd_str)
    """

    #print("convert the ligand into PDBQT format")
    smina_docker.smiles_to_pdbqt(ligand_smile, smina_docker.ligand_pdbqt_path)
    docking_factor=smina_docker.getsminadata()

def main_wrapper(ligand_smile, receptor_name, pocket_center, pocket_size):

    @hydra.main(config_path="../src/config", config_name="config")
    def smina_main(cfg: DictConfig):
        output_dir=os.getcwd()+'/smina_output/'+str(index)+'/'+receptor_name
        run_smina(ligand_smile, receptor_name, output_dir, hydra.utils.get_original_cwd(), pocket_center, pocket_size)

    smina_main()

#"args": ["+arg1=NOD2","+arg2=1","+arg3='CC(C(\\C)(-c1ccc(C)cc1))([C@])OC1C(C(OC(C1O)([S@](=O)(=O))))C'", "+hydra/job_logging=disabled"]
#NOD2, 7pn1 for pep, 
#python smina/runsmina.py +arg1=NOD2 +arg2=1 +arg3="'CC(C(\C)(-c1ccc(C)cc1))([C@])OC1C(C(OC(C1O)([S@](=O)(=O))))C'" +hydra/job_logging=disabled

def convert_pdb_2_pdbqt(pdb_path, pdbqt_path, pH=7.4):
        molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
        # add hydrogens at given pH
        molecule.OBMol.CorrectForPH(pH)
        molecule.addh()
        # add partial charges to each atom
        for atom in molecule.atoms:
            atom.OBAtom.GetPartialCharge()
        molecule.write("pdbqt", str(pdbqt_path), overwrite=True)

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
    #convert_pdb_2_pdbqt("/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/data/pdb/W887A.pdb", "/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/data/pdbqt/NOD2.pdbqt")
    split_sdf_file("/Users/ruhichoudhary/runs_data/smina_output/0/NOD2/docking_poses.sdf")
    exit()
    receptor_name='NOD2'
    index=1
    ligand_smile="CC(C(\C)(-c1ccc(C)cc1))([C@])OC1C(C(OC(C1O)([S@](=O)(=O))))C"
    #ligand_smile="CC(C(c1cc(\C)c(C)cc1)([C@H]([NH3+])))1C(/[C@H]([S@@])(OC(C1(C)[S@](=O)[C@H])))"

    pocket_center = ['48.688555823432075', '58.76977835761176', '111.3519999186198']
    pocket_size = ['20', '20', '20']
 
    try:
        main_wrapper(ligand_smile, receptor_name, pocket_center, pocket_size)
    except subprocess.CalledProcessError as e:
        print(e.output)
    except Exception as e:
        print(e)

#python /Users/ruhic/code/gitrepos/DD_Reinforcement/run_binana.py -receptor NOD2.pdbqt -ligand docking_poses_1.pdbqt -output_dir . > outputNOD2.txt 
#python /Users/ruhic/code/gitrepos/DD_Reinforcement/src/docking/vina_ligands.py

#/Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/smina/smina.osx --ligand /Users/ruhichoudhary/runs_data1/smina_output/246/NOD2/ligand.pdbqt --receptor /Users/ruhichoudhary/code/gitrepos/dd_reinforcement-niagara/data/pdbqt/NOD2.pdbqt --out /Users/ruhichoudhary/runs_data1/smina_output/246/NOD2/docking_poses.sdf --log /Users/ruhichoudhary/runs_data1/smina_output/246/NOD2/smina.log --center_x 48.688555823432075 --center_y 58.76977835761176 --center_z 111.3519999186198 --size_x 20 --size_y 20 --size_z 20 --num_modes 10 --exhaustiveness 10 --atom_term_data --atom_terms atom_terms.csv
    

 