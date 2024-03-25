import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
src_root_dir = os.getcwd()


import json
from config.config import cs
import hydra
from omegaconf import DictConfig
from smina.runsmina import run_smina
import threading
from smina.interaction_mct import getmctlist
from smina.interaction_pep import getpeplist
from smina.interaction_nod2 import getnod2list


class SMINA_data(object):

    """
    A class to manage SMINA docking data.
    
    Attributes:
        output_dir_dict (dict): Stores the output directory paths.
        ligand_smile (str): The SMILES representation of the ligand.
        output_base_dir (str): The base directory for output files.
        index (int): An index used for organizing output directories.
        original_dir (str): The original directory of the project.
    """

    #result_empty_dict={'NOD2': [0, 0, 0, 0, 0, 0, 0, 0], 'MCT1': [0, 0], '7pn1': [0, 0]}

    result_empty_dict = {
            'NOD2_hydrophobicContacts': 0,
            'NOD2_hydrogenBonds':0,
            'NOD2_halogenBonds':0,
            'NOD2_piPiStackingInteractions':0,
            'NOD2_tStackingInteractions':0,
            'NOD2_cationPiInteractions':0,
            'NOD2_saltBridges':0,
            'NOD2_electrostaticEnergies':0,
            'MCT1_hydrogenBonds':0,
            'MCT1_hydrophobicContacts':0,
            'PEP_hydrogenBonds':0,
            'PEP_closeContacts':0,
            'NOD2_score':0,
            'MCT1_score':0,
            '7pn1_score':0,
    }

    #for test
    receptor_names_one={
        'NOD2': { 
            'pocket_center' : ['48.688555823432075', '58.76977835761176', '111.3519999186198'],
            'pocket_size' : ['20', '20', '20'],
            'get_data_fn' : getnod2list
            }
    }

    receptor_names={
        'NOD2': { 
            'pocket_center' : ['48.688555823432075', '58.76977835761176', '111.3519999186198'],
            'pocket_size' : ['20', '20', '20'],
            'get_data_fn' : getnod2list
            },
        'MCT1': {
            'pocket_center' : ['108.029499',	'112.36800',	'106.02998'],
            'pocket_size' : ['25', '25', '25'],
            'get_data_fn' : getmctlist
        },
        '7pn1': {
            'pocket_center': ['100.366',	'102.7385',	'109.2055'],
            'pocket_size' : ['25', '25', '25'],
            'get_data_fn' : getpeplist
        }
    }

    def __init__(self, output_base_dir, original_dir) -> None:
        self.output_dir_dict={}  #all full directory paths of output are stored here
        self.ligand_smile=""
        self.output_base_dir=output_base_dir
        self.index=0
        self.original_dir=original_dir #current directory of the project. The outputs directory is in this path

        """Function runs as a worker thread."""
    def worker(self, ligand_smile, receptor_name, output_dir, original_dir):
        pocket_center=self.receptor_names[receptor_name]['pocket_center']
        pocket_size=self.receptor_names[receptor_name]['pocket_size']
        try:
            print("run_smina "+ligand_smile+' '+receptor_name)
            run_smina(ligand_smile, receptor_name, output_dir, original_dir, pocket_center, pocket_size)
        except Exception as e:
            print(e)


    def run_smina_processes(self):
        threads = []

        for i, receptor_name in enumerate(self.receptor_names):

            """
            self.output_dir_dict[receptor_name]=self.output_base_dir+'/smina_output/'+str(self.index)+'/'+receptor_name
            pocket_center=self.receptor_names[receptor_name]['pocket_center']
            pocket_size=self.receptor_names[receptor_name]['pocket_size']
            run_smina(self.ligand_smile, receptor_name, self.output_dir_dict[receptor_name], self.original_dir, pocket_center, pocket_size)
            """
            self.output_dir_dict[receptor_name]=self.output_base_dir+'/smina_output/'+str(index)+'/'+receptor_name
            t = threading.Thread(target=self.worker, args=(self.ligand_smile, receptor_name
                                    , self.output_dir_dict[receptor_name], self.original_dir))
            threads.append(t)
            t.start()
         # Wait for all of them to finish
        for x in threads:
            x.join()
        
    def run_smina_n_processes(self, smiles, start_index):
        """
        Runs SMINA docking processes for a set of ligands.

        Args:
            smiles (list): A list of SMILES strings for the ligands.
            start_index (int): The starting index for output organization.

        Returns:
            int: The number of processed smiles.
        """
        threads = []

        smile_count = len(smiles)
        if smile_count > 6:
            smile_count = 6
        smiles_processed = 0
        cpu_count=1
        index=start_index
        while smiles_processed < smile_count:
            for i in range(cpu_count):
                
                print("Processing smile {} of {}".format(str(smiles_processed), str(smile_count)))

                if smiles_processed >= smile_count:
                    break

                for i, receptor_name in enumerate(self.receptor_names):
                    output_dir=self.output_base_dir+'/smina_output/'+str(index)+'/'+receptor_name
                    t = threading.Thread(target=self.worker, args=(smiles[smiles_processed], receptor_name
                                            , output_dir, self.original_dir))
                    threads.append(t)
                    t.start()
                # Wait for all of them to finish
                smiles_processed=smiles_processed+1
                index=index+1
            for x in threads:
                x.join()
        
        return smile_count

    def get_data(self, output_dir, index):
        """
        Retrieves docking data from output files.

        Args:
            output_dir (str): The base output directory.
            index (int): The index for the specific output to retrieve.

        Returns:
            dict: A dictionary containing the retrieved docking data.
        """
        results_dict={}
        temp_dict={}
        output_base_path=output_dir+'/smina_output/'+str(index)
        print("output file {}".format(output_base_path))
        for i, receptor_name in enumerate(self.receptor_names):
            output_path=output_base_path+'/'+receptor_name
            json_file=output_path+'/output.json'
            docking_json=output_path+'/docking_result.json'
            if os.path.exists(json_file) and os.path.exists(json_file):
                temp_dict=self.receptor_names[receptor_name]['get_data_fn'](json_file)
                for index, key in enumerate(temp_dict):
                    results_dict[key]=temp_dict[key]
                with open(docking_json, 'r') as f:
                    data = json.load(f)
                    results_dict[receptor_name+'_score']=data['docking_factor']
            else:
                return self.result_empty_dict
        return results_dict

@hydra.main(config_path="../config/", config_name="config")
def run_smina_main(cfg: DictConfig):
    #ligand_smile='CC(C(\C)(-c1ccc(C)cc1))([C@])OC1C(C(OC(C1O)([S@](=O)(=O))))C'  #gives all zeroes
    #ligand_smile='OC[C@H]1O[C@@H](O)[C@@H]([C@H]([C@@H]1O)O[C@@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N)CCC(=O)O)C)C)NC(=O)C' #
    ligand_smile='OCC1OC(O)C(C(C1O)OC(C(=O)NC(C(=O)NC(C(=O)N)CCC(=O)O)C)C)NC(=O)C'

    output_base_dir=os.getcwd()
    smina_data_object=SMINA_data(ligand_smile, 100, output_base_dir, hydra.utils.get_original_cwd())
    smina_data_object.run_smina_processes()
    print(smina_data_object.get_data())

if __name__ == "__main__":
    run_smina_main()


