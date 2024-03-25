import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import hydra
from config.config import cs
from omegaconf import DictConfig
import torch
import time
import warnings
import shutil
from datetime import datetime

print(sys.path)

from smina.runsmina_parallel import SMINA_data


warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle

import torch.nn.functional as F

from mcts.Model.model import RolloutNetwork
from Utils.utils import read_smilesset, parse_smiles, convert_smiles, RootNode, ParentNode, NormalNode, \
    trans_infix_ringnumber

from Utils.utils import VOCABULARY
from Utils.reward import getReward

from outerreward import outerreward, currentstate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTS(object):
    def __init__(self, init_smiles, model, vocab, Reward, cfg, max_seq=81, c=1, num_prll=256, limit=5, step=0, n_valid=0,
                 n_invalid=0, sampling_max=False, max_r=-1000):
        self.init_smiles = parse_smiles(init_smiles.rstrip("\n"))
        self.model = model
        self.vocab = vocab
        self.Reward = Reward
        self.max_seq = max_seq
        self.valid_smiles = {}
        self.c = c
        self.count = 0
        self.ub_prll = num_prll
        self.limit = np.sum([len(self.init_smiles)+1-i for i in range(limit)])
        self.sq = set([s for s in self.vocab if "[" in s])
        self.max_score = max_r
        self.step = step
        self.n_valid = n_valid
        self.n_invalid = n_invalid
        self.sampling_max = sampling_max
        self.total_nodes = 0
        self.cfg = cfg

    def select(self):
        raise NotImplementedError()

    def expand(self):
        raise NotImplementedError()

    def simulate(self):
        raise NotImplementedError()

    def backprop(self):
        raise NotImplementedError()

    def search(self, n_step):
        raise NotImplementedError()


class ParseSelectMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        super(ParseSelectMCTS, self).__init__(*args, **kwargs)
        self.root = RootNode()
        self.current_node = None
        self.next_token = {}
        self.rollout_result = {}
        self.l_replace = int(len(self.init_smiles)/4)

    def select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()
                self.current_node = self.root

    def expand(self, epsilon=0.1, loop=10, gamma=0.90):
        """

        """
        # Preparation of prediction using RNN model, list -> tensor
        x = np.zeros([1, self.max_seq])
        c_path = convert_smiles(self.current_node.path[2:], self.vocab, mode="s2i")
        x[0, :len(c_path)] = c_path
        x = torch.tensor(x, dtype=torch.long)
        x_len = [len(c_path)]

        # Predict the probabilities of next token following current node
        with torch.no_grad():
            y = self.model(x, x_len)
            y = F.softmax(y, dim=2)
            y = y.to('cpu').detach().numpy().copy()
            y = np.array(y[0, len(self.current_node.path)-3, :])
            y = np.log(y)
            prob = np.exp(y) / np.sum(np.exp(y))

        # Sampling next token based on the probabilities
        self.next_token = {}
        while len(self.next_token) == 0:
            for j in range(loop):
                if np.random.rand() > epsilon * (gamma ** len(self.current_node.path)):
                    ind = np.random.choice(range(len(prob)), p=prob)
                else:
                    ind = np.random.randint(len(self.vocab))
                self.next_token[self.vocab[ind]] = 0
            if self.current_node.depth == 1:
                self.next_token["("] = 0
        self.check()

        #print("".join(self.current_node.path[2:]), len(self.next_token))
        #print(self.next_token.keys())

    def check(self):
        if "\n" in self.next_token.keys():
            tmp_node = self.current_node
            while tmp_node.depth != 1:
                tmp_node = tmp_node.parent
            original_smiles = tmp_node.original_smiles
            pref, suf = original_smiles.split("*")
            inf = "".join(self.current_node.path[3:])
            smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

            dictkey= self.cfg["mcts"]["reward_key"]

            scoredict = self.Reward.reward(smiles_concat)
            score = scoredict[dictkey]

            self.max_score = max(self.max_score, score)
            self.next_token.pop("\n")
            if score > -100:
                self.valid_smiles["%d:%s" % (-self.step, smiles_concat)] = scoredict
                #print(score, smiles_concat)
                self.max_score = max(self.max_score, score)
                self.n_valid += 1
            else:
                self.n_invalid += 1

        if len(self.next_token) < 1:
            self.current_node.cum_score = -100000
            self.current_node.visit = 100000
            self.current_node.remove_Node()

    def simulate(self):
        tmp_node = self.current_node
        while tmp_node.depth != 1:
            tmp_node = tmp_node.parent
        original_smiles = tmp_node.original_smiles
        pref, suf = original_smiles.split("*")
        self.rollout_result = {}

        #######################################

        l = len(self.current_node.path)
        part_smiles = [[] for i in range(len(self.next_token))]
        x = np.zeros([len(self.next_token), self.max_seq])
        x_len = []
        for i, k in enumerate(self.next_token.keys()):
            part_smiles[i].extend(self.current_node.path[2:])
            part_smiles[i].append(k)
            x[i, :len(part_smiles[i])] = convert_smiles(part_smiles[i], self.vocab, mode="s2i")
            x_len.append(len(part_smiles[i]))
        x = torch.tensor(x, dtype=torch.long)

        is_terminator = [True]*len(self.next_token)
        step = 0

        while np.sum(is_terminator) > 0 and step+l < self.max_seq-1:
            with torch.no_grad():
                y = self.model(x, x_len)
                y = F.softmax(y, dim=2)
                y = y.to('cpu').detach().numpy().copy()
                prob = y[:, step+l-2, :]

            if self.sampling_max:
                ind = np.argmax(prob, axis=1)
            else:
                ind = [np.random.choice(range(len(self.vocab)), p=prob[i]) for i in range(len(self.next_token))]

            for i in range(len(x_len)):
                x_len[i] += 1

            for i in range(len(self.next_token)):
                x[i, step+l-1] = ind[i]
                if is_terminator[i] and ind[i] == self.vocab.index("\n"):
                    is_terminator[i] = False
                    inf = "".join(convert_smiles(x[i, 1:step+l-1], self.vocab, mode="i2s"))
                    smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

                    dictkey= self.cfg["mcts"]["reward_key"]

                    scoredict = self.Reward.reward(smiles_concat)
                    score = scoredict[dictkey]

                    self.next_token[list(self.next_token.keys())[i]] = score
                    self.rollout_result[list(self.next_token.keys())[i]] = (smiles_concat, score)
                    if score > self.Reward.vmin:
                        # self.valid_smiles[smiles_concat] = score
                        self.valid_smiles["%d:%s" % (self.step, smiles_concat)] = scoredict
                        self.max_score = max(self.max_score, score)
                        #print(score, smiles_concat)
                        self.n_valid += 1
                    else:
                        # print("NO", smiles_concat)
                        self.n_invalid += 1
            step += 1

    def backprop(self):
        for i, key in enumerate(self.next_token.keys()):
            child = NormalNode(key, c=self.c)
            child.id = self.total_nodes
            self.total_nodes += 1
            try:
                child.rollout_result = self.rollout_result[key]
                #print("HERE", child.rollout_result)
            except KeyError:
                child.rollout_result = ("Termination", -10000)
            self.current_node.add_Node(child)
        max_reward = max(self.next_token.values())
        # self.max_score = max(self.max_score, max_reward)
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward/(1+abs(max_reward)))
            self.current_node = self.current_node.parent

    def search(self, n_step, epsilon=0.1, loop=10, gamma=0.90, rep_file=None):
        self.set_repnode(rep_file=rep_file)

        while self.step < n_step:
            self.step += 1
            #print("--- step %d ---" % self.step)
            #print("MAX_SCORE:", self.max_score)
            if self.n_valid+self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid/(self.n_valid+self.n_invalid)
            #print("Validity rate:", valid_rate)
            self.select()
            self.expand(epsilon=epsilon, loop=loop, gamma=gamma)
            if len(self.next_token) != 0:
                self.simulate()
                self.backprop()
    def set_repnode(self, rep_file=None):
        if len(rep_file) > 0:
            for smiles in read_smilesset(hydra.utils.get_original_cwd()+rep_file):
                n = ParentNode(smiles)
                self.root.add_Node(n)
                c = NormalNode("&")
                n.add_Node(c)
        else:
            for i in range(self.l_replace+1):
                for j in range(len(self.init_smiles)-i+1):
                    infix = self.init_smiles[j:j+i]
                    prefix = "".join(self.init_smiles[:j])
                    suffix = "".join(self.init_smiles[j + i:])

                    sc = prefix + "(*)" + suffix
                    mol_sc = Chem.MolFromSmiles(sc)
                    if mol_sc is not None:
                        n = ParentNode(prefix + "(*)" + suffix)
                        self.root.add_Node(n)
                        c = NormalNode("&")
                        n.add_Node(c)

    def save_tree(self, dir_path):
        for i in range(len(self.root.children)):
            stack = []
            stack.extend(self.root.children[i].children)
            sc = self.root.children[i].original_smiles
            score = [self.root.children[i].cum_score]
            ids = [-1]
            parent_id = [-1]
            children_id = [[c.id for c in self.root.children[i].children]]
            infix = [sc]
            rollout_smiles = ["Scaffold"]
            rollout_score = [-10000]

            while len(stack) > 0:
                c = stack.pop(-1)
                for gc in c.children:
                    stack.append(gc)

                # save information
                score.append(c.cum_score)
                ids.append(c.id)
                parent_id.append(c.parent.id)
                ch_id = [str(gc.id) for gc in c.children]
                children_id.append(",".join(ch_id))
                infix.append("".join(c.path))
                rollout_smiles.append(c.rollout_result[0])
                rollout_score.append(c.rollout_result[1])

            df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score"])
            df["ID"] = ids
            df["Score"] = score
            df["P_ID"] = parent_id
            df["C_ID"] = children_id
            df["Infix"] = infix
            df["Rollout_SMILES"] = rollout_smiles
            df["Rollout_Score"] = rollout_score

            df.to_csv(dir_path+f"/tree{i}.csv", index=False)


def runEpisode(config, input_smiles, n_valid, n_invalid, reward, runIndex, action, smina_index):
    """
    Runs a single episode of the MCTS process.

    Args:
        Various arguments including configuration, input smiles, etc.

    Returns:
        A list containing the new SMILES string and other related information.
    """
    vocab = VOCABULARY 
    dictkey= config["mcts"]["reward_key"]
    model = RolloutNetwork(len(vocab))
    model_ver = config['mcts']['model_ver']
    reward.setquerymolecule(input_smiles)
    reward.setweight(action[1:5])
    model.load_state_dict(torch.load(hydra.utils.get_original_cwd()+config['mcts']['model_dir'] + f"model-ep{model_ver}.pth",  map_location=torch.device('cpu')))
    mcts = ParseSelectMCTS(input_smiles, model=model, vocab=vocab, Reward=reward, cfg = config,
                                    max_seq=config["mcts"]["seq_len"], step=config["mcts"]["n_step"] * (runIndex-1),
                                    n_valid=n_valid, n_invalid=n_invalid, c=(action[0]/2.0), max_r=reward.max_r)
    mcts.search(n_step=config['mcts']['n_step']*runIndex, epsilon = 0, loop = 10, rep_file=config["mcts"]["rep_file"])
    reward.max_r = mcts.max_score
    n_valid += mcts.n_valid
    n_invalid += mcts.n_invalid
    gen = sorted(mcts.valid_smiles.items(), key = lambda x: x[1]['score'], reverse = True)
    new_smiles = gen[0][0].split(":")[1]

    generated_smiles = pd.DataFrame(columns = ['SMILES', 'Reward', 'Imp', 'MW', 'step', 'smina_MDP'])
    start_reward = reward.reward(input_smiles)
    for kv in mcts.valid_smiles.items():
            step, smi = kv[0].split(":")
            step = int(step)

            try:
                w = Descriptors.MolWt(Chem.MolFromSmiles(smi))
            except:
                w = 0

            generated_smiles.at[smi.rstrip('\n'), "SMILES"] = smi
            generated_smiles.at[smi.rstrip('\n'), "C"] = action[0]/2.0
            generated_smiles.at[smi.rstrip('\n'), "inputsmiles"] = input_smiles
            generated_smiles.at[smi.rstrip('\n'), "Start Reward"] = start_reward[dictkey]

            current_reward = kv[1][dictkey]
            generated_smiles.at[smi.rstrip('\n'), "Reward"] = kv[1][dictkey]
            #generated_smiles.at[smi.rstrip('\n'), "Imp"] = kv[1][dictkey] - start_reward
            for key in kv[1].keys():
                generated_smiles.at[smi.rstrip('\n'), key] = kv[1][key]

            generated_smiles.at[smi.rstrip('\n'), "Imp"] = kv[1][dictkey] - start_reward[dictkey]
            generated_smiles.at[smi.rstrip('\n'), "MW"] = w
            generated_smiles.at[smi.rstrip('\n'), "step"] = step
                

    generated_smiles = generated_smiles.sort_values("Reward", ascending=False)
    

    reward_describe = generated_smiles['Reward'].describe() #this is returning highest reward, needs to 
                                                            #account for smina and binana and valid/invalid ratio and average reward of top performing

    generated_smiles = generated_smiles.reset_index(drop=True)
    generated_smiles['smina_MDP'] = 0
    generated_smiles['smina_PEP'] = 0
    generated_smiles['smina_MCT1'] = 0    
    generated_smiles['hydrophobicContacts_NOD2'] =0
    generated_smiles['hydrogenBonds_NOD2'] =0
    generated_smiles['halogenBonds_NOD2'] =0
    generated_smiles['piPiStackingInteractions_NOD2'] =0
    generated_smiles['tStackingInteractions_NOD2'] =0
    generated_smiles['cationPiInteractions_NOD2'] =0
    generated_smiles['saltBridges_NOD2'] =0  
    generated_smiles['electrostaticEnergies_NOD2'] = 0
    generated_smiles['ASN_PEP'] =0
    generated_smiles['GLU_PEP'] =0
    generated_smiles['LYS_MCT'] =0
    generated_smiles['hydrophobicContacts_MCT'] =0


    valid_df = generated_smiles.loc[(generated_smiles['qed'] != -1) & (generated_smiles['pains'] == 1)& (generated_smiles['macro'] == 1)]
    valid_row_count = valid_df.shape[0]

    output_base_dir=os.getcwd()# hydra.utils.get_original_cwd()+'/outputs'
    print(output_base_dir)
    print(valid_df.shape[0])
    smina_data_object=SMINA_data(output_base_dir, hydra.utils.get_original_cwd())
    smina_data_object.run_smina_n_processes(valid_df['SMILES'].to_list(), smina_index)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>==========================================================")
    print(os.getcwd())

    for index, row in valid_df.iterrows():
        print("getting data")
        smina_result=smina_data_object.get_data(output_base_dir, smina_index+index)
        print("got data")
        generated_smiles.at[index,'smina_MDP'] = smina_result['NOD2_score']
        generated_smiles.at[index,'smina_PEP'] = smina_result['7pn1_score']
        generated_smiles.at[index,'smina_MCT1'] = smina_result['MCT1_score']

        generated_smiles.at[index,'hydrophobicContacts_NOD2'] = smina_result['NOD2_hydrophobicContacts']
        generated_smiles.at[index,'hydrogenBonds_NOD2'] =smina_result['NOD2_hydrogenBonds']
        generated_smiles.at[index,'halogenBonds_NOD2'] =smina_result['NOD2_halogenBonds']
        generated_smiles.at[index,'piPiStackingInteractions_NOD2'] =smina_result['NOD2_piPiStackingInteractions']
        generated_smiles.at[index,'tStackingInteractions_NOD2'] =smina_result['NOD2_tStackingInteractions']
        generated_smiles.at[index,'cationPiInteractions_NOD2'] =smina_result['NOD2_cationPiInteractions']
        generated_smiles.at[index,'saltBridges_NOD2'] =smina_result['NOD2_saltBridges'] 
        generated_smiles.at[index,'electrostaticEnergies_NOD2'] = smina_result['NOD2_electrostaticEnergies']
        generated_smiles.at[index, 'ASN_PEP'] = smina_result['PEP_hydrogenBonds']
        generated_smiles.at[index, 'GLU_PEP'] = smina_result['PEP_closeContacts']
        generated_smiles.at[index, 'LYS_MCT'] = smina_result['MCT1_hydrogenBonds']
        generated_smiles.at[index, 'hydrophobicContacts_MCT'] = smina_result['MCT1_hydrophobicContacts']

    outerreward_score = outerreward(generated_smiles,current_reward)
    state = currentstate(generated_smiles,current_reward)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(os.getcwd())
    
    generated_smiles.to_csv( "./data/"+ "No-{:04d}.csv".format(runIndex), index=False)
    #filtered_smiles = generated_smiles.loc[generated_smiles['smina_MDP'] != 0]
    filtered_smiles = generated_smiles.loc[generated_smiles['OutReward'].idxmax()]
    new_smiles = filtered_smiles['SMILES']

    return [new_smiles, n_valid, n_invalid, outerreward_score, state, smina_index+valid_df.shape[0]]




@hydra.main(config_path="../config/", config_name="config")
def testrunEpisode(config):
    input_smiles = 'CC(C(=O)NC(CCC(=O)O)C(=O)N)NC(=O)C(C)OC1C(C(OC(C1O)CO)O)NC(=O)C'
    reward = getReward(name = config['mcts']['reward_name'], init_smiles = input_smiles)
    n_valid = 0
    n_invalid = 0 
    action = [0.5,0,1,0,0]
    runEpisode(config, input_smiles, n_valid, n_invalid, reward, 3602, action)





if __name__ == "__main__":
    #main()
    testrunEpisode()


