import math

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


@dataclass
class PreProcess:
    datapath: str = "/input/sample_data.smi"
    outdir: str = "/preprocessed/"
    ratio: float = 0.8
    max_len: int = 20


@dataclass
class ModelConfig:
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dr_rate: float = 0.2
    fr_len: int = 25


@dataclass
class TrainConfig:
    lr: float = 0.0001
    epoch: int = 100
    datapath: str = "/data/input/fragments.smi"
    ratio: float = 0.2
    batch_size: int = 128
    seq_len: int = 25
    shuffle: bool = True
    drop_last: bool = True
    start_epoch: int = 0
    eval_step: int = 1
    save_step: int = 10
    model_dir: str = "/ckpt/"
    log_dir: str = "/logs/"
    ckptdir: str = "/ckpt/"
    log_filename: str = "log.txt"


@dataclass
class MCTSConfig:
    n_step: int = 5
    n_iter: int = 3
    seq_len: int = 25
    in_smiles_file: str = "/data/input/init_smiles.smi"
    rep_file: str = ""
    modeL_dir: str = "/data/models/"
    out_dir: str = "/Users/ruhic/runs_data/run1/Data"
    ucb_c: float = 1/(math.sqrt(2))
    model_ver: int = 100
    reward_name: str = "MultiReward"
    reward_key: str = "score"
    model_dir: str = "/data/models/ckpt/"


@dataclass
class SMINAConfig:
    NOD2: str = '/data/pdb/NOD2.pdb'
    pep: str = '/data/pdb/7pn1.pdb'
    MCT1: str = '/data/pdb/MCT1.pdb'

@dataclass
class Config:
    prep: PreProcess = PreProcess()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    mcts: MCTSConfig = MCTSConfig()
    smina: SMINAConfig = SMINAConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

