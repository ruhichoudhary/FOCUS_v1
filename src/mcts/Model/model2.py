import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf
# load Keras libraries
import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd


class RolloutNetwork():


    def __init__(self,vocab):
        self.model = load_model('/Users/ruhichoudhary/code/mermaid/Model/RC_fragment_model3.h5')
        self.vocab = vocab
        self.char_to_int = dict((c,i) for i,c in enumerate(self.vocab))
        self.int_to_char = dict((i,c) for i,c in enumerate(self.vocab))
        self.embed = 50 

    def vectorize(self,smiles):
        print((smiles.shape))
        one_hot = np.zeros((smiles.shape[0], self.embed, len(self.vocab)), dtype=np.int8)
        for i, smile in enumerate(smiles):
            print("HERE", smile)
    
        # encode the start
            one_hot[i,0,self.char_to_int["!"]] = 1
        #encode the smiles characters
            for j, c in enumerate(smile):
                one_hot[i,j+1,self.char_to_int[c]] = 1
        # encode the end of the smiles string
            one_hot[i,len(smile)+1:,self.char_to_int["E"]] = 1
    # return two items, one for input and one for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]       


    def generatekeys(self, smiles):
        X, y = self.vectorize(smiles)
    
        char1 = ""
        char2 = ""
        char3 = ""
        v = self.model.predict([X[0:1], X[0:1]]) 

        indices = v[0][0].argsort()
        char1 = self.int_to_char[indices[-1]]
        char2 = self.int_to_char[indices[-2]]
        char3 = self.int_to_char[indices[-3]]
        return {char1, char2, char3}
            

    def generatesmiles(self, smiles):
        X, y = self.vectorize(smiles)
    
        pred = ""
        v = self.model.predict([X[0:1], X[0:1]]) 
        idxs = np.argmax(v, axis=2)
        pred="".join([self.int_to_char[h] for h in idxs[0]])[:-1]
        pred=pred.replace("E", "")
        return pred
  

    
@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    VOCABULARY2 = ['!', '#', '(', ')', '+', '-', '1', '2', '3', '=', 'C', 'E', 'H', 'N', 'O', 'S', '[', ']']


    mydict={"1","2"}
    mydict["3"]=1
    obj = RolloutNetwork(VOCABULARY2)
    X = np.array(['!'], dtype = str)

    keys = obj.generatekeys(X)
    print(keys)

    for key in keys:
        X = np.array([key], dtype = str)
        smiles = obj.generatesmiles(X)
        print(smiles)



if __name__ == "__main__":
    main()


