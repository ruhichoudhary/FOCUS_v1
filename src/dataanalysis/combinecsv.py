import os
import sys
import glob
import numpy as np
import pandas as pd

outputDFCreated = False
outputDF=None

for name in glob.glob('/Users/ruhichoudhary/runs_data/run1/2023-01-11/22-16-05/data/*.csv'):
    print(name)
    df=pd.read_csv(name)
    #print(df)
    df=df.iloc[:10]
    df. replace(np. nan,0, inplace=True)
    df=df.loc[df['smina_MCT1']!=0]
    if(outputDFCreated==False):
        outputDFCreated=True
        outputDF=df
    else:
        outputDF=outputDF.append(df)

print(outputDF)
outputDF.to_csv("~/runs_data/run1/summary.csv")

