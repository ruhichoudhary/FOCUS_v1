import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("/Users/ruhic/runs_data/run1/2023-01-03/13-41-33/rewardscopy.txt", header = None)
df = df.drop(columns = 16)
df.columns = ['max_reward', 'outerreward', 'sminaNOD2', 'sminaPEP', 'sminaMCT', 'hydrophobicNOD', 
            'hydrogen', 'pipi', 't-stack', 'cationpi', 'saltbridges', 'electrostatic', 'ASN_PEP', 'GLU_PEP', 'LYS_MCT','hydrophobicMCT']
df['outerreward'] = df['outerreward'].str.strip('[').astype(float)

#Correlations of whole dataframe

corr_df = df.drop(columns = ['outerreward'])

heatmap = sns.heatmap(corr_df.corr(method="spearman"), cmap="PiYG",vmin=-1, 
                      vmax=1, annot=True)
plt.title("Spearman Correlation")
plt.show()


#every 100 timesteps aka PER BATCH
df100 = df.groupby(np.arange(len(df))//100).mean()
"""
corr_df = df100.drop(columns = ['outerreward'])

heatmap = sns.heatmap(corr_df.corr(method="spearman"), cmap="PiYG",vmin=-1, 
                      vmax=1, annot=True)
plt.title("Spearman Correlation")
plt.show()
"""

plt.plot(df100.index,df100['max_reward'])
plt.xlabel('Per Batch')
plt.plot(df100.index,df100['outerreward'])
plt.legend(['max inner reward', 'outer reward'])
#plt.show()
#plt.plot(df100.index,df100['sminaNOD2'])
#plt.plot(df100.index,df100['sminaPEP'])
#plt.plot(df100.index,df100['sminaMCT'])
#plt.legend(['NOD2', 'PEP', 'MCT'])
#plt.show()


#every 50 timseteps aka PER EPISODE
df50 = df.groupby(np.arange(len(df))//50).mean()
"""
corr_df = df50.drop(columns = ['outerreward'])
matrix = np.triu(corr_df)

heatmap = sns.heatmap(corr_df.corr(method="spearman"), cmap="PiYG",vmin=-1, 
                      vmax=1, annot=True, mask = matrix)
plt.title("Spearman Correlation")
plt.show()
"""
"""
plt.plot(df50.index,df50['max_reward'])
plt.xlabel('Per Batch')
plt.plot(df50.index,df50['outerreward'])
plt.legend(['max inner reward', 'outer reward'])
plt.show()
"""

#Arrange based on outerreward
df = df.sort_values('outerreward', ascending=False)
best5 = df[:5]
#print(best5)