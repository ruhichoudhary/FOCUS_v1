#!/usr/bin/env python
"""
Model training
"""

import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from tdc.benchmark_group import admet_group
from config import config

# get task


name = "caco2_wang"

fp_test = np.load(open("./smiles.npy", "rb"))

predictions_list = []
feature_imp_list = []

for random_state in range(5):
    clf = xgb.XGBRegressor()
    clf.load_model("./model"+str(random_state)+".json")
    pred_xgb = clf.predict(fp_test)
    # add to predicitons dict
    predictions = {}
    predictions[name] = pred_xgb
    predictions_list.append(predictions)
    # get feature importance
    feature_imp_list.append(clf.feature_importances_)
    del clf

print(predictions_list)
print(predictions_list[0])
print((predictions_list[0][name][0]+ predictions_list[1][name][0]+predictions_list[2][name][0]+predictions_list[3][name][0]+predictions_list[4][name][0])/5)
#print(avg_caco)


# feature importance by fingerprints and descriptors
# ending 0 is added as placeholder
#feature_imp = []
#feature_size = [167, 2048, 300, 1613, 208, 0]

#for feature in feature_imp_list:
#    feature_imp_cur = []
#    running_size = 0
#    for size in feature_size:
#        feature_imp_cur.append(
#            np.sum(feature[running_size: running_size + size])
#        )
#        running_size += size
#    feature_imp.append(feature_imp_cur)

#feature_imp_mean = np.mean(feature_imp, axis=0)

#print(
#    f"maccskeys: {feature_imp_mean[0]*100:.2f}% ",
#    f"circular: {feature_imp_mean[1]*100:.2f}% ",
#    f"mol2vec: {feature_imp_mean[2]*100:.2f}% ",
#    f"mordred: {feature_imp_mean[3]*100:.2f}% ",
#    f"rdkit: {feature_imp_mean[4]*100:.2f}%"
#)
