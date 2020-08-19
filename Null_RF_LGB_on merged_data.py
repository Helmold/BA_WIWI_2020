#import necessary packagesD
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import warnings
import gc
import shap
shap.initjs()
gc.enable()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, show


warnings.simplefilter('ignore', UserWarning)

import time
start_time = time.time()

#import data and copy
data = pd.read_csv('merged_final.csv')
#raw_data = data.copy()


#handle JSON
import re
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#raw_data = raw_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

x,y = data.drop(['TARGET','SK_ID_CURR'], axis=1), data['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3 , random_state=42)
x_train_raw = x_train.copy()
x_test_raw = x_test.copy()
y_train_raw = y_train.copy()
y_test_raw = y_test.copy()


pd.set_option("display.precision",2)
print(x_test)
print(y_train)
print(y_test)
print(x_train)


#get categorical_features train
categorical_feats = [
    f for f in x_train.columns if x_train[f].dtype == 'object'
]
#handle cat as type cat in train
categorical_feats
for f_ in categorical_feats:
    x_train[f_], _ = pd.factorize(x_train[f_])
    # Set feature type as categorical
    x_train[f_] = x_train[f_].astype('category')

    train_features = [f for f in x_train if f not in ['TARGET','SK_ID_CURR']]
    # Shuffle target if required


x_train = x_train[train_features]
print(x_train)

#get categorical_features test
categorical_feats = [
    f for f in x_test.columns if x_test[f].dtype == 'object'
]
#handle cat as type cat in test
categorical_feats
for f_ in categorical_feats:
    x_test[f_], _ = pd.factorize(x_test[f_])
    # Set feature type as categorical
    x_test[f_] = x_test[f_].astype('category')

    train_features = [f for f in x_test if f not in ['TARGET','SK_ID_CURR']]
    # Shuffle target if required


x_test = x_test[train_features]
print(x_test)


# Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
dtrain = lgb.Dataset(x_train, y_train,silent=True)
dtest = lgb.Dataset(x_test, y_test,silent=True)
lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': None,
        'bagging_freq': 1,
        'n_jobs': 4
    }
    
    # Fit the model
clf = lgb.train(lgb_params, dtrain, 200, categorical_feature=categorical_feats)





#make prediction and transform to binomial
y_pred = clf.predict(x_test)
print(y_test)
print(y_pred)
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
            y_pred [i] = 1
    else:
            y_pred [i] = 0
print(y_pred)
#make confusion matrix and accuracy score
cm = confusion_matrix(y_pred,y_test,labels=[1,0])
print(cm)
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

print("--- %s seconds ---" % (time.time() - start_time))



#Different approaches to apply SHAP and get plots
'''
# number 1
#Train the explainer
explainer = shap.TreeExplainer(clf)
#calculate SHAP values
shap_values = explainer.shap_values(x_train)

#print(shap_values[1][1,:])
#print(raw.loc[x_train.iloc[2,:].index])
#print(x_train_raw.iloc[0,:])
#print(x_train.iloc[0,:])
#print(x_train.iloc[1,:])

#creating new dataframe containing Shap values and merge them with the feature values
#values1 = pd.DataFrame(shap_values[1][1,:],columns = ['SHAP Values'] ,index = x_train.iloc[1,:].index)
#values2 = pd.DataFrame (x_train.iloc[1,:].values,columns = ['Feature Values'] ,index = x_train.iloc[1,:].index)
#values = values1.merge(values2, left_index=True, right_index=True)
#print(shap_values[1][1,:])
#print (values)

#Plotting the Shap out of it
#shap.summary_plot(shap_values, x_train)
#shap.summary_plot(shap_values[1], x_train)
#shap.force_plot(explainer.expected_value[1],shap_values[1][0,:],x_train.loc[x_train_raw.iloc[0,:]], matplotlib=True)
shap.force_plot(explainer.expected_value[1],shap_values[1][1,:], x_train_raw.iloc[0,:] ,matplotlib = True)

shap.decision_plot(explainer.expected_value[1], shap_values[1], x_test_raw)
shap.dependence_plot(0, shap_values[0], x_test)
#shap.decision_plot(explainer.expected_value[1], shap_values[1], x_train_raw)
'''
#number 2 with selected range

#range selection in advance only one row if force plot is desired
select = range(0,10000)
features = x_test.iloc[select]
features_display = x_test.loc[features.index]
#train the explainer and calculate the SHAP values select class with [0],[1] or nothing which only works for summary plot
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(features)[1]#[select]
#expected_value = explainer.expected_value[0]

#optional normalisation of values because the values dont look like expected
expected_value = clf.predict(features).mean()
#print(expected_value)
for cell in np.nditer(shap_values, op_flags=['readwrite']):
   cell[...] = (cell/(explainer.expected_value[0]*2))

converter_value = (np.sum(np.abs(shap_values), axis = 1,keepdims=True))
print(converter_value)
shap_values_modification = (shap_values/converter_value)*clf.predict(features).mean()

shap_values = shap_values - shap_values_modification

#some printings to check
print(shap_values)
#print(np.sum(shap_values, axis = 1,keepdims=True))
print(expected_value)
#print(np.max(shap_values))
#print(np.min(shap_values))
#print(clf.predict(features).mean())
#print(clf.predict(features))
#print(shap_values)
#print(np.sum(shap_values, axis = 1,keepdims=True))
#print(np.sum(shap_values, axis = 1,keepdims=True))
print(features)
#print(y_test.iloc[select])
#print(clf.predict(features))

#Plotting the SHAP out of it
shap.summary_plot(shap_values, features)
#shap.decision_plot(expected_value, shap_values, x_test_raw.iloc[select])
#shap.dependence_plot(1, shap_values, features)
#Force plot only works for single input
#shap.force_plot(expected_value, shap_values, x_test_raw.iloc[select],matplotlib=True)
print("--- %s seconds ---" % (time.time() - start_time))
