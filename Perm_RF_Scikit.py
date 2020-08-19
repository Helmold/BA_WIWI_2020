#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from decimal import *
from sklearn.metrics import confusion_matrix
getcontext().prec = 10
#load data
import time
start_time = time.time()

data = pd.read_csv('Data_Red_Perm_Tra.csv')
print("Data loaded")
#split in target and train
X,y = data.drop(['Target','ID'], axis=1), data['Target']
#make a list of categorical features and sepearate from numeric in lists
categorical_feature_mask = X.dtypes==object
categorical_features = X.columns[categorical_feature_mask].tolist()
numeric_feature_mask = X.dtypes!=object
numeric_features = X.columns[numeric_feature_mask].tolist()
print (categorical_features)
print(numeric_features)

#rearrange
X = data[categorical_features + numeric_features]
print(X)
#train_test_splt
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size = 0.3 , random_state=42)
#make pipeline
categorical_pipe = Pipeline([
   ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#strategy for numeric missing: mean
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])
# preprocessing for cat to onehot dummies
preprocessing = ColumnTransformer(
   [('cat', categorical_pipe, categorical_features),
     ('num', numerical_pipe, numeric_features)])
#select classifier for pipeline
rf = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', RandomForestClassifier(random_state=42))
])
#fit model


rf.fit(X_train, y_train)

#print accuracy
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))

y_pred = rf.predict(X_test)

cm = confusion_matrix(y_pred,y_test,labels=[1,0])
print(cm)

print("--- %s seconds ---" % (time.time() - start_time))

#make prediction on test_set
print(rf.predict(X_test))
#possible plot for casual feature importaces
ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = ohe.get_feature_names(input_features=categorical_features)
feature_names = np.r_[feature_names, numeric_features]
'''
tree_feature_importances = (
    rf.named_steps['classifier'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances (MDI)")
fig.tight_layout()
plt.show()'''

X_test_prep = pd.DataFrame(data = rf.named_steps["preprocess"].transform(X_test), columns = feature_names)
print(X_test_prep)
#select range in advance for forceplot only one row
select = range(0,100)

#train the explainer and calculate SHAP / expected value
explainer = shap.TreeExplainer(rf['classifier'])
#select class [1],[0] or nothing for both (only works for summary plot)
shap_values = explainer.shap_values(X_test_prep.iloc[select])[1]
expected_value = explainer.expected_value[1]

#plotting the SHAP, delete te comment # to get a plot
#shap.summary_plot(shap_values , X_test_prep.iloc[select], plot_size=(10,5))
shap.decision_plot(expected_value, shap_values, X_test_prep.iloc[select])
#shap.dependence_plot("EXT2", shap_values, X_test_prep.iloc[select])

#Force plot only works for single input Range 1 eg
#shap.force_plot(expected_value, shap_values, X_test_prep.iloc[select],matplotlib=True)

#check time
print("--- %s seconds ---" % (time.time() - start_time))
