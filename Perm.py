#import packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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
getcontext().prec = 10
#load data
data = pd.read_csv('merged_final.csv')
data_copy = data.copy()
print("Data loaded")

#split in target and train
X,y = data.drop(['TARGET','SK_ID_CURR'], axis=1), data['TARGET']

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
   X, y, stratify=y , random_state=42)

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
#make prediction on test_set
print(rf.predict(X_test))

#possible plot for casual feature importaces
"""ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = ohe.get_feature_names(input_features=categorical_features)
feature_names = np.r_[feature_names, numeric_features]

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
plt.show()"""

#make Permutation sorted Importance Ranking
result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=1)
sorted_idx = result.importances_mean.argsort()

#possible plot for Permutation Importances
"""fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()"""

#Dataframe with sorted Permutation Importance per feature
df = pd.DataFrame(columns=['Feature','Importance_Mean','S_dev'])
#possible selection for threshold
for i in result.importances_mean.argsort()[::-1]:
     if result.importances_mean[i] >= """here"""0.000 or result.importances_mean[i] < """and here"""0.000:

          df = df.append({'Feature': X_test.columns[i],'Importance_Mean':result.importances_mean[i],'S_dev':result.importances_std[i]}, ignore_index=True)

     
print(df)

#create new DataFrame with selected features by threshold
selected_features = pd.DataFrame(data = data_copy, columns = ['TARGET'])
selected_features[df['Feature'].tolist()]= pd.DataFrame(data = data_copy, columns = df['Feature'].tolist())
#save as csv
df.to_csv('Perm_Imp_Score.csv')
selected_features.to_csv('Selected_Data_Perm_Imp.csv',index=False)

print(selected_features)
