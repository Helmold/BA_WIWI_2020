#import necessary packages
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import gc

gc.enable()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier



warnings.simplefilter('ignore', UserWarning)

#import data and copy
data = pd.read_csv('merged_final.csv')
raw_data = data.copy()
#handle JSON
import re
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
raw_data = raw_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

#get categorical_features
categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
#handle cat as type cat
categorical_feats
for f_ in categorical_feats:
    data[f_], _ = pd.factorize(data[f_])
    # Set feature type as categorical
    data[f_] = data[f_].astype('category')

def get_feature_importances(data, shuffle, seed=None):
    # Gather all features
    train_features = [f for f in data if f not in ['TARGET','SK_ID_CURR']]
    
    
    # Shuffle target if required
    y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['TARGET'].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
    #make prediction and transform to binomial
    y_pred = clf.predict(data[train_features])
    print(y)
    print(y_pred)
    for i in range(0, len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred [i] = 1
        else:
            y_pred [i] = 0
    print(y_pred)
    
    #confusion matrix and accuracy for the model
    cm = confusion_matrix(data['TARGET'], y_pred)
    print(cm)
    accuracy=accuracy_score(y_pred,data['TARGET'])
    print(accuracy)

    return imp_df


np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(data=data, shuffle=False)
print(actual_imp_df.head())

# get null importances with shuffled target, selectable number of runs
null_imp_df = pd.DataFrame()
nb_runs = """here"""60
#needed time per run
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=data, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)
    print(null_imp_df.head())
#make scoring for importances vs null importances (75 percentile)
feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))
#sort scores in new dataframe
scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score']).sort_values('split_score',ascending=False)


split_score = pd.DataFrame(data=scores_df, columns = ['feature','split_score']).sort_values('split_score', ascending=False)
gain_score = pd.DataFrame(data=scores_df, columns = ['feature','gain_score']).sort_values('gain_score', ascending=False)

#drop unimportant features with selectable threshold
split_score = split_score.drop(split_score[abs(split_score.split_score)"""here""" < 0.5].index)
gain_score = gain_score.drop(split_score[abs(split_score.split_score)"""and here""" < 5].index)

#create new reduced dataframe from important features
data_new = pd.DataFrame(data = raw_data,columns = ['TARGET'])
data_new[split_score['feature'].tolist()] = pd.DataFrame(data = raw_data,columns = split_score['feature'].tolist() )
print(data_new)
#save seperated null importace by split score
data_new.to_csv('Selected_Data_Null_Imp_Split.csv')
#save seperated null importance by gain score
data_new = pd.DataFrame(data = raw_data,columns = ['TARGET'])
data_new[gain_score['feature'].tolist()] = pd.DataFrame(data = raw_data,columns = gain_score['feature'].tolist() )
print(data_new)
#save together as frame
data_new.to_csv('Selected_Data_Null_Imp_Gain.csv')

#print(split_score)
print(scores_df)
scores_df.to_csv('Split_Gain_Imp_Scores.csv',index=False)

#plotting for null importance distributions for one single feature
def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
    plt.show()


#example plot for livingapartment_avg
display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='LIVINGAPARTMENTS_AVG')
