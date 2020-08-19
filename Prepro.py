import numpy as np
import pandas as pd

#read main and meta data
appl = pd.read_csv('application_train.csv')
bur = pd.read_csv('bureau.csv')
ccb = pd.read_csv('credit_card_balance.csv')
pos = pd.read_csv('pos_cash_balance.csv')
pre = pd.read_csv('previous_application.csv')

#create new columns in main data
appl ['debt_to_income_ratio'] = appl['AMT_ANNUITY']/appl['AMT_INCOME_TOTAL']
appl ['dur_payments_years'] = appl['AMT_CREDIT']/appl['AMT_ANNUITY']
appl.loc[appl['DAYS_EMPLOYED'] == 365243,'DAYS_EMPLOYED'] = np.nan
print(appl)

#preprocess and select meta bureau
bur = bur[['SK_ID_CURR','CREDIT_DAY_OVERDUE','DAYS_ENDDATE_FACT','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']]
bur['active'] = np.where(bur['DAYS_ENDDATE_FACT']> 0, 1,0)
bur = bur.groupby('SK_ID_CURR').mean()
bur['active_outer'] = bur['active'].apply(lambda x: 1 if x >0 else 0)
bur = bur.drop(['active','DAYS_ENDDATE_FACT'],axis=1)
print(bur)

#preprocess and selcect meta data credit card balance
ccb = ccb[['SK_ID_CURR','SK_DPD_DEF']]
ccb = ccb.rename({'SK_DPD_DEF': 'DPD_AVR_REV'}, axis=1)
ccb = ccb.groupby('SK_ID_CURR').mean()
print(ccb)

#preprocess and select meta data POS_cash
pos = pos[['SK_ID_CURR','SK_DPD_DEF']]
pos = pos.rename({'SK_DPD_DEF': 'DPD_AVR_CASH'}, axis=1)
pos = pos.groupby('SK_ID_CURR').mean()
print(pos)

# preprocess nad select meta data previous_credits
pre = pre[['SK_ID_CURR','DAYS_TERMINATION','NAME_CONTRACT_STATUS']]
pre = pre.drop(pre[pre.NAME_CONTRACT_STATUS == 'Refused'].index)
pre['active'] = np.where(pre['DAYS_TERMINATION'] > 0, 1,0)
pre = pre.drop(['DAYS_TERMINATION','NAME_CONTRACT_STATUS'],axis=1)
pre = pre.groupby('SK_ID_CURR').mean()
pre['active_inner'] = pre['active'].apply(lambda x: 1 if x >0 else 0)
del pre['active']
print(pre)

#merg to final data
result = appl.merge(bur,on='SK_ID_CURR',how='left').merge(ccb ,on='SK_ID_CURR',how='left').merge(pos ,on='SK_ID_CURR',how='left').merge(pre ,on='SK_ID_CURR',how='left')
print(result)

#save to csv
result.to_csv('merged_final.csv',index=False)
appl.to_csv('appl.csv', index=False)
bur.to_csv('bur.csv')
ccb.to_csv('ccb.csv')
pos.to_csv('pos.csv')
pre.to_csv('pre.csv')

    
