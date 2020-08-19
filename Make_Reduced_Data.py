import numpy as np
import pandas as pd

df = pd.read_csv('merged_final.csv')
#reduce merged final to selected features by importance ranking throught Perm_Imp
de = df[['TARGET','SK_ID_CURR','EXT_SOURCE_2','EXT_SOURCE_3','AMT_CREDIT_SUM_OVERDUE','AMT_CREDIT_SUM_DEBT','DAYS_BIRTH','REGION_POPULATION_RELATIVE','DAYS_LAST_PHONE_CHANGE','EMERGENCYSTATE_MODE','dur_payments_years','REGION_RATING_CLIENT_W_CITY','FLAG_OWN_CAR','CREDIT_DAY_OVERDUE','DAYS_ID_PUBLISH','FLAG_DOCUMENT_3','REGION_RATING_CLIENT','NAME_EDUCATION_TYPE','DAYS_EMPLOYED','REG_CITY_NOT_WORK_CITY','DPD_AVR_CASH','AMT_GOODS_PRICE']]
#reduce merged final to selected features by importance ranking throught Null_Imp
dd = df[['TARGET','SK_ID_CURR','NAME_EDUCATION_TYPE','EXT_SOURCE_2','EXT_SOURCE_3','REGION_RATING_CLIENT_W_CITY','FLAG_DOCUMENT_3','CODE_GENDER','FLAG_EMP_PHONE','EXT_SOURCE_1','REGION_RATING_CLIENT','NAME_CONTRACT_TYPE','AMT_CREDIT_SUM_OVERDUE','DPD_AVR_CASH','DAYS_EMPLOYED','dur_payments_years','OCCUPATION_TYPE','REG_CITY_NOT_LIVE_CITY','AMT_GOODS_PRICE','AMT_CREDIT_SUM_DEBT','NAME_INCOME_TYPE','FLAG_OWN_CAR']]
de.to_csv('Data_Red_Perm.csv', index = False)
dd.to_csv('Data_Red_Null.csv', index = False)
