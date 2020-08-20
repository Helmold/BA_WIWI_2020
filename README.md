# BA_WIWI_2020
A review on SHAP applications.

["OneDrive Link to Data, Code and Plots"](https://1drv.ms/u/s!Ai_kVD0Jmj3nkFuepfC-UkJUME-V?e=syCCDd "OneDrive Link to Data, Code and Plots")

To reproduce, first download the original data from ["Kaggle"](https://www.kaggle.com/c/home-credit-default-risk/data) or from the DATA folder linked above.
Because I dropped some information you do not need application_test.csv (no target variables to control), bureau_balance.csv(no direct information about individuals),
installments_payments.csv (moslty information that's is also found summed up in previous_application.csv) and sample_submission.csv(only needed for the competition.

The rest is needed if you want to run the code. Use the following order.     
1. Prepro.py    
2. Null.py/Perm.py to get the importance values    
3. Make_Reduced_Data.py if you take the same 20 most important features for each set I selected, otherwise the selection has to be changed manually in the code.    
4. Get transformed data from ["DATA"](https://1drv.ms/u/s!Ai_kVD0Jmj3nkFuepfC-UkJUME-V?e=syCCDd "OneDrive Link to Data, Code and Plots")  with added  _tra.csv the 20 most important features I selected and renamed them. Otherwise incode change pd.load_csv and del the _Tra addon in the csv.file    
5. Null_RF_LGB.py/Null_RF_LGB_on merged_data.py/Perm_RF_Scikit.py to get the plots. You can variate the range of the sample in code or switch on/off plotting by commenting out.  
  
OR you start the toghether.py file therefore you definitely need the Data_Red_Null_Tra.csv and the Data_Red_Perm_Tra.csv file.It's used in step 5.  
  
  Have fun plotting

appl.csv - rearranged and preprocessed application_train.csv  
bur.csv - rearranged and preprocessed bureau.csv  
pre.csv - rearranged and preprocessed previous_application.csv  
pos.csv - rearranged and preprocessed pos_cash_balance.csv  
ccb.csv - rearranged and preprocessed credit_card_balance.csv  
Split_Gain_Imp_Scores.csv - Scores calculated by null importance for each feature  
Perm_Imp_Score.csv - Scores calculated by permutation importance for each feature  
Features.xlsx -  Manually created table to select the 20 most important features  
merged_final.csv - Ready to use merged data from all meta files  
Data_Red_Perm.csv - Table with only the most important features from permutation importance  
Data_Red_Null.csv - Table with only the most important features from null importance  
Data_Red_Perm_Tra.csv - Table with only the most important features from permutation importance, renamed variables  
Data_Red_Null_Tra.csv - Table with only the most important features from null importance, renamed variables  
Null.py - Algorithm to calculate the null importance ranking  
Perm.py - Algorithm to calculate the permutation importance  
Null_RF_LGB.py - Algorithm applied on the reduced data from the null importance ranking - Renamed variables data used in code  
Perm_RF_Scikit.py - Algorithm applied on the reduced data from the permutation importance ranking - Renamed variables data used in code  
Null_RF_LGB_on merged_data.py - Algorithm applied on the whole merged dataset  
Prepro.py - Preprocessing of the data and meta data. Creating of the new sheets  
Make_Reduced_Data.py - Reduced data created from merged data  
