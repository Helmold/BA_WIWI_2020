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
