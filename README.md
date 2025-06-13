**1. PROJECT OVERVIEW**
   
This repository contains the code to perform cross-validated sparse Canonical Correlation Analysis (sCCA) on 2 datasets. The cross-validation design employed is a 2-layer multiple holdout cross-validation, with the aim of reducing sampling biases and increase generalizability. 

The code was developed to investigate multivariate associations between datasets (e.g., biological and behavioral features), while minimizing the risk of data leakage.

The pipeline includes:
1) confound regression and data standardization, performed independently within training and test sets to avoid data leakage;
3) a grid search to identify optimal regularization parameters for sparsity control in the sCCA model;
4) model evaluation through out-of-sample canonical correlation and permutation testing.

The cross-validation structure consists of:
1) an outer loop (50 iterations in the present example) performing train/test splits to evaluate model performance and reduce overfitting;
2) an inner loop (100 iterations in the present example) performing hyperparameter tuning via grid search to select the optimal pair of regolarizzazione parameters.

This pipeline is designed for flexible application and easy customization to paired multivariate datasets where dimensionality reduction and interpretability are critical.



**2. USAGE**

1) the **"1_multiple holdout_sCCA"** script is the core script and the first to be run. It creates and tests the multiple holdout sCCA model. The input required are 4 csv files: dataset1, dataset2, covariates_DS1, covariates_DS2. Note: all 4 files must contain a first column called "Participant_ID", which serves as matching test between the 4 files. This setting, and all model parameters (e.g., regularization parameters to be optimized, train/test ratio, number of resamplings and permutations) are editable on any Python IDE. The outputs of this code are: i) A .txt file containing infos about chosen parameters, canonical correlations and p values; ii) A .xlsx file containing canonical weights across cross-validation folds.

2) the **"2_produce_tablesummary"** script organizes the performance of the model (for each canonical pair across each outer resampling) in an Excel file. It takes as input the .txt file created by "1_multiple holdout_sCCA".

3) the **"3_organize_canonical_weights"** script organizes the canonical weights (for one selected canonical pair across each outer resampling) in a excel file. It takes as input the .xlsx file created by "1_multiple holdout_sCCA".

4) the **"4_count_parameters_frequency"** script returns the best pair of regularization parameters (i.e., the most frequently selected across the outer resamplings). It takes as input the .txt file created by "1_multiple holdout_sCCA".

5) the **"5_single_sCCA_given_regparam"** script allows to fit the sCCA model on the whole sample with the best regularization parameters, in order to extract individual canonical scores and stability of canonical weights. Note: since in the "1_multiple holdout_sCCA" script covariate correction is performed within the cross-validation scheme, before running this code you have to externally residualize the 2 datasets for selected covariates.
