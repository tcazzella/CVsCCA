"""

MULTIPLE HOLDOUT sCCA PIPELINE

@author: Tommaso Cazzella, PhD Student
         Psychiatry and Clinical Psychobiology Unit
         Division of Neuroscience
         Vita-Salute San Raffaele University
         IRCCS San Raffaele Hospital
         
"""

###############################################################################
######################## IMPORT NECESSARY PACKAGES ############################

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sparsecca._cca_pmd import cca
import random
from openpyxl import Workbook
from sklearn.linear_model import LinearRegression

######################## IMPORT NECESSARY PACKAGES ############################
###############################################################################





###############################################################################
################################ DATA LOADING #################################

# DATA FORMAT
## Both datasets need to be in csv format
## Both datasets should contain labels in first rows
## Both datasets must have a first column called "Participant_ID"
dataset1 = pd.read_csv('') # Load dataset 1
dataset2 = pd.read_csv('') # Load dataset 2

# COVARIATES FORMAT
## The covariates file's formats must be the same of the datasets
covariates_DS1 = pd.read_csv('') # Load covariates for dataset 1
covariates_DS2 = pd.read_csv('') # Load covariates for dataset 2



# Verify alignment by Participant_ID
assert (dataset1["Participant_ID"] == dataset2["Participant_ID"]).all(), "Mismatch in Participant IDs"
assert (dataset1["Participant_ID"] == covariates_DS1["Participant_ID"]).all(), "Mismatch in Participant IDs for X between data and covariates"
assert (dataset2["Participant_ID"] == covariates_DS2["Participant_ID"]).all(), "Mismatch in Participant IDs for Y between data and covariates"

# Drop Participant_ID column
dataset1 = dataset1.drop(columns=["Participant_ID"])
dataset2 = dataset2.drop(columns=["Participant_ID"])
covariates_DS1 = covariates_DS1.drop(columns=["Participant_ID"])
covariates_DS2 = covariates_DS2.drop(columns=["Participant_ID"])

# Convert datasets to numpy arrays
dataset1 = dataset1.values
dataset2 = dataset2.values

# Convert covariates to numpy arrays
covariates_DS1 = covariates_DS1.values
covariates_DS2 = covariates_DS2.values

################################ DATA LOADING #################################
###############################################################################





###############################################################################
######################### MODEL'S PARAMETERS SETTING ##########################

# Define the test set ratio
test_ratio = 0.3 # percentage of test sets

# Define the validation set ratio
validation_ratio = 0.3 # percentage of validation sets

# Set the number of canonical variates to include
n_components = 1

# Set the number of repetitions in internal and external loops 
n_splits = 100 # number of train/validation resamplings in the internal loop
n_iterations = 50 # number of train/test resamplings in the external loop 

# Initialize regularization parameter lists for X and Y
x_penalties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_penalties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Set the number of permutations
n_perm_outer_test = 10000

# Provide the paths for the model summary files
# The first file has to be in .txt format and will contain infos about choosen parameters, canonical correlations and p values 
# The second file has to be in .xlsx format and will contain canonical weights across cross-validation folds
output_file_txt  = ''
output_file_excel = ''

######################### MODEL'S PARAMETERS SETTING ##########################
###############################################################################





###############################################################################
################### RESIDUALIZATION and SPLITS FUNCTIONS ######################

# Residualizing function (based on linear regression)
def residualize(data, covariates):
    residualized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        model = LinearRegression()
        model.fit(covariates, data[:, i])
        predicted = model.predict(covariates)
        residualized_data[:, i] = data[:, i] - predicted
    return residualized_data


# Train/test splits generation function
def generate_splits(data, n_splits, test_size):
    splits = []
    for _ in range(n_splits):
        train_idx, val_idx = train_test_split(range(len(data)), test_size=test_size, random_state=random.randint(0, 10000))
        splits.append((train_idx, val_idx))
    return splits

################### RESIDUALIZATION and SPLITS FUNCTIONS ######################
###############################################################################





###############################################################################
####################### ITERATIONS (EXTERNAL LOOP) ############################

# Calculate the number of subjects to include in the train set
num_total_subjects = dataset1.shape[0]  
number_of_subjects_train = int(num_total_subjects * (1 - test_ratio)) 

# Store results for each iteration
results = []

for iteration in range(1, n_iterations + 1):
    print("\n\n------------------------------------------------------------------------")
    print(f"STARTING ITERATION {iteration}...")

    # Randomly select indices for the training set
    train_indices = random.sample(range(len(dataset1)), number_of_subjects_train)
    test_indices = list(set(range(len(dataset1))) - set(train_indices))

    # Split the data manually
    # Generate train and test subsamples and relative covariates
    dataset1_train = dataset1[train_indices]
    dataset2_train = dataset2[train_indices]
    covariates_DS1_train = covariates_DS1[train_indices]
    covariates_DS2_train = covariates_DS2[train_indices]

    dataset1_test = dataset1[test_indices]
    dataset2_test = dataset2[test_indices]
    covariates_DS1_test = covariates_DS1[test_indices]
    covariates_DS2_test = covariates_DS2[test_indices]
    
    print("Performing data residualization and standardization...")

    # Perform residualization on train and test data
    dataset1_train = residualize(dataset1_train, covariates_DS1_train)
    dataset2_train = residualize(dataset2_train, covariates_DS2_train)
    dataset1_test = residualize(dataset1_test, covariates_DS1_test)
    dataset2_test = residualize(dataset2_test, covariates_DS2_test)

    # Standardize the residualized training data
    scaler_brain = StandardScaler().fit(dataset1_train)
    scaler_behavior = StandardScaler().fit(dataset2_train)

    dataset1_train = scaler_brain.transform(dataset1_train)
    dataset2_train = scaler_behavior.transform(dataset2_train)

    # Apply the same scaler to the test data
    # This step ensures that the test data is scaled consistently with the training data while avoiding data leakage
    dataset1_test = scaler_brain.transform(dataset1_test)
    dataset2_test = scaler_behavior.transform(dataset2_test)

####################### ITERATIONS (EXTERNAL LOOP) ############################
###############################################################################





###############################################################################
##################### RESAMPLING SPLITS (INTERNAL LOOP) #######################


    splits = generate_splits(dataset1_train, n_splits=n_splits, test_size=validation_ratio)

    # Storage for best parameters across splits and canonical correlations
    canonical_correlations_grid = {}
    
    
    # Create train - validation splits within train set to optimize hyperparameters 
    print(f"Performing grid search for iteration {iteration}...")
    for split_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = dataset1_train[train_idx], dataset1_train[val_idx]
        Y_train, Y_val = dataset2_train[train_idx], dataset2_train[val_idx]

        # Grid search over all parameters combinations for sCCA
        for x_pen in x_penalties:
            for y_pen in y_penalties:
                # Fit the sCCA model on the train subset with the current hyperparameter combination 
                Wx, Wy, correlation_info = cca(X_train, Y_train, penaltyx=x_pen, penaltyz=y_pen, K=n_components)
                X_val_scores = X_val @ Wx # Project X weights on validation set
                Y_val_scores = Y_val @ Wy # Project Y weights on validation set
                
                # Store the 1st canonical correlation to evaluate hyperparameter performance 
                first_corr = np.corrcoef(X_val_scores[:, 0], Y_val_scores[:, 0])[0, 1]
                canonical_correlations_grid.setdefault((x_pen, y_pen), []).append(first_corr)
    
    # Calculate average correlation of the 1st canonical couple for each parameter pair
    average_canonical_correlations = {
        params: np.mean(corrs)
        for params, corrs in canonical_correlations_grid.items()
    }

    # Select optimal parameters based on maximum average correlation for the first canonical pair
    optimal_params = max(average_canonical_correlations, key=average_canonical_correlations.get)

    # Compute canonical weights under best L1 combination on the entire training set
    Wx, Wy, _ = cca(dataset1_train, dataset2_train, penaltyx=optimal_params[0], penaltyz=optimal_params[1], K=n_components)

    # Calculate canonical correlations for the entire training set (inner correlation)
    X_scores_train = dataset1_train @ Wx
    Y_scores_train = dataset2_train @ Wy
    canonical_correlations_train = [
        np.corrcoef(X_scores_train[:, i], Y_scores_train[:, i])[0, 1] for i in range(n_components)
    ]
    
##################### RESAMPLING SPLITS (INTERNAL LOOP) #######################
###############################################################################



    

###############################################################################
########################### ITERATIONS (EXTERNAL LOOP) ###########################

    # Compute test scores projecting train weights on test data
    X_scores_test = dataset1_test @ Wx
    Y_scores_test = dataset2_test @ Wy
    
    # Calculate out-of-sample correlations 
    out_of_sample_correlations = [
        np.corrcoef(X_scores_test[:, i], Y_scores_test[:, i])[0, 1] for i in range(n_components)
    ]

########################### ITERATIONS (EXTERNAL LOOP) ###########################
###############################################################################





###############################################################################
############################ PERMUTATION TEST #################################
    
    # Create an empty list for the shuffled correlations for each component
    permuted_correlations = [[] for _ in range(n_components)]
    
    print(f"Running permutation test for iteration {iteration}...")
    
    for perm in range(n_perm_outer_test):
        # Randomly permute the behavior data to break any existing relationships
        permuted_dataset2 = np.random.permutation(dataset2_test)
        
        # Compute the permuted scores for X and Y
        # The relationship is broken by shuffling Y dataset
        X_scores_permuted = dataset1_test @ Wx
        Y_scores_permuted = permuted_dataset2 @ Wy
        
        for i in range(n_components):
            
            # Calculate canonical correlations for shuffled data
            permuted_corr = np.corrcoef(X_scores_permuted[:, i], Y_scores_permuted[:, i])[0, 1]
            permuted_correlations[i].append(permuted_corr)

    # Calculate p values by comparing actual and shuffled canonical correlations
    permuted_correlations = [np.array(pc) for pc in permuted_correlations]
    p_values = [
        np.mean(permuted_corr >= out_of_sample_correlations[i])
        for i, permuted_corr in enumerate(permuted_correlations)
    ]
    
        
############################ PERMUTATION TEST #################################
###############################################################################





###############################################################################
################################ PRINT RESULTS ################################

    # Create a dictionary to store the results of each outer fold
    result = {
        "iteration": iteration,
        "optimal_params": optimal_params,
        "canonical_correlations_train": canonical_correlations_train,
        "out_of_sample_correlations_test": out_of_sample_correlations,
        "p_values": [round(p, 5) for p in p_values],
        "weights_X": Wx,
        "weights_Y": Wy
    }

    results.append(result)

    print(f"\nITERATION {iteration} RESULTS:")
    print(f"Optimal Parameters: {optimal_params}")
    print(f"Canonical Correlations (Train):")
    for i, corr in enumerate(canonical_correlations_train):
        print(f"Canonical Pair {i + 1}: {corr:.4f}")
    for i in range(n_components):
        print(f"Canonical Pair {i + 1} (Test):")
        print(f"Out-of-sample Canonical Correlation: {out_of_sample_correlations[i]}")
        print(f"P-value: {p_values[i]:.4f}")
    print("------------------------------------------------------------------------")
    
################################ PRINT RESULTS ################################
###############################################################################



        

###############################################################################
############################# SAVE MODEL SUMMARIES ############################
        
# Initialize the .txt file 
with open(output_file_txt, "w") as f:
    for result in results:
        f.write(f"Iteration {result['iteration']} Results:\n")
        f.write(f"Optimal Parameters: {result['optimal_params']}\n")
        f.write(f"Canonical Correlations (Train - Full Training Set):\n")
        for i, corr in enumerate(result['canonical_correlations_train']):
            f.write(f"Canonical Pair {i + 1}: {corr:.4f}\n")
        for i in range(n_components):
            f.write(f"Canonical Pair {i + 1} (Test):\n")
            f.write(f"Out-of-sample Canonical Correlation: {result['out_of_sample_correlations_test'][i]}\n")
            f.write(f"P-value: {result['p_values'][i]:.5f}\n")

        



# Initialize an Excel writer
with pd.ExcelWriter(output_file_excel, engine='openpyxl') as writer:
    # Save each iteration result to different sheets
    for result in results:
        iteration = result['iteration']

        # Save X weights for all iterations
        df_weights_x = pd.DataFrame({
            f"Pair_{i+1}_Weight": Wx.flatten() for i, Wx in enumerate(result['weights_X'].T)
        })
        df_weights_x.to_excel(writer, sheet_name=f"Iteration_{iteration}_X_Weights", index=False)

        # Save Y weights for all iterations
        df_weights_y = pd.DataFrame({
            f"Pair_{i+1}_Weight": Wy.flatten() for i, Wy in enumerate(result['weights_Y'].T)
        })
        df_weights_y.to_excel(writer, sheet_name=f"Iteration_{iteration}_Y_Weights", index=False)


print(f"\n\n------------------------------------------------------------------------")
print(f":) Analyses completed. Results have been saved to the selected paths! :)")
print(f"------------------------------------------------------------------------")

############################# SAVE MODEL SUMMARIES ############################
###############################################################################
