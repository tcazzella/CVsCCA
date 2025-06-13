"""

MULTIPLE HOLDOUT sCCA PIPELINE

@author: Tommaso Cazzella, PhD Student
         Psychiatry and Clinical Psychobiology Unit
         Division of Neuroscience
         Vita-Salute San Raffaele University
         IRCCS San Raffaele Hospital
         
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sparsecca._cca_pmd import cca
import numpy as np


##############################################################
############## INPUT TO BE MODIFIED BY THE USER ##############

# Load data
dataset1 = pd.read_csv('') # Load dataset 1
dataset2 = pd.read_csv('') # Load dataset 2

output_excel_path = '' # define the path of the .xlsx file to save individual canonical scores

# Verify alignment by Participant_ID (optional)
assert (dataset1["Participant_ID"] == dataset2["Participant_ID"]).all(), "Mismatch in Participant IDs"

# Drop Participant_ID column
dataset1 = dataset1.drop(columns=["Participant_ID"])
dataset2 = dataset2.drop(columns=["Participant_ID"])

penaltyx = 0.9 # select the best regularization parameter for dataset1 according to the count of parameter frequency across the resamplings
penaltyz = 1 # select the best regularization parameter for dataset1 according to the count of parameter frequency across the resamplings
n_components = 1 # define the number of canonical variates

n_permutations = 10000
n_bootstraps = 5000

############## INPUT TO BE MODIFIED BY THE USER ##############
##############################################################

# Convert to numpy arrays
dataset1 = dataset1.values
dataset2 = dataset2.values

# Standardize data
scaler = StandardScaler()
dataset1 = scaler.fit_transform(dataset1)
dataset2 = scaler.fit_transform(dataset2)

Wx, Wy, _ = cca(dataset1, dataset2, penaltyx=penaltyx, penaltyz=penaltyz, K=n_components)

# Compute canonical correlations
dataset1_scores = dataset1 @ Wx
dataset2_scores = dataset2 @ Wy
canonical_correlations = [
    np.corrcoef(dataset1_scores[:, i], dataset2_scores[:, i])[0, 1]
    for i in range(n_components)
]

# Permutation test for p-values
permuted_correlations = np.zeros((n_permutations, n_components))

for perm in range(n_permutations):
    print(f"running permutation {perm + 1}...")
    # Shuffle dataset2
    permuted_dataset2 = np.random.permutation(dataset2)
    
    # Perform SCCA on permuted data
    permuted_Wx, permuted_Wy, _ = cca(dataset1, permuted_dataset2, penaltyx=penaltyx, penaltyz=penaltyz, K=n_components)
    
    # Compute scores for permuted data
    permuted_dataset1_scores = dataset1 @ permuted_Wx
    permuted_dataset2_scores = permuted_dataset2 @ permuted_Wy
    
    # Compute canonical correlations for permuted data
    permuted_correlations[perm, :] = [
        np.corrcoef(permuted_dataset1_scores[:, i], permuted_dataset2_scores[:, i])[0, 1]
        for i in range(n_components)
    ]

# Calculate p-values
p_values = [
    (np.sum(permuted_correlations[:, i] >= canonical_correlations[i]) + 1) / (n_permutations + 1)
    for i in range(n_components)
]

# Bootstrap for confidence intervals
bootstrap_Wx = np.zeros((n_bootstraps, Wx.shape[0]))
bootstrap_Wy = np.zeros((n_bootstraps, Wy.shape[0]))

for bootstrap in range(n_bootstraps):
    print(f"running bootstrap {bootstrap + 1}...")
    # Sample with replacement
    bootstrap_indices = np.random.choice(dataset1.shape[0], dataset1.shape[0], replace=True)
    bootstrap_dataset1 = dataset1[bootstrap_indices]
    bootstrap_dataset2 = dataset2[bootstrap_indices]
    
    # Perform SCCA on bootstrap sample
    bootstrap_Wx_temp, bootstrap_Wy_temp, _ = cca(bootstrap_dataset1, bootstrap_dataset2, penaltyx=penaltyx, penaltyz=penaltyz, K=n_components)
    
    # Store weights
    bootstrap_Wx[bootstrap, :] = bootstrap_Wx_temp[:, 0]
    bootstrap_Wy[bootstrap, :] = bootstrap_Wy_temp[:, 0]

# Compute 95% confidence intervals
Wx_CI = np.percentile(bootstrap_Wx, [2.5, 97.5], axis=0)
Wy_CI = np.percentile(bootstrap_Wy, [2.5, 97.5], axis=0)

# Stack weights with confidence intervals
Wx_with_CI = np.column_stack((Wx[:, 0], Wx_CI.T))
Wy_with_CI = np.column_stack((Wy[:, 0], Wy_CI.T))

# Save canonical scores to Excel
dataset1_scores_df = pd.DataFrame(dataset1_scores, columns=[f"Dataset1_Score_{i+1}" for i in range(n_components)])
dataset2_scores_df = pd.DataFrame(dataset2_scores, columns=[f"Dataset2_Score_{i+1}" for i in range(n_components)])

canonical_scores_df = pd.concat([dataset1_scores_df, dataset2_scores_df], axis=1)
canonical_scores_df.to_excel(output_excel_path, index=False)

# Output results
print("Canonical Correlations:", canonical_correlations)
print("P-Values:", p_values)

