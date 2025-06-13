"""

MULTIPLE HOLDOUT sCCA PIPELINE

@author: Tommaso Cazzella, PhD Student
         Psychiatry and Clinical Psychobiology Unit
         Division of Neuroscience
         Vita-Salute San Raffaele University
         IRCCS San Raffaele Hospital
         
"""

import re
import pandas as pd

##############################################################
############## INPUT TO BE MODIFIED BY THE USER ##############

# Load the txt file produced by sCCA pipeline
input_file = '' # .txt format
output_file = '' # .xlsx format

############## INPUT TO BE MODIFIED BY THE USER ##############
##############################################################

# Define the regex patterns
iteration_pattern = r"Iteration (\d+) Results:"
optimal_params_pattern = r"Optimal Parameters: \(([^,]+), ([^\)]+)\)"
canonical_train_pattern = r"Canonical Pair (\d+): ([\d.-]+)"
canonical_test_pattern = r"Canonical Pair (\d+) \(Test\):\nOut-of-sample Canonical Correlation: ([\d.-]+)\nP-value: ([\d.]+)"

# Initialize a list to store results
results = []

# Read the input file
with open(input_file, 'r') as file:
    data = file.read()

# Extract data for each iteration
iterations = re.split(iteration_pattern, data)

# Loop through the split content to include all iterations, starting from the first
for i in range(1, len(iterations), 2):  # Start at index 1 to include "Iteration 1"
    iteration = iterations[i]  # The iteration number
    content = iterations[i + 1]  # The corresponding data block

    # Extract optimal parameters
    optimal_params_match = re.search(optimal_params_pattern, content)
    if optimal_params_match:
        param1, param2 = optimal_params_match.groups()
    else:
        param1, param2 = None, None

    # Extract training canonical correlations
    train_canonical_matches = re.findall(canonical_train_pattern, content)
    train_canonicals = {int(pair): float(correlation) for pair, correlation in train_canonical_matches}

    # Extract testing canonical correlations and p-values
    test_canonical_matches = re.findall(canonical_test_pattern, content)
    for pair, correlation, p_value in test_canonical_matches:
        results.append({
            'Iteration': int(iteration),
            'Optimal Param 1': param1,
            'Optimal Param 2': param2,
            'Canonical Pair': int(pair),
            'Train Correlation': train_canonicals.get(int(pair), None),
            'Test Correlation': float(correlation),
            'P-value': float(p_value)
        })


# Convert results to DataFrame
df = pd.DataFrame(results)

# Save results to an Excel file
df.to_excel(output_file, index=False)

print(f"Results have been organized and saved to {output_file}")
