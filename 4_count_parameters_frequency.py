"""

MULTIPLE HOLDOUT sCCA PIPELINE

@author: Tommaso Cazzella, PhD Student
         Psychiatry and Clinical Psychobiology Unit
         Division of Neuroscience
         Vita-Salute San Raffaele University
         IRCCS San Raffaele Hospital
         
"""
from collections import Counter

def count_parameters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting optimal parameters from each iteration
    parameters = []
    for line in lines:
        if "Optimal Parameters:" in line:
            start_idx = line.index('(')
            end_idx = line.index(')') + 1
            param = line[start_idx:end_idx]
            parameters.append(param)

    # Count the frequency of each parameter
    param_counts = Counter(parameters)

    # Display the results
    print("\n\nParameter Frequency:")
    for param, count in param_counts.most_common():
        print(f"{param}: {count} times")

# Replace 'your_file_path.txt' with the actual file path
file_path = ''
count_parameters(file_path)
