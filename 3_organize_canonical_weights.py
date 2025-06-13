"""

MULTIPLE HOLDOUT sCCA PIPELINE

@author: Tommaso Cazzella, PhD Student
         Psychiatry and Clinical Psychobiology Unit
         Division of Neuroscience
         Vita-Salute San Raffaele University
         IRCCS San Raffaele Hospital
         
"""

import pandas as pd


######################################################################
################## INPUT TO BE MODIFIED BY THE USER ##################

# Parameters
input_file = '' # provide the xlsx path of canonical weights (output of sCCA code)
output_file = '' # .xlsx format
weight_identifier = 'Pair_1_Weight'  # Specify the column identifier. In this case, the code saves the canonical weights for the first canonical pair

################## INPUT TO BE MODIFIED BY THE USER ##################
######################################################################


def extract_specific_weights(input_file, output_file, weight_identifier):
    # Load the Excel file
    excel_data = pd.ExcelFile(input_file)

    # Filter the sheet names for X and Y Weights
    x_weights_sheets = [sheet for sheet in excel_data.sheet_names if "_X_Weights" in sheet]
    y_weights_sheets = [sheet for sheet in excel_data.sheet_names if "_Y_Weights" in sheet]

    # Define a function to filter columns
    def filter_columns(sheet, identifier):
        df = excel_data.parse(sheet)
        return df[[col for col in df.columns if identifier in col]]

    # Combine the filtered data from X Weights sheets
    x_weights_combined = pd.concat([filter_columns(sheet, weight_identifier) for sheet in x_weights_sheets], axis=1)

    # Combine the filtered data from Y Weights sheets
    y_weights_combined = pd.concat([filter_columns(sheet, weight_identifier) for sheet in y_weights_sheets], axis=1)

    # Save the filtered data into a new Excel file
    with pd.ExcelWriter(output_file) as writer:
        x_weights_combined.to_excel(writer, sheet_name='X Weights', index=False)
        y_weights_combined.to_excel(writer, sheet_name='Y Weights', index=False)



# Run the extraction
extract_specific_weights(input_file, output_file, weight_identifier)

