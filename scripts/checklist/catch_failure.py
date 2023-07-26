#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
from tqdm import tqdm

# Define a function to check if the row represents a failure based on the mode
'''
add_neg : 0
add_pos : 1
change : 2
not_change : 3
'''
def isFailure(mode, row):
    if mode == 0:
        change = row['res_pos_score'] - row['original_pos_score']
        if change >= 0.1:
            return True
        else:
            return False
    elif mode == 1:
        change = row['res_neg_score'] - row['original_neg_score']
        if change >= 0.1:
            return True
        else:
            return False
    elif mode == 2:
        change = abs(row['res_pos_score'] - row['original_pos_score'])
        if change <= 0.1:
            return True
        else:
            return False
    elif mode == 3:
        change = abs(row['res_pos_score'] - row['original_pos_score'])
        if change >= 0.1:
            return True
        else:
            return False
    else:
        raise Exception

# Define the file name and read the CSV file into a DataFrame
file = "add_pos_t5.csv"
df = pd.read_csv("/Users/seedoilz/Desktop/JOIN_checklist_sst_" + file)

# Determine the mode based on the file name
mode = -1
if "add_neg" in file:
    mode = 0
elif "add_pos" in file:
    mode = 1
elif "not_change" in file:
    mode = 3
elif "change" in file:
    mode = 2

# Create an empty DataFrame with the same columns as the original DataFrame
res_df = pd.DataFrame(columns=df.columns)

# Loop through each row in the DataFrame and check if it represents a failure based on the mode
for index, row in tqdm(df.iterrows()):
    if isFailure(mode, row):
        # If it's a failure, add the row to the result DataFrame
        res_df.loc[len(res_df)] = row

# Save the result DataFrame to a new CSV file
res_df.to_csv("/Users/seedoilz/Desktop/result_checklist_sst_" + file)
