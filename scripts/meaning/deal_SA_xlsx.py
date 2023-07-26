#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
from tqdm import tqdm

# Read data from an Excel file into a DataFrame
df = pd.read_excel('/Users/seedoilz/Downloads/res_1_sentiment_meaning_t5.xlsx')

# Function to check if two strings have only one different word at the same position
def one_position_dif(str1: str, str2: str):
    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    if len(str1_list) != len(str2_list):
        return False
    if len(str1_list) == 1:
        return True
    count = 0
    for i in range(0, len(str1_list)):
        if str1_list[i] != str2_list[i]:
            count += 1
    if count == 1:
        return True
    else:
        return False

# Initialize variables
group_no = 1
former_text = ''
row_list = []
alone_df = pd.DataFrame(columns=df.columns.tolist())
res_df = pd.DataFrame(columns=df.columns.tolist() + ['group_no'])
group_add = False
count = 0

# Iterate through each row in the DataFrame
for index, row in tqdm(df.iterrows()):
    count += 1
    if count >= 100:
        break
    if index == 0:
        former_text = row['original_text']
    # Check if the current 'original_text' is the same as the previous one
    if row['original_text'] == former_text:
        row_list.append(row)
    else:
        # If the 'original_text' is different, process the rows in 'row_list'
        if len(row_list) == 1:
            # If there is only one row, store it in 'alone_df'
            alone_df = pd.concat([alone_df, row_list[0].to_frame().T], axis=0)
            row_list = [row]
            former_text = row['original_text']
            continue
        # Process rows with the same 'original_text' to group similar 'insert_text'
        sub_list_list = []
        sub_former_text = ''
        temp_list = []
        for sub_row in row_list:
            if sub_former_text == '':
                sub_former_text = sub_row['insert_text']
                temp_list.append(sub_row)
                continue
            else:
                if one_position_dif(sub_former_text, sub_row['insert_text']):
                    sub_former_text = sub_row['insert_text']
                    temp_list.append(sub_row)
                else:
                    sub_list_list.append(temp_list)
                    sub_former_text = sub_row['insert_text']
                    temp_list = [sub_row]
        sub_list_list.append(temp_list)
        for sub_list in sub_list_list:
            if len(sub_list) == 0:
                continue
            elif len(sub_list) == 1:
                # If there is only one row in the sub-list, store it in 'alone_df'
                alone_df = pd.concat([alone_df, sub_list[0].to_frame().T], axis=0)
            else:
                # If there are multiple rows with similar 'insert_text', assign them a common group number and store in 'res_df'
                group_add = True
                for sub_row in sub_list:
                    row_to_insert = sub_row
                    row_to_insert['group_no'] = group_no
                    res_df = pd.concat([res_df, row_to_insert.to_frame().T], axis=0)
            if group_add:
                group_no = group_no + 1
                group_add = False
        row_list = [row]
    former_text = row['original_text']

# Save the result DataFrames to CSV files
alone_df.to_csv('./deberta_SA_词义理解_alone_result.csv', index=False)
res_df.to_csv('./deberta_SA_词义理解_result.csv', index=False)
