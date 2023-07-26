#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import spacy
from tqdm import tqdm
import os

# Set the directory path to the current directory (where the script is located)
dir_path = '.'

# Iterate through each file in the directory
for file_path in os.listdir(dir_path):
    # Skip files with extensions 'csv' or 'py'
    if file_path.endswith('csv') or file_path.endswith('py'):
        continue

    # Read data from an Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Create a copy of the DataFrame to store the processed results
    df_copy = df.copy()
    df_copy = df_copy.iloc[0:0]  # Clear the copy DataFrame

    # List of entity types to be detected by spaCy
    entity_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL',
                   'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    # Add columns for each entity type and corresponding text to the original DataFrame and copy DataFrame
    for entity_kind in entity_list:
        df[entity_kind] = 0
        df[entity_kind + '_text'] = 'blank'
        df_copy[entity_kind] = 0
        df_copy[entity_kind + '_text'] = 'blank'

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_lg")

    # Iterate through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        judge_text = row['insert_text']
        doc = nlp(judge_text)

        # Check if there are entities detected in the text
        if len(doc.ents) > 0:
            # Process each entity in the text
            for ent in doc.ents:
                # Increment the count for the entity type in the original DataFrame
                row[ent.label_] += 1

                # If it's the first occurrence of the entity type, store the entity text as it is
                if row[ent.label_] == 1:
                    row[ent.label_ + '_text'] = ent.text
                else:
                    # If there are multiple occurrences of the entity type, concatenate the entity text
                    row[ent.label_ + '_text'] += ',' + ent.text

            # Append the processed row to the copy DataFrame
            df_copy = pd.concat([df_copy, row.to_frame().T], axis=0)

    # Save the processed copy DataFrame to a new CSV file
    df_copy.to_csv('res' + file_path.replace('xlsx', 'csv'), index=False)
