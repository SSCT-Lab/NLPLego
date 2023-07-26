# Import necessary libraries
import pandas as pd
from tqdm import tqdm

# Set the input file path and model type
input_file = '/Users/seedoilz/Desktop/checklist_sst_add_neg'
model = 't5'

# Read data from an Excel file into a DataFrame
df = pd.read_excel(input_file + '_' + model + '.xlsx')

# Initialize variables for sent_id and index
sent_id = 0
index = 0

# Create a new column 'sent_id' in the DataFrame and set initial values to -1
df['sent_id'] = -1

# Read data from a text file and update 'sent_id' values in the DataFrame based on the text file content
with open(input_file + '.txt') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        if line == '\n':
            continue
        elif line.startswith('sent_id'):
            # Extract the sent_id value from the line and convert it to an integer
            sent_id = int(line[10:].replace("\n", ""))
        else:
            # Update the 'sent_id' value in the DataFrame row and increment the index
            if not index >= len(df):
                df.loc[index, 'sent_id'] = sent_id
                index += 1
            else:
                # If the DataFrame is already filled, stop processing further lines from the text file
                break

# Save the updated DataFrame to a new CSV file
df.to_csv(input_file + '_' + model + '.csv')
