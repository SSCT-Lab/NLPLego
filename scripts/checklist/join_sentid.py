# Import necessary libraries
import pandas as pd
from tqdm import tqdm

# Read the original Excel file into a DataFrame
original_df = pd.read_excel("/Users/seedoilz/Desktop/sst_t5.xlsx")

# Define the name of the CSV file to be read
file = "checklist_sst_not_change_t5.csv"

# Read the CSV file into another DataFrame
df = pd.read_csv("/Users/seedoilz/Desktop/" + file)

# Create an empty DataFrame with the combined columns from both DataFrames
res_df = pd.DataFrame(columns=(original_df.columns.append(df.columns)))

# Loop through each row in the CSV DataFrame using tqdm for progress visualization
for index, row in tqdm(df.iterrows()):
    # Retrieve the corresponding row from the original DataFrame based on 'sent_id'
    original_row = original_df.iloc[int(row['sent_id'])]

    # Create a new row containing selected columns from both DataFrames
    insert_row = {
        'sent_id': row['sent_id'],
        'original_text': original_row['original_text'],
        'original_sentiment': original_row['original_sentiment'],
        'original_neg_score': original_row['original_neg_score'],
        'original_pos_score': original_row['original_pos_score'],
        'res_text': row['res_text'],
        'res_sentiment': row['res_sentiment'],
        'res_neg_score': row['res_neg_score'],
        'res_pos_score': row['res_pos_score']
    }

    # Add the new row to the result DataFrame
    res_df.loc[len(res_df)] = insert_row

# Drop the column named 'Unnamed: 0' from the result DataFrame
res_df.drop(columns='Unnamed: 0', inplace=True)

# Write the result DataFrame to a new CSV file with a prefix 'JOIN_'
res_df.to_csv("/Users/seedoilz/Desktop/" + "JOIN_" + file)
