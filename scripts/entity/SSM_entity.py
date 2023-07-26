import pandas as pd
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_lg")

# Read data from a tab-separated values (tsv) file into a DataFrame
df = pd.read_csv('qqp_lego.tsv', sep='\t')

# Create a copy of the DataFrame to store the processed results
df_copy = df.copy()
df_copy = df_copy.iloc[0:0]

# List of entity types to be detected by spaCy
entity_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG',
               'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

# Add columns for each entity type and corresponding text to the original DataFrame and copy DataFrame
for entity_kind in entity_list:
    df['insert_text'] = ''
    df_copy['insert_text'] = ''
    df[entity_kind] = 0
    df[entity_kind + '_text'] = 'blank'
    df_copy[entity_kind] = 0
    df_copy[entity_kind + '_text'] = 'blank'


# Function to find the text that is inserted in one string to get to another string
def find_insert_text(str1, str2):
    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    i = 0
    j = 0
    res = ''
    for j in range(len(str2_list)):
        if str1_list[i] != str2_list[j]:
            res += str2_list[j] + ' '
        else:
            i += 1
        if i == len(str1_list):
            break
    return res


# Iterate through each row in the DataFrame and process the 'text_a' and 'text_b' columns
for index, row in df.iterrows():
    text_a = row['text_a']
    text_b = row['text_b']

    # Find the inserted text in 'text_b' to get from 'text_a' to 'text_b'
    insert_text = find_insert_text(text_a, text_b)
    row['insert_text'] = insert_text

    # Process the inserted text using spaCy to detect named entities
    doc = nlp(insert_text)
    if len(doc.ents) > 0:
        # Increment entity counts and store entity texts in the original DataFrame
        for ent in doc.ents:
            row[ent.label_] += 1
            if row[ent.label_] == 1:
                row[ent.label_ + '_text'] += ent.text
            else:
                row[ent.label_ + '_text'] += ',' + ent.text
        # Append the processed row to the copy DataFrame
        df_copy = pd.concat([df_copy, row.to_frame().T], axis=0)

# Save the processed copy DataFrame to a new CSV file
df_copy.to_csv('res_qqp_lego.csv', index=False)
