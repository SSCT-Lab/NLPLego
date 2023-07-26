#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# Function to find the inserted text between two strings
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

# Set the model to be used (either 't5' or 'deberta')
model = 'deberta'

# Load the appropriate model and tokenizer based on the selected model
if model == 't5':
    model_name = "PavanNeerudu/t5-base-finetuned-qqp"
    MODEL = AutoModelForSeq2SeqLM.from_pretrained("./t5")
    tokenizer = AutoTokenizer.from_pretrained("./t5")
    nlp = pipeline('text2text-generation', model=MODEL, tokenizer=tokenizer, device=0)
elif model == 'deberta':
    model_name = "Tomor0720/deberta-large-finetuned-qqp"
    MODEL = AutoModelForSequenceClassification.from_pretrained("./deberta")
    tokenizer = AutoTokenizer.from_pretrained("./deberta")
    nlp = pipeline('text-classification', model=MODEL, tokenizer=tokenizer, device=0)
else:
    raise Exception

# Read data from a tab-separated values (tsv) file into a DataFrame
df = pd.read_csv('./qqp_lego.tsv', sep='\t')

# Function to get the label (duplicate or not_duplicate) for a pair of questions using the selected model
def t5_get_label(question1, question2, nlp):
    input_text = "qqp question1: " + question1 + "question2: " + question2
    res = nlp(input_text)
    if 'generated_text' in res[0]:
        return res[0]['generated_text']
    else:
        return ''

def deberta_get_label(question1, question2, nlp):
    input_text = question1 + " " + question2
    res = nlp(input_text)
    if res[0]['label'] == 'LABEL_0':
        return 'not_duplicate'
    elif res[0]['label'] == 'LABEL_1':
        return 'duplicate'
    else:
        return "ERROR"

def get_label(question1, question2, nlp):
    if model == 't5':
        return t5_get_label(question1, question2, nlp)
    elif model == 'deberta':
        return deberta_get_label(question1, question2, nlp)
    else:
        return ''

# Initialize variables
former_text = ''
row_list = []
res_df = pd.DataFrame(columns=['group_no', 'original_text', 'text_a', 'insert_text_a', 'text_b', 'insert_text_b', 'wrong_reason'])
group_no = 1
group_add = False

# Iterate through each row in the DataFrame
for index, row in tqdm(df.iterrows()):
    if index == 0:
        former_text = row['text_a']
    # Check if the current 'text_a' is the same as the previous one
    if str(row['text_a']) == str(former_text):
        row_list.append(row)
    else:
        # Process each pair of rows with the same 'text_a'
        for i in range(len(row_list)-1):
            for j in range(i+1, len(row_list)):
                insert_text_a = find_insert_text(row_list[i]['text_a'], row_list[i]['text_b'])
                insert_text_b = find_insert_text(row_list[j]['text_a'], row_list[j]['text_b'])
                insert_same = get_label(insert_text_a, insert_text_b, nlp)
                context_same = get_label(row_list[i]['text_b'], row_list[j]['text_b'], nlp)
                if insert_same != context_same:
                    group_add = True
                    if insert_same == 'duplicate':
                        # Create a new row with information about the pair of questions and the wrong reason
                        new_row = {'group_no': group_no, 'original_text': row_list[i]['text_a'], 'text_a': row_list[i]['text_b'], 'insert_text_a': insert_text_a, 'text_b': row_list[j]['text_b'], 'insert_text_b': insert_text_b, 'wrong_reason': 'same meaning insert, different meaning context'}
                        # Append the new row to the result DataFrame
                        res_df.loc[len(res_df)] = new_row
                    else:
                        # Create a new row with information about the pair of questions and the wrong reason
                        new_row = {'group_no': group_no, 'original_text': row_list[i]['text_a'], 'text_a': row_list[i]['text_b'], 'insert_text_a': insert_text_a, 'text_b': row_list[j]['text_b'], 'insert_text_b': insert_text_b, 'wrong_reason': 'same meaning context, different meaning insert'}
                        # Append the new row to the result DataFrame
                        res_df.loc[len(res_df)] = new_row
        if group_add:
            group_no = group_no + 1
            group_add = False
        former_text = str(row['text_a'])
        row_list = [row]
    former_text = str(row['text_a'])

# Save the result DataFrame to a CSV file
res_df.to_csv(model + '_SSM_词义理解_result.csv', index=False)
