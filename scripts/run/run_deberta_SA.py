# Import necessary libraries
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import re
from tqdm import tqdm
from openpyxl import Workbook

# Load the pre-trained model and tokenizer
MODEL = AutoModelForSequenceClassification.from_pretrained("/mnt/deberta_SA")
tokenizer = AutoTokenizer.from_pretrained("/mnt/deberta_SA")
nlp = pipeline('text-classification', model=MODEL, tokenizer=tokenizer, device=0)

# Set the input file name
input = 'checklist_sst_change'

# Create a new workbook and its sheet to store the results
workbook = Workbook()
sheet = workbook.active

# Open and read the input text file
with open(input + '.txt', 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        if line.startswith('sent_id') or line.startswith('insert') or line == '\n':
            continue
        # Split the input line into separate texts
        texts = line.split('|')
        for i in range(0, len(texts)):
            texts[i] = re.sub(r'\s+', ' ', texts[i]).strip()
        sentiments = []
        scores = []
        # Iterate through each text and classify its sentiment using the pre-trained model
        for text in texts:
            output_list = nlp('sentiment: ' + text)
            if output_list[0]['label'] == 'positive':
                sentiments.append(1)
                scores.append([1 - output_list[0]['score'], output_list[0]['score']])
            else:
                sentiments.append(0)
                scores.append([output_list[0]['score'], 1 - output_list[0]['score']])
        data = []
        # Append the text, sentiment, and scores to the sheet
        for i in range(0, len(texts)):
            data.append(texts[i])
            data.append(sentiments[i])
            data.append(scores[i][0])
            data.append(scores[i][1])
        sheet.append(data)

# Save and close the workbook
workbook.save(input + '_deberta.xlsx')
workbook.close()
