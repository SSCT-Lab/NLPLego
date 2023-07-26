# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import pipeline
from torch.nn.functional import softmax
import torch
import re
from tqdm import tqdm
from openpyxl import Workbook

# Set the model name and load the pre-trained T5 model and tokenizer
model_name = "michelecafagna26/t5-base-finetuned-sst2-sentiment"
model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/t5_SA")
tokenizer = AutoTokenizer.from_pretrained("/mnt/t5_SA")

# Move the model to the GPU for faster computation
model = model.to(torch.device("cuda"))

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
        labels = ["p", "n"]
        for text in texts:
            # Encode the text using the tokenizer and move it to the GPU
            input_ids = tokenizer.encode(text, return_tensors='pt').to(torch.device("cuda"))
            # Generate model output and compute probabilities using softmax
            outputs = model.generate(input_ids, output_scores=True, return_dict_in_generate=True)
            logits = outputs.scores[1]
            probs = softmax(logits, dim=-1)
            top_prob, top_idx = torch.max(probs[0], dim=0)

            # Decode the predicted label
            result_label = tokenizer.decode([top_idx])
            # Compute positive and negative probabilities based on the label
            if result_label == 'n':
                negative_prob = float(top_prob.item())
                positive_prob = 1 - negative_prob
            else:
                positive_prob = float(top_prob.item())
                negative_prob = 1 - positive_prob
            probabilities_list = []
            probabilities_list.append(negative_prob)
            probabilities_list.append(positive_prob)
            scores.append(probabilities_list)
            # Append sentiment label based on the predicted label
            if result_label == 'p':
                sentiments.append(1)
            else:
                sentiments.append(0)
        data = []
        # Append the text, sentiment, and scores to the sheet
        for i in range(0, len(texts)):
            data.append(texts[i])
            data.append(sentiments[i])
            data.append(scores[i][0])
            data.append(scores[i][1])
        sheet.append(data)

# Save and close the workbook
workbook.save(input + '.xlsx')
workbook.close()
