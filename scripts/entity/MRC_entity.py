import pandas as pd
import spacy

# data and result, depending on the model used 
data = './deberta_mrc_result.csv'
result = './deberta_entity_mrc_result_all.csv'
from tqdm import tqdm

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

df = pd.read_csv(data, keep_default_na=False, encoding='utf-8-sig')
# list the entity name
entity_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
for entity_kind in entity_list:
    df[entity_kind] = 0
    df[entity_kind + '_text'] = ''
    df[entity_kind + '_start_end_error'] = 0
df_copy = df.copy()
df_copy = df_copy.iloc[0:0]

for index, row in tqdm(df.iterrows()):
    if row['result'] == 1:
        continue
    exp_answers = eval(row['answers'])
    if row['answer_starts'] is None:
        exp_answer_starts = []
    else:
        exp_answer_starts = eval(row['answer_starts'])
    answer = row['answer_can_be_null']
    for i in range(len(exp_answers)):
        doc = nlp(exp_answers[i])
        answer_doc = ()
        if len(doc.ents) > 0:
            # look up for answer entity
            row['answers'] = exp_answers[i]
            row['answer_starts'] = exp_answer_starts[i]
            for ent in doc.ents:
                row[ent.label_] += 1
                if row[ent.label_] == 1:
                    row[ent.label_ + '_text'] += ent.text
                else:
                    row[ent.label_ + '_text'] += '|' + ent.text
            if answer != '':
                answer_doc = nlp(answer)
            else:
                # answer is null, no entity in it
                df_copy = df_copy.append(row)
                continue
            if len(answer_doc.ents) > 0:
                # check whether the answer entity location error occurred
                for ent in answer_doc.ents:
                    texts = row[ent.label_+'_text'].split('|')
                    for text in texts:
                        if ent.text in text and ent.text != text:
                            row[ent.label_+'_start_end_error'] = 1
                            if row[ent.label_+'_start_end_error'] != 1:
                                print('Error')
            df_copy = df_copy.append(row)

# store the result
df_copy.to_csv(result, index=False, encoding='utf-8-sig')