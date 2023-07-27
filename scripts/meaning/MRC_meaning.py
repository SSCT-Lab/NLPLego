import pandas as pd

# data and result, depending on the model used
data = './deberta_mrc_result.csv'
result = './deberta_meaning_mrc_result.csv'

def get_replaced_token(seed_question, question):
    # get replaced tokens in seed question and new question
    diff = [] # a list containing all different tokens' index
    seed_question_tokens = seed_question.split()
    question_tokens = question.split()
    if len(seed_question_tokens) != len(question_tokens):
        return None, None
    for i in range(len(question_tokens)):
        if seed_question_tokens[i] != question_tokens[i]:
            diff.append(i)
    if len(diff) != 1:
        # if there are more than one different, then the question is not expected 
        return None, None
    return seed_question_tokens[diff[0]], question_tokens[diff[0]]

df = pd.read_csv(data, keep_default_na=False, encoding='utf-8-sig')
df['origin_token'] = ''
df['replaced_token'] = ''
df_copy = df.copy()
df_copy = df_copy.iloc[0:0]

from tqdm import tqdm
cnt = 0
for index, row in tqdm(df.iterrows()):
    # if the origin question' s answer is wrong or the new question' s answer is correct, the situation should be skipped
    if row['origin_result'] == 0:
        continue
    if row['result'] == 1:
        continue
    origin_token, replaced_token = get_replaced_token(row['origin'], row['question'])
    if origin_token:
        # write down the replaced tokens
        row['origin_token'] = origin_token
        row['replaced_token'] = replaced_token
        df_copy = df_copy.append(row)
        
# store the result
df_copy.to_csv(result, index=False, encoding='utf-8-sig')