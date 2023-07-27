import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

def get_dependency(question):
    # get dependency in the question, return a list containing the result, where 1 stands for the dependency exists, 0 for not
    res = []
    dependency = nlp.dependency_parse(question)
    flag = True
    for relation in dependency:
        # coordination
        if 'conj' in relation or 'cc' in relation:
            res.append(1)
            flag = False
            break
    if flag:    
        res.append(0)
    # judge the causality and hypothesis through specific tokens
    causality_tokens = ['why', 'because', 'because of', 'since']
    hypothesis_tokens = ['if', 'only if', 'if only', 'unless', 'lest', 'otherwise', 'as soon as', 'as long as', 'in case', 
                        'suppose that', 'supposing that', 'provided that', 'providing that', 'when', 'whenever', 'with', 'if it is the case'
                        'in this scene', 'once']
    flag = True
    for token in causality_tokens:
        # causality
        if token in question:
            res.append(1)
            flag = False
            break
    if flag:
        res.append(0)
    
    flag = True
    for token in hypothesis_tokens:
        # hypothesis
        if token in question:
            res.append(1)
            flag = False
            break
    if flag:
        res.append(0)
    return res

# data and result, depending on the model used
data = './deberta_mrc_result.csv'
result = './deberta_logic_mrc_all_result.csv'

nlp = StanfordCoreNLP(r'./stanford-corenlp-4.5.4')

df = pd.read_csv(data, keep_default_na=False, encoding='utf-8-sig')
# coordination, causality, hypothesis
df['coordination'] = 0
df['causality'] = 0
df['hypothesis'] = 0
df_copy = df.copy()
df_copy = df_copy.iloc[0:0]

for index, row in tqdm(df.iterrows()):
    if row['result'] == 0:
        # if the result is wrong, get the dependency of question
        res = get_dependency(row['question'], eval(row['answers']))
        row['coordination'] = res[0]
        row['causality'] = res[1]
        row['hypothesis'] = res[2]
        df_copy = df_copy.append(row)

# store the result
df_copy.to_csv(result, index=False, encoding='utf-8-sig')