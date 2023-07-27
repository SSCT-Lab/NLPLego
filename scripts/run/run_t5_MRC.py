from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm
import re

# get output result of the model
def get_answer(question, context):
    input_text = "question: %s  context: %s" % (question, context)
    features = tokenizer([input_text], return_tensors='pt').to("cuda:0")

    output = model.generate(input_ids=features['input_ids'].to('cuda:0'), 
               attention_mask=features['attention_mask'].to('cuda:0'))
  
    return tokenizer.decode(output[0])

# judge whether the answer is correct with the metamorphic relation
def get_result(answers_set, answer, is_leaf_node):
    if is_leaf_node == 0:
        # if the question is not leaf node, the answer should be null 
        if answer == '':
            return True, ''
        else :
            return False, 'expected answer is null, but answer is not null'
    if not answers_set: 
        # if the answers_set is null, the answer should be null
        if answer == '':
            return True, ''
        else:
            return False, 'answer_set is empty, but answer is not null'
    if answer in answers_set:
        return True, ''
    else:
        return False, 'answer_set is not empty, but answer not in answer_set'

# judge whether the origin_answer is correct
def get_origin_result(answer, answers_set):
    if not answers_set:
        return answer == ''
    return answer in answers_set

# judge whether the generated output should be regarded as no answer
def is_answer_null(answer):
    pattern = re.compile(r'.*(add|subtract|multiply|divide|factorial|choose|power|lcm)[ ]?\(.*?\)')
    result = re.findall(pattern, answer)
    return result == []

if __name__ == "__main__":
    model_name = "mrm8488/t5-base-finetuned-squadv2"

    data = './squad_ques_context_input_answer.tsv'
    result = './t5_mrc_result.csv'

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to("cuda:0")

    df = pd.read_csv(data, keep_default_na=False, sep='\t')
    df_copy = df.copy()
    df_copy['origin_answer'] = ''
    df_copy['origin_result'] = 0
    df_copy['answer'] = ''
    df_copy['result'] = 0
    df_copy['wrong_reason'] = ''

    for id, row in tqdm(df_copy.iterrows()):
        origin = row['origin']
        question = row['question']
        context = row['context']
        answers = eval(row['answers'])
        origin_answer = get_answer(origin, context)
        answer = get_answer(question, context)
        df_copy.at[id, 'origin_answer'] = '' if is_answer_null(origin_answer) else origin_answer
        df_copy.at[id, 'answer'] = '' if is_answer_null(answer) else answer
        df_copy.at[id, 'origin_result'] = 1 if get_origin_result(answers) else 0
        df_copy.at[id, 'result'], df_copy[id, 'wrong_reason'] =  get_result(answers, answer, row['is_leaf_node'])
        
    df_copy.to_csv(result, index=False)    
