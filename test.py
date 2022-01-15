import json

def get_sentence_len(file_name):
    path = "./Squad2/" + file_name + ".txt"
    orig_sents = open(path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    result = []
    sent_result_all = []

    while sent:
        sent = sent[:-1]
        if "context_id=" in sent:
            num = 0
            sent = orig_sents.readline()
            sent = sent[:-1]
            sent_result = []
            while sent != "":
                num += 1
                sent_result.append(sent)
                sent = orig_sents.readline()
                sent = sent[:-1]
            result.append(num)
            sent_result_all.append(sent_result)
            num = 0
        sent = orig_sents.readline()
    return result, sent_result_all

def gen_new_json():
    file_name = "dev-v2.0.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)

    file_name = "dev_v2.0_clear.json"
    path = "./" + file_name
    with open(path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(prediction_json, indent=4, ensure_ascii=False))

def valid(sent_li, context):
    result = True
    for i in sent_li:
        result &= i in context
    return result


def gen_context_json(sent_list):
    file_name = "dev_null.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    null_json = json.load(predict_file)
    null_data = null_json["data"]
    start_json = null_data[0]["paragraphs"]
    file_name = "dev_v2.0_clear.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)
    prediction_data = prediction_json["data"]
    for sent_li in sent_list:
        for i in prediction_data:
            for j in i["paragraphs"]:
                if sent_li[0] in j["context"].replace("\n", ""):
                    if valid(sent_li, j["context"].replace("\n", "")):
                        start_json.append(j)
                    else:
                        print(sent_li)
                        print(j["context"])
    file_name = "dev_start.json"
    path = "./Squad2/" + file_name
    with open(path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(null_json, indent=4, ensure_ascii=False))

def read_json():
    file_name = "dev_start.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)
    prediction_data = prediction_json["data"]
    p = prediction_data[0]["paragraphs"]
    print(len(p))

def read_question_index():
    file_name = "ans_context.txt"
    path = "./Squad2/" + file_name
    orig_sents = open(path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    result = []
    temp = []
    while sent:
        sent = sent[:-1]
        if "context_id = " in sent:
            index = sent.split("context_id = ")[1]
            sent = orig_sents.readline()[:-1]
            while sent != "==========FIN==========":
                if "context_idx: " in sent:
                    temp.append(sent.split("context_idx: ")[1])
                sent = orig_sents.readline()[:-1]
            result.append(temp)
            temp = []
        sent = orig_sents.readline()
    return result


if __name__ == '__main__':
    # gen_new_json()
    # 获取所有句子的<=beam_size个变体
    # file_name = "context"
    #
    # # 获取context里句子个数的列表
    # context_sentence_len_list, origin_sent_list = get_sentence_len(file_name)
    # print(len(context_sentence_len_list))
    # gen_context_json(origin_sent_list)
    # read_json()

    result = read_question_index()
    print(result)
    print(len(result))


