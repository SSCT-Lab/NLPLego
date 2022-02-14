import nltk
from nltk.corpus import wordnet
from gen_temp import *
import re
import copy
import time,hashlib
import itertools
from nltk import CoreNLPParser
from nltk.corpus import stopwords
import spacy
import transformers
import json
import random
import requests

nlp = spacy.load("en_core_web_sm")
sbar_pattern = re.compile(r't\d+')
unmasker = transformers.pipeline('fill-mask', model='bert-base-uncased')
BERT_SCORE = 0.1

## Remove punctuation and named entities
def filer_word(pos_list, adjunct, ner_list):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    stops = set(stopwords.words("english"))
    #tag_list = nltk.pos_tag(adjunct_word)
    doc = nlp(adjunct)
    word_pos = [tok.pos_ for tok in doc]
    masked_word = []
    masked_adjunct = []
    adjunct_word = [tok.text for tok in doc]
    if "-" in adjunct_word:
        index = adjunct_word.index("-")
    for i in range(len(adjunct_word)):
        word = adjunct_word[i]
        if "-" in adjunct_word:
            if i not in [index - 1, index, index + 1]:
                for ner in ner_list:
                    if word in ner:
                        continue
                if (word not in stops) & (word not in english_punctuations) & (word_pos[i] in pos_list):
                    masked_word.append(word)
                    temp_phrase = list(adjunct_word)
                    temp_phrase[i] = "[MASK]"
                    masked_adjunct.append(" ".join(temp_phrase).replace(" - ", "-"))
        else:
            for ner in ner_list:
                if word in ner:
                    continue
            if (word not in stops) & (word not in english_punctuations) & (word_pos[i] in pos_list):
                masked_word.append(word)
                temp_phrase = list(adjunct_word)
                temp_phrase[i] = "[MASK]"
                masked_adjunct.append(" ".join(temp_phrase).replace(" - ", "-"))

    if len(masked_adjunct) == 0:
        masked_adjunct.append(" ".join(adjunct_word).replace(" - ", "-"))
        masked_word.append("X")

    return masked_word, masked_adjunct


def gen_mask_phrase(adjunct_list, pos_list, all_ner):
    all_masked_adjunct = []
    all_masked_word = []
    for i in range(len(adjunct_list)):
        adjuncts = adjunct_list[i]
        ner_list = all_ner[i]
        masked_adjunct_list = []
        masked_word_list = []
        for adjunct in adjuncts:
            masked_word, masked_adjunct = filer_word(pos_list, adjunct, ner_list)
            masked_word_list.append(masked_word)
            masked_adjunct_list.append(masked_adjunct)
        all_masked_adjunct.append(masked_adjunct_list)
        all_masked_word.append(masked_word_list)
    return all_masked_word, all_masked_adjunct


def gen_masked_sent(j, temp, masked_adjuncts):
    pred_list = []
    new_temp = []
    slot = ["t" + str(j)] * temp.count("t" + str(j))
    for i in range(len(masked_adjuncts)):
        new_sent = temp.replace(" ".join(slot), masked_adjuncts[i])
        result = set(sbar_pattern.findall(new_sent))
        sent_word = new_sent.split(" ")
        if len(result) != 0:
            new_temp.append(new_sent)
            for r in result:
                rep_slot = [r] * sent_word.count(r)
                new_sent = new_sent.replace(" ".join(rep_slot), "")
                new_sent = format_sent(new_sent)
            pred_list.append(new_sent)
        else:
            new_temp.append(new_sent)
            pred_list.append(new_sent)
    return pred_list, new_temp

def format_abbr(sent):
    abbr = ["n't", "'s", "'re", "'ll", "'m"]
    words = sent.split(" ")
    for w in words:
        if w in abbr:
            idx = sent.find(w)
            sent = sent[:idx - 1] + sent[idx:]
    return sent


def pred_sent_by_bert(step_list, masked_temp, words, round, pre_score):
    tests_set = set()
    new_temps = set()
    score_list = []
    print("上一次的分数: ",pre_score)
    for i in range(len(step_list)):
        mask_sent = step_list[i]
        word = words[i]
        print(mask_sent)
        if "[MASK]" in mask_sent:
            # print(mask_sent)
            pred_res = unmasker(mask_sent)
            for r in pred_res:
                if (r['score'] > BERT_SCORE) & ("##" not in r['token_str']) & ("_" not in r['token_str']) & ("," not in r['token_str']):
                    print("token_str: "+r['token_str']+"    bert_score: "+str(r['score']))
                    token_str = r['token_str']
                    new_sent = mask_sent.replace("[MASK]", token_str)
                    new_temp = masked_temp[i].replace("[MASK]", token_str)
                    new_sent = format_abbr(new_sent)
                    new_temp = format_abbr(new_temp)
                    if new_sent not in tests_set:
                        score_list.append(r['score']*pre_score)
                    tests_set.add(new_sent)
                    new_temps.add(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_sent = format_abbr(new_sent)
            new_temp = format_abbr(new_temp)
            if new_sent not in tests_set:
                score_list.append(0.5*pre_score)
            new_temps.add(new_temp)
            tests_set.add(new_sent)
        else:
            new_sent = format_abbr(mask_sent)
            tests_set.add(new_sent)
            new_temps.add(format_abbr(masked_temp[i]))
            score_list.append(0.5*pre_score)
    print("处理后分数： ",score_list)
    return tests_set, new_temps, score_list


def search_syn(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return set(synonyms)


# def pred_sent_by_syn(pred_list, masked_temp, words):
#     tests_set = set()
#     new_temps = set()
#     for i in range(len(pred_list)):
#         mask_sent = pred_list[i]
#         word = words[i]
#         if "[MASK]" in mask_sent:
#             print(mask_sent)
#             syns = search_syn(word)
#             for s in syns:
#                 new_sent = mask_sent.replace("[MASK]", s)
#                 new_temp = masked_temp[i].replace("[MASK]", s)
#                 new_sent = format_abbr(new_sent)
#                 new_temp = format_abbr(new_temp)
#                 tests_set.add(new_sent)
#                 new_temps.add(new_temp)
#             new_sent = mask_sent.replace("[MASK]", word)
#             new_temp = masked_temp[i].replace("[MASK]", word)
#             new_sent = format_abbr(new_sent)
#             new_temp = format_abbr(new_temp)
#             new_temps.add(new_temp)
#             tests_set.add(new_sent)
#         else:
#             new_sent = format_abbr(mask_sent)
#             tests_set.add(new_sent)
#             new_temps.add(format_abbr(masked_temp[i]))
#     return tests_set, new_temps


# def gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
#     w = open(file_path, mode="a")
#     all_tests = []
#     for i in range(0, 10):
#         w.write("sent_id = " + str(i) + "\n")
#         comp = comp_list[i]
#         w.write(format_abbr(comp) + "\n")
#         temp = temp_list[i]
#         tests_list = []
#         tests_list.append([format_abbr(comp)])
#         masked_adjunct_list = all_masked_adjunct[i]
#         masked_word_list = all_masked_word[i]
#         next_temp_list = []
#         for j in range(len(masked_adjunct_list)):
#             w.write("insert t" + str(j) + "\n")
#             if j == 0:
#                 pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
#                 words = masked_word_list[j]
#                 new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
#                 # new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
#                 next_temp_list.extend(new_temps)
#                 for test in new_tests:
#                     w.write(test + "\n")
#                 w.write("\n")
#                 tests_list.append(new_tests)
#             else:
#                 new_temp_list = []
#                 tests_list.append([])
#                 for t in range(len(next_temp_list)):
#                     pred_list, masked_temp = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
#                     words = masked_word_list[j]
#                     new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
#                     # new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
#                     new_temp_list.extend(new_temps)
#                     tests_list[-1].extend(new_tests)
#                     for test in new_tests:
#                         w.write(test + "\n")
#                     w.write("\n")
#                 next_temp_list = new_temp_list
#         w.write("FIN\n")
#         all_tests.append(tests_list)
#     w.close()
#     return all_tests


def gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open(file_path, mode="w", encoding="utf-8")
    all_tests = []
    final_result = []
    sent_result = []
    num_avg,sum_bert,avg_index = 0,0,0
    for i in range(len(comp_list)):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        w.write(format_abbr(comp) + "\n")
        temp = temp_list[i]
        tests_list = []
        tests_list.append([format_abbr(comp)])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        next_temp_list = []
        new_score_list = []
        old_score_list = []
        for j in range(len(masked_adjunct_list)):
            w.write("insert t" + str(j) + "\n")
            if j == 0:
                pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                # score_list t0阶段的分数
                new_tests, new_temps, score_list = pred_sent_by_bert(pred_list, masked_temp, words, j, 1)
                print("t0: ", score_list)
                old_score_list = score_list
                next_temp_list.extend(new_temps)
                # sum_bert, avg_index = calculate_avg(score_list, sum_bert, avg_index)
                score_temp = 0
                for test in new_tests:
                    w.write(test + " " + str(score_list[score_temp]) + "\n")
                    score_temp += 1
                tests_list.append(new_tests)
            else:
                new_temp_list = []
                new_temp_list_all = []
                tests_list_all = []
                tests_list.append([])
                for t in range(len(next_temp_list)):
                    pred_list, masked_temp = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
                    words = masked_word_list[j]
                    print("next_temp_list: "+next_temp_list[t])
                    # tj阶段
                    new_tests, new_temps, score_list = pred_sent_by_bert(pred_list, masked_temp, words, j, old_score_list[t])
                    # sum_bert,avg_index = calculate_avg(score_list, sum_bert, avg_index)
                    new_score_list.extend(score_list)
                    new_temp_list_all.extend(new_temps)
                    tests_list_all.extend(new_tests)
                    # new_temp_list.extend(new_temps)
                    # tests_list[-1].extend(new_tests)
                    # score_temp = 0
                    # for test in new_tests:
                    #     w.write(test + " " + str(score_list[score_temp]) + "\n")
                    #     score_temp += 1
                    # w.write("\n")
                # old_score_list = new_score_list
                print("t" + str(j) + "的第" + str(t) + "轮: ", new_score_list)
                score_tests_dict = dict(zip(tests_list_all, new_score_list))
                tests_temp_dict = dict(zip(tests_list_all, new_temp_list_all))
                score_tests_dict = sorted(score_tests_dict.items(), key=lambda d: d[1], reverse=True)
                print("排序后结果：", score_tests_dict)
                next_test_list = []
                old_score_list = []
                next_temp_list = []
                # 缩小指数，原为5
                for dic_item in score_tests_dict[0:min(len(score_tests_dict), 4)]:
                    next_test_list.append(dic_item[0])
                    old_score_list.append(dic_item[1])
                    next_temp_list.append(tests_temp_dict[dic_item[0]])
                    w.write(dic_item[0] + " " + str(dic_item[1]) + "\n")
                tests_list[-1].extend(next_test_list)
                new_score_list = []
            w.write("\n")
            if j == len(masked_adjunct_list) - 1:
                if j == 0:
                    sent_result = new_tests
                else:
                    sent_result = next_test_list
        w.write("FIN\n")
        w.write("\n")
        all_tests.append(tests_list)
        final_result.append(sent_result)
    w.close()
    # print("总计生成"+str(avg_index)+"个有效句子,总分为"+str(sum_bert))
    # print("平均分为"+str(sum_bert/avg_index))
    return all_tests, final_result

def calculate_avg(list,sum,index):
    for i in list:
        if str(i) != 'orig' and str(i) != 'word':
            index += 1
            sum += i
    return sum,index

def calculate_context_num(file_path):
    orig_sents = open(file_path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    sent_list = []
    index = 0
    while sent:
        sent = sent[:-1]
        if "context_id" in sent:
            if index != 0:
                sent_list.append(index)
            index = 0
        else:
            if sent != "":
                print("句子：", sent)
                index += 1
        sent = orig_sents.readline()
    if index != 0:
        sent_list.append(index)
    return sent_list

def load_unchange_sent():
    file_name = "ans_context_simple.txt"
    path = "./Squad2/"+file_name
    orig_sents = open(path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    result = []
    sent_list = []
    index = 0
    while sent:
        sent = sent[:-1]
        if "context_id =" in sent:
            sent_list.append(sent.split(" ")[-1])
            sent = orig_sents.readline()
            sent = orig_sents.readline()
            sent = orig_sents.readline()
            sent_list.append(sent.split(" ")[-1].strip('\n').strip())
            sent = orig_sents.readline()
            sent_list.append(sent[9:-1].strip("\n"))
            result.append(sent_list)
            sent_list = []
        sent = orig_sents.readline()
    return result

def read_context_first():
    file_name = "context1.txt"
    path = "./Squad2/"+file_name
    orig_sents = open(path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    result = []
    for i in range(4):
        sent = orig_sents.readline()
        if sent!="":
            result.append(sent.strip('\n').strip())
    print(len(result))
    return result

def create_id():
    m = hashlib.md5()
    m.update(bytes(str(time.perf_counter()),encoding='utf-8'))
    return m.hexdigest()

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

def mapping_context_sentence(list, result):
    dic = {}
    num = 0
    plus = 0
    for i in list:
        dic["context"+str(num)] = result[plus:plus+i]
        plus += i
        num += 1
    return dic

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

# 调用百度翻译api
def back_translation(sent):
    # host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=Tl22jvCjsihL3CPLQeTzKqUL&client_secret=U4QW2LeDioD8arLDM75RTLpesdWn1qrj'
    # response = requests.get(host)
    # if response:
    #     print(response.json())

    token = "24.12fdf025f1336049538b7d198c440f5b.2592000.1647431058.282335-25606143"
    url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token
    # For list of language codes, please refer to `https://ai.baidu.com/ai-doc/MT/4kqryjku9#语种列表`
    from_lang = 'en'  # example: en
    to_lang = 'zh'  # example: zh
    term_ids = ''  # 术语库id，多个逗号隔开
    # Build request
    headers = {'Content-Type': 'application/json'}
    payload = {'q': sent, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)

    result = r.json()
    zh_trans_result = result['result']['trans_result'][0]['dst']

    from_lang = 'zh'  # example: en
    to_lang = 'en'  # example: zh
    term_ids = ''  # 术语库id，多个逗号隔开
    # Build request
    headers = {'Content-Type': 'application/json'}
    payload = {'q': zh_trans_result, 'from': from_lang, 'to': to_lang, 'termIds': term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    en_trans_result = result['result']['trans_result'][0]['dst']

    return en_trans_result

# 通过回译（英文-》中文-》英文）生成新context，输入是ans_context.txt 读取之后的list，输出为ques为key，新context为value的键值对（1对1）构成的list
def gen_ans_context_by_trans(ans_context):
    new_context_list = []
    sid = 3500
    eid = 4000
    for i in range(sid, eid):
        print('i: ', i)
        ques_ans = ans_context[i]
        ques = ques_ans["question"]
        new_context = {}
        if ques_ans["answer"] == "None":
            continue
        context = ques_ans["context"]
        new_context["ques"] = ques
        new_context["con"] = [back_translation(context)]
        new_context["idx"] = str(i)
        new_context_list.append(new_context)
    return new_context_list


# 通过同义词生成新context，输入是ans_context.txt 读取之后的list，输出为ques为key，新context为value的键值对（1对多）构成的list
def gen_ans_context_by_syn(ans_context):
    stop_words = set(stopwords.words("english"))
    new_context_list = []
    for i in range(len(ans_context)):
        print('i: ', i)
        ques_ans = ans_context[i]
        ques = ques_ans["question"]
        new_context = {}
        new_context["ques"] = ques
        new_context["con"] = []
        new_context["idx"] = str(i)
        filter_words = word_extraction(ques)
        if ques_ans["answer"] == "None":
            continue
        ans_list = ques_ans["answer"].split("|")
        for ans in ans_list:
            ans_word = word_extraction(ans)
            filter_words.extend(ans_word)
        filter_words = set(filter_words)
        context = ques_ans["context"]
        con_words = context.strip().rstrip().split()
        # print(con_words)
        con_word_pos = nltk.pos_tag(con_words)
        #print(con_word_pos)
        for j in range(len(con_word_pos)):
            if con_word_pos[j][1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                if (con_words[j] not in filter_words) & (con_words[j] not in stop_words):
                    print(con_words[j], " can replace")
                    syns = get_synonyms(con_words[j])
                    syns = list(filter(lambda s: s and s.strip(), syns))
                    if len(syns) != 0:
                        print(syns)
                        for syn in syns:
                            if (syn != con_words[j].lower()) & (len(syn.split(" ")) == 1):
                                syn_pos = nltk.pos_tag([syn])
                                if syn_pos[0][1] == con_word_pos[j][1]:
                                    new_words = list(con_words)
                                    new_words[j] = syn
                                    print("new sentence: ", " ".join(new_words))
                                    new_context["con"].append(" ".join(new_words))
        new_context_list.append(new_context)
    return new_context_list

# 读取ans_context.txt
def load_context_ans(file_path):
    ans_context = []
    f = open(file_path, "r")
    line = f.readline()
    ques_ans = {}
    while line:
        if ("question:" in line) & (len(ques_ans) == 0):
            ques_ans["question"] = line[:-1].replace("question: ","")
        if ("answer:" in line) & (len(ques_ans) == 1):
            ques_ans["answer"] = line[:-1].replace("answer: ", "")
        if ("context:" in line) & (len(ques_ans) == 2):
            context_sent = line[:-1].replace("context: ", "")
            context_sent = format_context_sent(context_sent)
            ques_ans["context"] = context_sent
            ans_context.append(ques_ans)
            ques_ans = {}
        line = f.readline()

    return ans_context

# 保存新context
def save_new_context(file_path, new_context_list):
    f = open(file_path, 'a')
    for i in range(len(new_context_list)):
        new_context = new_context_list[i]
        f.write("ans_context_idx: " + new_context["idx"] + "\n")
        f.write("ques: " + new_context["ques"] + "\n")
        f.write("new_context:" + "\n")
        for con in new_context["con"]:
            f.write(con + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    # 获取所有句子的<=beam_size个变体
    file_name = "context"
    temp_list, adjunct_list, comp_list, ner_list = gen_sent_temp_main(file_name)
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    all_masked_word, all_masked_adjunct = gen_mask_phrase(adjunct_list, pos_list, ner_list)
    file_path = "./" + file_name + "_bert_test.txt"
    all_tests, final_result = gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct)
    print(final_result)

    # 获取context里句子个数的列表
    context_sentence_len_list, origin_sent_list = get_sentence_len(file_name)
    final_result_dic = mapping_context_sentence(context_sentence_len_list, final_result)
    print(final_result_dic)

    index = 0

    # 获取每个context对应的问题
    question_sent_li_all = read_question_index()

    # 读处理后的源文件
    file_name = "dev_start.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)
    prediction_data = prediction_json["data"]
    item_list = []

    # 循环处理context
    for i in range(len(context_sentence_len_list[0:50])):
        # w.write("context_id = " + str(i) + "\n")
        context_input = final_result_dic["context"+str(i)]
        if context_sentence_len_list[i] == 1:
            pro_context_list = origin_sent_list[i]
        else:
            if context_sentence_len_list[i] > 4:
                ll_len = context_sentence_len_list[i]
                ll_temp_list = [x for x in range(0, ll_len)]
                random.shuffle(ll_temp_list)
                for s in ll_temp_list[4:ll_len]:
                    context_input[s] = [origin_sent_list[i][s]]
            # 对context_input中，最多4个句子改变，其他len-4个不变，随机取
            pro_context_list = itertools.product(*context_input)
        # print(i, context_input)
        # continue
        print("========")
        final_context_list = []
        for pro_context in pro_context_list:
            final_context_list.append(list(pro_context))
            # pro_context_item = "".join(pro_context)
            # test_set.add(pro_context_item)

        # if i == 0:
        # 读对应的那一条context的问题
        item_temp = copy.deepcopy(prediction_data[0]["paragraphs"][i])
        question_list = []
        for jj in item_temp["qas"]:
            question_list.append(jj["question"])
        print(question_list)
        # exit(0)
        test_set = []
        index = 0
        question_sent_li = question_sent_li_all[i]

        for pro_context_index in range(len(final_context_list)):

            context_for_input_li = final_context_list[pro_context_index]
            for question_sent_index in range(len(question_sent_li)):
                index_i = int(question_sent_li[question_sent_index])
                if index_i != -1:
                    context_for_input_li[index_i] = origin_sent_list[i][index_i]
                question_temp = item_temp["qas"][question_sent_index]
                question_temp["id"] = create_id()
                item_to_add = []
                item_to_add.append(question_temp)
                item_temp_modify = copy.deepcopy(item_temp)
                item_temp_modify["qas"] = item_to_add
                context_all_line = "".join(context_for_input_li)
                item_temp_modify["context"] = context_all_line
                valid_temp = context_all_line+str(question_list[question_sent_index])
                if valid_temp in test_set:
                    continue
                else:
                    test_set.append(valid_temp)
                    item_list.append(copy.deepcopy(item_temp_modify))
                    index += 1
            print(index)
    # exit(0)
    prediction_data[0]["paragraphs"] = item_list
    file_name = "dev_modify.json"
    path = "./" + file_name
    with open(path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(prediction_json, indent=4, ensure_ascii=False))
    # print(len(test_set))
    print(index)
    # w.close()

