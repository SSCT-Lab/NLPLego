import nltk
from nltk.corpus import wordnet
from gen_temp import *
import re
import itertools
from nltk import CoreNLPParser
from nltk.corpus import stopwords
import spacy
import transformers

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


def pred_sent_by_syn(pred_list, masked_temp, words):
    tests_set = set()
    new_temps = set()
    for i in range(len(pred_list)):
        mask_sent = pred_list[i]
        word = words[i]
        if "[MASK]" in mask_sent:
            print(mask_sent)
            syns = search_syn(word)
            for s in syns:
                new_sent = mask_sent.replace("[MASK]", s)
                new_temp = masked_temp[i].replace("[MASK]", s)
                new_sent = format_abbr(new_sent)
                new_temp = format_abbr(new_temp)
                tests_set.add(new_sent)
                new_temps.add(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_sent = format_abbr(new_sent)
            new_temp = format_abbr(new_temp)
            new_temps.add(new_temp)
            tests_set.add(new_sent)
        else:
            new_sent = format_abbr(mask_sent)
            tests_set.add(new_sent)
            new_temps.add(format_abbr(masked_temp[i]))
    return tests_set, new_temps


def gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open(file_path, mode="a")
    all_tests = []
    for i in range(0, 10):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        w.write(format_abbr(comp) + "\n")
        temp = temp_list[i]
        tests_list = []
        tests_list.append([format_abbr(comp)])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        next_temp_list = []
        for j in range(len(masked_adjunct_list)):
            w.write("insert t" + str(j) + "\n")
            if j == 0:
                pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
                # new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
                next_temp_list.extend(new_temps)
                for test in new_tests:
                    w.write(test + "\n")
                w.write("\n")
                tests_list.append(new_tests)
            else:
                new_temp_list = []
                tests_list.append([])
                for t in range(len(next_temp_list)):
                    pred_list, masked_temp = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
                    words = masked_word_list[j]
                    new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
                    # new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
                    new_temp_list.extend(new_temps)
                    tests_list[-1].extend(new_tests)
                    for test in new_tests:
                        w.write(test + "\n")
                    w.write("\n")
                next_temp_list = new_temp_list
        w.write("FIN\n")
        all_tests.append(tests_list)
    w.close()
    return all_tests


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
                for dic_item in score_tests_dict[0:min(len(score_tests_dict), 5)]:
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


if __name__ == '__main__':
    file_name = "context1"
    temp_list, adjunct_list, comp_list, ner_list = gen_sent_temp_main(file_name)
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    all_masked_word, all_masked_adjunct = gen_mask_phrase(adjunct_list, pos_list, ner_list)
    file_path = "./" + file_name + "_bert_test.txt"
    all_tests, final_result = gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct)
    print(final_result)
    # pro_context = itertools.product(*final_result)
    # for i in pro_context:
    #     print("句子：",i)
    context_list = calculate_context_num("./Squad2/"+file_name+".txt")
    print("context_list",context_list)
    index = 0
    w = open("./" + file_name + "_bert_context.txt", mode="w", encoding="utf-8")
    # for i in range(len(context_list)):
    #     w.write("context_id = " + str(i) + "\n")
    #     for j in range(index, index+context_list[i]):
    #         w.write("sent_id = " + str(j-index) + "\n")
    #         for k in final_result[j]:
    #             w.write(k + "\n")
    #         w.write("\n")
    #     w.write("FIN\n")
    #     w.write("\n")
    #     index += context_list[i]
    for i in range(len(context_list)):
        w.write("context_id = " + str(i) + "\n")
        context_input = final_result[index:index+context_list[i]]
        mul = 1
        for temp in context_input:
            mul *= len(temp)
        w.write("此context有"+str(context_list[i])+"个句子,共生成" + str(mul) + "个新的context\n")
        if context_list[i] == 1:
            pro_context_list = final_result[index]
        else:
            pro_context_list = itertools.product(*context_input)
        pro_context_id = 0
        for pro_context in pro_context_list:
            pro_context_item = "".join(pro_context)
            w.write("pro_context_id = " + str(pro_context_id) + "\n")
            w.write(pro_context_item + "\n")
            pro_context_id += 1
            w.write("\n")
        w.write("FIN\n")
        w.write("\n")
        index += context_list[i]
    w.close()

