import copy
import json
import os
import random
import itertools
from eng_inflection.get_plural import *
from eng_inflection.get_comparative import *
from eng_inflection.get_conjugation import *
from gen_temp import *
import re
import time, hashlib
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import transformers
import requests
import csv
import pandas as pd

nlp = spacy.load("en_core_web_lg")
sbar_pattern = re.compile(r't\d+')
stops = set(stopwords.words("english"))
unmasker = transformers.pipeline('fill-mask', model='bert-base-uncased')
wnl = WordNetLemmatizer()
BERT_SCORE = 0.1


def format_mask_adjunct(mask_adjunct, adjunct):
    if len(mask_adjunct.split(" ")) != len(adjunct.split(" ")):
        mask_adjunct = mask_adjunct.replace(" - ", "-").replace(" – ", "–").replace(" − ", "−").rstrip().strip()
        if ("( " in mask_adjunct) & ("( " not in adjunct):
            mask_adjunct = mask_adjunct.replace("( ", "(")
        if (" )" in mask_adjunct) & (" )" not in adjunct):
            mask_adjunct = mask_adjunct.replace(" )", ")")
        if (" / " in mask_adjunct) & (" / " not in adjunct):
            mask_adjunct = mask_adjunct.replace(" / ", "/")
        if (" +" in mask_adjunct) & (" +" not in adjunct):
            mask_adjunct = mask_adjunct.replace(" +", "+")
        if (" m " in mask_adjunct) & (" m " not in adjunct):
            mask_adjunct = mask_adjunct.replace(" m ", "m ")
        if (" & " in mask_adjunct) & (" & " not in adjunct):
            mask_adjunct = mask_adjunct.replace(" & ", "&")
        if (" °" in mask_adjunct) & (" °" not in adjunct):
            mask_adjunct = mask_adjunct.replace(" °", "°")
        if ("° C" in mask_adjunct) & ("° C" not in adjunct):
            mask_adjunct = mask_adjunct.replace("° C", "°C")
        if ("etc ." in mask_adjunct) & ("etc ." not in adjunct):
            mask_adjunct = mask_adjunct.replace("etc .", "etc.")
        if ("£ " in mask_adjunct) & ("£ " not in adjunct):
            mask_adjunct = mask_adjunct.replace("£ ", "£")
        if (" al . " in mask_adjunct) & (" al . " not in adjunct):
            mask_adjunct = mask_adjunct.replace(" al . ", " al. ")
        if ("ca . " in mask_adjunct) & ("ca . " not in adjunct):
            mask_adjunct = mask_adjunct.replace("ca . ", "ca. ")
    return mask_adjunct


def get_cannot_rep_words(ner_list, hyp_words, for_list):
    unrep_words = []
    for w in hyp_words:
        unrep_words.extend(re.split("-|–|−", w[1]))
    for f in for_list:
        unrep_words.extend(f.split(" "))
    for n in ner_list:
        unrep_words.extend(n.split(" "))

    return unrep_words


def filer_word(pos_list, adjunct, unrep_words, for_list):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '–', '—',
                            '--', '--', '-']
    doc = nlp(adjunct)
    word_pos = [tok.pos_ for tok in doc]
    masked_word = []
    masked_adjunct = []
    masked_word_pos = []
    adjunct_word = [tok.text for tok in doc]
    for i in range(len(adjunct_word)):
        word = adjunct_word[i]
        if (word not in stops) & (word not in english_punctuations) & (word_pos[i] in pos_list) & (
                word not in unrep_words) & (len([f for f in for_list if word in f]) == 0):
            temp_phrase = list(adjunct_word)
            temp_phrase[i] = "[MASK]"
            mask_phrase = format_mask_adjunct(" ".join(temp_phrase), adjunct)
            if i != 0:
                mask_index = mask_phrase.index("[MASK]")
                if mask_phrase[mask_index - 1] != " ":
                    continue
            masked_word.append(word)
            masked_word_pos.append(word_pos[i])
            masked_adjunct.append(mask_phrase)

    if len(masked_adjunct) == 0:
        masked_adjunct.append(adjunct)
        masked_word.append("X")
        masked_word_pos.append("X")

    return masked_word, masked_adjunct, masked_word_pos


def gen_mask_phrase(adjunct_list, pos_list, all_ner, all_for, all_hyp_words):
    all_masked_adjunct = []
    all_masked_word = []
    all_masked_word_pos = []
    for i in range(len(adjunct_list)):
        adjuncts = adjunct_list[i]
        ner_list = all_ner[i]
        for_list = all_for[i]
        hyp_words_list = all_hyp_words[i]
        unrep_words = get_cannot_rep_words(ner_list, hyp_words_list, for_list)
        masked_adjunct_list = []
        masked_word_list = []
        masked_pos_list = []
        for adjunct in adjuncts:
            masked_word, masked_adjunct, masked_word_pos = filer_word(pos_list, adjunct, unrep_words, for_list)
            masked_word_list.append(masked_word)
            masked_adjunct_list.append(masked_adjunct)
            masked_pos_list.append(masked_word_pos)
        all_masked_adjunct.append(masked_adjunct_list)
        all_masked_word.append(masked_word_list)
        all_masked_word_pos.append(masked_pos_list)
    return all_masked_word, all_masked_adjunct, all_masked_word_pos


def gen_masked_sent(j, temp, masked_adjuncts):
    pred_list = []
    new_temp = []
    temp_item = "t" + str(j)
    temp_index_list = temp.split(" ")
    left_index = temp_index_list.index(temp_item)
    right_index = len(temp_index_list) - 1 - temp_index_list[::-1].index(temp_item)
    if left_index == -1:
        # 占位符
        slot = "###"
    else:
        slot = " ".join(temp_index_list[left_index:right_index + 1])
    for i in range(len(masked_adjuncts)):
        new_sent = temp
        if not slot == "###":
            if len(slot.split(" ")) == len(masked_adjuncts[i].split(" ")):
                if len(slot.split(" ")) == 1:
                    masked_adjuncts_temp = temp.split(" ")
                    for temp_masked_index in range(len(masked_adjuncts_temp)):
                        if masked_adjuncts_temp[temp_masked_index] == temp_item:
                            masked_adjuncts_temp[temp_masked_index] = masked_adjuncts[i]
                    new_sent = " ".join(masked_adjuncts_temp)
                else:
                    new_sent = temp.replace(slot, masked_adjuncts[i])
        # result = set(sbar_pattern.findall(new_sent))
        sent_word = new_sent.split(" ")
        result = set()
        for w in sent_word:
            find_res = sbar_pattern.findall(w)
            if len(find_res) != 0:
                if len(find_res[0]) == len(w):
                    result.add(find_res[0])
        if len(result) != 0:
            result = sorted(list(result), reverse=True)
            new_temp.append(new_sent)
            for r in result:
                left_index_r = sent_word.index(r)
                right_index_r = len(sent_word) - 1 - sent_word[::-1].index(r)
                rep_slot = " ".join(sent_word[left_index_r:right_index_r + 1])
                new_sent = new_sent.replace(rep_slot, "")
            pred_list.append(format_punct(new_sent))
        else:
            new_temp.append(new_sent)
            pred_list.append(new_sent)

    return pred_list, new_temp


def format_abbr(sent):
    abbr = ["n't", "'s", "'re", "'ll", "'m"]
    words = sent.split()
    sent = " ".join(words)
    for w in words:
        if w in abbr:
            idx = sent.find(w)
            sent = sent[:idx - 1] + sent[idx:]
    return sent


def format_punct(test):
    test_words = test.split()
    del_idx = []
    for i in range(len(test_words) - 1):
        if (test_words[i] in eng_punctuation) & (test_words[i + 1] in eng_punctuation):
            del_idx.append(i)

    for i in reversed(del_idx):
        del test_words[i]
    if (test_words[0] != "[MASK]") & (test_words[0].islower()):
        test_words[0] = test_words[0].title()
    test = " ".join(test_words)
    return test


def pred_sent_by_bert(step_list, masked_temp, masked_adjuncts, words):
    tests_set = []
    new_temps = []
    new_adjuncts = []
    for i in range(len(step_list)):
        mask_sent = step_list[i]
        word = words[i]
        print(mask_sent)
        if "[MASK]" in mask_sent:
            pred_res = unmasker(mask_sent)
            for r in pred_res:
                if (r['score'] > BERT_SCORE) & ("##" not in r['token_str']) & ("_" not in r['token_str']) & (
                        "," not in r['token_str']) & (r['token_str'] != ""):
                    token_str = r['token_str']
                    new_sent = mask_sent.replace("[MASK]", token_str)
                    new_adjunct = masked_adjuncts[i].replace("[MASK]", token_str)
                    new_temp = masked_temp[i].replace("[MASK]", token_str)
                    if new_sent not in tests_set:
                        tests_set.append(format_punct(format_abbr(new_sent)))
                        new_temps.append(new_temp)
                        new_adjuncts.append(new_adjunct)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_adjunct = masked_adjuncts[i].replace("[MASK]", word)
            if new_sent not in tests_set:
                new_temps.append(new_temp)
                tests_set.append(format_punct(format_abbr(new_sent)))
                new_adjuncts.append(new_adjunct)
        else:
            new_sent = format_punct(format_abbr(mask_sent))
            tests_set.append(new_sent)
            new_temps.append(masked_temp[i])
            new_adjuncts.append(masked_adjuncts[i])

    return tests_set, new_temps, new_adjuncts


def pred_sent_by_bert_score(step_list, masked_temp, words, round, pre_score):
    tests_set = set()
    new_temps = set()
    score_list = []
    print("上一次的分数: ", pre_score)
    for i in range(len(step_list)):
        mask_sent = step_list[i]
        word = words[i]
        print(mask_sent)
        if "[MASK]" in mask_sent:
            pred_res = unmasker(mask_sent)
            for r in pred_res:
                if (r['score'] > BERT_SCORE) & ("##" not in r['token_str']) & ("_" not in r['token_str']) & (
                        "," not in r['token_str']):
                    print("token_str: " + r['token_str'] + "    bert_score: " + str(r['score']))
                    token_str = r['token_str']
                    new_sent = mask_sent.replace("[MASK]", token_str)
                    new_temp = masked_temp[i].replace("[MASK]", token_str)
                    new_temp = new_temp[0].upper() + new_temp[1:]
                    new_sent = format_punct(format_abbr(new_sent))
                    if new_sent not in tests_set:
                        score_list.append(r['score'] * pre_score)
                    tests_set.add(new_sent)
                    new_temps.add(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_temp = new_temp[0].upper() + new_temp[1:]
            new_sent = format_punct(format_abbr(new_sent))
            if new_sent not in tests_set:
                score_list.append(0.5 * pre_score)
            new_temps.add(new_temp)
            tests_set.add(new_sent)
        else:
            new_sent = format_punct(format_abbr(mask_sent))
            tests_set.add(new_sent)
            new_temps.add(masked_temp[i])
            score_list.append(0.5 * pre_score)
    print("处理后分数： ", score_list)
    return tests_set, new_temps, score_list


def search_tense(word, infinitive, conju, new_infin, new_conju):
    if (word == infinitive) | (len(new_conju) < 4):
        return new_infin
    elif word in conju[0]:
        return new_conju[0].replace("TS:", "")
    elif word in conju[1]:
        return new_conju[1].replace("PC:", "")
    elif word in conju[2]:
        return new_conju[2].replace("PA:", "")
    else:
        return new_conju[3].replace("PP:", "")


def search_syn(word, pos):
    synonyms = set()
    if pos == "NOUN":
        # root = wnl.lemmatize(word, pos=wordnet.NOUN)
        singular_root, plural_root = get_plural(word)
        for syn in wordnet.synsets(singular_root, pos=wordnet.NOUN):
            for lm in syn.lemmas():
                singular_new, plural_new = get_plural(lm.name())
                if singular_new != singular_root:
                    if word == singular_root:
                        synonyms.add(singular_new)
                    else:
                        if len(plural_new) != 0:
                            synonyms.add(plural_new[0].replace("PL:", ""))
                        else:
                            synonyms.add(singular_new)
    if pos == "VERB":
        # root = wnl.lemmatize(word, pos=wordnet.VERB)
        infinitive, conju = get_conjugation(word)
        for syn in wordnet.synsets(infinitive, pos=wordnet.VERB):
            for lm in syn.lemmas():
                new_inf, new_conju = get_conjugation(lm.name())
                if new_inf != infinitive:
                    synonyms.add(search_tense(word, infinitive, conju, new_inf, new_conju))

    if pos == "ADJ":
        # root = wnl.lemmatize(word, pos=wordnet.ADJ)
        orig, comp = get_comparative(word)
        for syn in wordnet.synsets(orig, pos=wordnet.ADJ):
            for lm in syn.lemmas():
                new_orig, new_comp = get_comparative(lm.name())
                if orig != new_orig:
                    if len(new_comp) != 0:
                        if word == orig:
                            synonyms.add(new_orig)
                        elif word in comp[0]:
                            synonyms.add(new_comp[0].replace('CO:', ''))
                        else:
                            synonyms.add(new_comp[1].replace('SU:', ''))
                    else:
                        synonyms.add(lm.name())

    if pos == "ADV":
        for syn in wordnet.synsets(word, pos=wordnet.ADV):
            for lm in syn.lemmas():
                if word != lm.name():
                    synonyms.add(lm.name())

    return synonyms


def extra_insert_phrases(new_sents, masked_temp, masked_adjunct):
    inserted_adjuncts = []
    s_idx = check_continuity(masked_adjunct.split(" "), masked_temp.split(" "), -1)
    e_idx = s_idx + len(masked_adjunct.split(" "))
    for sent in new_sents:
        inserted_adjunct = " ".join(sent.split(" ")[s_idx:e_idx])
        inserted_adjuncts.append(inserted_adjunct)
    return inserted_adjuncts


def gen_tests_for_sst(comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open("./sst_test_map.txt", "w")
    sst_tests = []
    sst_adjuncts = []
    for i in range(len(comp_list)):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        temp = temp_list[i]
        tests_list = []
        adjunct_list = []
        next_temp_list = []
        tests_list.append([format_abbr(comp)])
        adjunct_list.append([])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        for j in range(len(masked_adjunct_list)):
            # w.write("insert t" + str(j) + "\n")
            if j == 0:
                last_sent = tests_list[j][0]
                pred_list, masked_temps = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                new_tests, new_temps, new_adjuncts = pred_sent_by_bert(pred_list, masked_temps, masked_adjunct_list[j],
                                                                       words)
                next_temp_list.extend(new_temps)
                tests_list.append(list(new_tests))
                adjunct_list.append(list(new_adjuncts))
                for p in range(len(new_tests)):
                    w.write(last_sent + " | " + new_adjuncts[p] + " | " + new_tests[p] + "\n")
                w.write("\n")
            else:
                tests_list.append([])
                adjunct_list.append([])
                new_temp_list = []
                for t in range(len(next_temp_list)):
                    last_sent = tests_list[j][t]
                    pred_list, masked_temps = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
                    words = masked_word_list[j]
                    # tj阶段
                    new_tests, new_temps, new_adjuncts = pred_sent_by_bert(pred_list, masked_temps,
                                                                           masked_adjunct_list[j], words)
                    tests_list[-1].extend(new_tests)
                    new_temp_list.extend(new_temps)
                    adjunct_list[-1].extend(new_adjuncts)
                    for p in range(len(new_tests)):
                        w.write(last_sent + " | " + new_adjuncts[p] + " | " + new_tests[p] + "\n")
                w.write("\n")
                next_temp_list = new_temp_list
        w.write("\n")
        sst_tests.append(tests_list)
        sst_adjuncts.append(adjunct_list)
    w.close()
    return sst_tests, sst_adjuncts


def gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open(file_path, mode="w", encoding="utf-8")
    all_tests = []
    final_result = []
    sent_result = []
    num_avg, sum_bert, avg_index = 0, 0, 0
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
                new_tests, new_temps, score_list = pred_sent_by_bert_score(pred_list, masked_temp, words, j, 1)
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
                    print("next_temp_list: " + next_temp_list[t])
                    # tj阶段
                    new_tests, new_temps, score_list = pred_sent_by_bert_score(pred_list, masked_temp, words, j,
                                                                               old_score_list[t])
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
        if len(sent_result) == 0:
            sent_result.append(comp)
        final_result.append(sent_result)
        sent_result = []
    w.close()
    # print("总计生成"+str(avg_index)+"个有效句子,总分为"+str(sum_bert))
    # print("平均分为"+str(sum_bert/avg_index))
    return all_tests, final_result


def calculate_avg(list, sum, index):
    for i in list:
        if str(i) != 'orig' and str(i) != 'word':
            index += 1
            sum += i
    return sum, index


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
    path = "./Squad2/" + file_name
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
    file_name = "context.txt"
    path = "./txt_files/" + file_name
    orig_sents = open(path, mode="r", encoding='utf-8')
    sent = orig_sents.readline()
    result = []
    for i in range(4):
        sent = orig_sents.readline()
        if sent != "":
            result.append(sent.strip('\n').strip())
    print(len(result))
    return result


def create_id():
    m = hashlib.md5()
    m.update(bytes(str(time.perf_counter()), encoding='utf-8'))
    return m.hexdigest()


def get_sentence_len(file_name):
    path = "./txt_files/" + file_name + ".txt"
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
    print(sum(result))
    return result, sent_result_all


def mapping_context_sentence(list, result, s_idx, e_idx):
    dic = {}
    # num = 0
    plus = 0
    # for i in list:
    for i in range(s_idx, e_idx):
        sent_count = list[i]
        # dic["context"+str(num + s_idx)] = result[plus:plus+i]
        dic["context" + str(i)] = result[plus:plus + sent_count]
        plus += sent_count
        # num += 1
    return dic


def read_question_index():
    file_name = "ans_context.txt"
    path = "./txt_files/" + file_name
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


def change_to_dic(syns, word):
    res = []
    similar_word = nlp(word)
    if (similar_word and similar_word.vector_norm):
        for i in syns:
            if i != word:
                token = nlp(i)
                if token and token.vector_norm:
                    res.append({"syn_str": token.text, "score": similar_word.similarity(token)})
    res = sorted(res, key=lambda d: d["score"], reverse=True)
    return res[0:min(len(res), 5)]


def pred_sent_by_syn(pred_list, masked_temp, words, pos_list, round, pre_score):
    tests_set = []
    new_temps = []
    score_list = []
    print("上一次的分数: ", pre_score)
    for i in range(len(pred_list)):
        mask_sent = pred_list[i]
        word = words[i]
        pos = pos_list[i]
        if "[MASK]" in mask_sent.split():
            # print(mask_sent)
            syns = search_syn(word, pos)
            syns = change_to_dic(syns, word)
            for r in syns:
                # print("syn_str: " + r['syn_str'] + "    syn_score: " + str(r['score']))
                syn_str = r['syn_str']
                if syn_str == "side":
                    print("side:", mask_sent)
                new_sent = mask_sent.replace("[MASK]", syn_str)
                new_temp = masked_temp[i].replace("[MASK]", syn_str)
                new_sent = format_punct(format_abbr(new_sent))
                if syn_str == "side":
                    print("side:", new_sent)
                if new_sent not in tests_set:
                    score_list.append(r['score'] * pre_score)
                    tests_set.append(new_sent)
                    new_temps.append(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_sent = format_punct(format_abbr(new_sent))
            if new_sent not in tests_set:
                score_list.append(0.5 * pre_score)
                tests_set.append(new_sent)
                new_temps.append(new_temp)
        else:
            new_sent = format_punct(format_abbr(mask_sent))
            if new_sent not in tests_set:
                tests_set.append(new_sent)
                new_temps.append(masked_temp[i])
                score_list.append(0.5 * pre_score)
    print("处理后分数： ", score_list)
    return tests_set, new_temps, score_list


def gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct, all_masked_word_pos):
    w = open(file_path, mode="w", encoding="utf-8")
    all_tests = []
    final_result = []
    sent_result = []
    for i in range(len(comp_list)):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        w.write(format_abbr(comp) + "\n")
        temp = temp_list[i]
        tests_list = []
        tests_list.append([format_abbr(comp)])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        masked_word_pos = all_masked_word_pos[i]
        next_temp_list = []
        new_score_list = []
        old_score_list = []
        for j in range(len(masked_adjunct_list)):
            w.write("insert t" + str(j) + "\n")
            if j == 0:
                pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                pos_list = masked_word_pos[j]
                new_tests, new_temps, score_list = pred_sent_by_syn(pred_list, masked_temp, words, pos_list, j, 1)
                print("t0: ", score_list)
                old_score_list = score_list
                next_temp_list.extend(new_temps)
                score_temp = 0
                if len(masked_adjunct_list) == 1:
                    new_tests = new_tests[0:min(4, len(new_tests))]
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
                    pos_list = masked_word_pos[j]
                    print(t, len(old_score_list))
                    new_tests, new_temps, score_list = pred_sent_by_syn(pred_list, masked_temp, words, pos_list, j,
                                                                        old_score_list[t])
                    new_score_list.extend(score_list)
                    new_temp_list_all.extend(new_temps)
                    tests_list_all.extend(new_tests)

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
        if len(masked_adjunct_list) != 0:
            final_result.append(sent_result)
        else:
            final_result.append(tests_list[0])
    w.close()
    return all_tests, final_result


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


def save_new_tests_for_sst(file_path, all_tests):
    w = open(file_path, "w")
    for i in range(len(all_tests)):
        w.write("sent_id = " + str(i) + "\n")
        for j in range(len(all_tests[i])):
            test_list = all_tests[i][j]
            for test in test_list:
                w.write(test + "\n")
        w.write("\n")
    w.close()


def gen_input_for_senta(out_file, sst_tests, sst_adjuncts):
    qid = 0
    w = open(out_file, "w")
    tsv_writer = csv.writer(w, delimiter='\t')
    tsv_writer.writerow(['qid', 'label', 'text_a'])
    for i in range(len(sst_tests)):
        for j in range(len(sst_tests[i])):
            test_list = sst_tests[i][j]
            adjunct_list = sst_adjuncts[i][j]
            for adjunct in adjunct_list:
                tsv_writer.writerow([str(qid), str(1), adjunct])
                qid += 1
            for test in test_list:
                tsv_writer.writerow([str(qid), str(1), test])
                qid += 1


def gen_input_for_treelstm(input_path, target_path, sst_tests, sst_adjuncts):
    input_file = open(input_path, "w")
    target_file = open(target_path, "w")
    for i in range(1100, len(sst_tests)):
        test_list = sst_tests[i]
        adjunct_list = sst_adjuncts[i]
        for adjunct in adjunct_list:
            input_file.write(adjunct + " .\n")
            target_file.write("2 Neutral\n")

        for test in test_list:
            if test.split(" ")[-1] not in [".", "!", "?", "'"]:
                test = test.rstrip().strip() + " ."
            input_file.write(test + "\n")
            target_file.write("2 Neutral\n")

    input_file.close()
    target_file.close()


def format_question(ques):
    abbr = ["n't", "'s", "'re", "'ll", "'m"]
    for a in abbr:
        if a in ques:
            idx = ques.find(a)
            if ques[idx - 1] != " ":
                ques = ques[0:idx] + " " + ques[idx:]
    ques = ques.replace(", ", " , ")
    ques = ques.replace("?", " ?")
    ques = ques.replace("\"", " \" ")
    ques = " ".join(ques.split())

    return ques


def generate_final_json(context_sentence_len_list, origin_sent_list, final_result_dic, s_idx, e_idx):
    # 获取每个context对应的问题
    question_sent_li_all = read_question_index()
    # 读处理后的源文件
    file_name = "dev_start_modify.json"
    path = "./Squad2/" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)
    prediction_data = prediction_json["data"]
    item_list = []
    # 循环处理context
    # for i in range(0, len(context_sentence_len_list)):
    for i in range(s_idx, e_idx):
        # w.write("context_id = " + str(i) + "\n")
        context_input = final_result_dic["context" + str(i)]
        if context_sentence_len_list[i] == 1:
            pro_context_list = [origin_sent_list[i]]
        else:
            if context_sentence_len_list[i] > 4:
                ll_len = context_sentence_len_list[i]
                ll_temp_list = [x for x in range(0, ll_len) if len(context_input[x]) > 1]
                if len(ll_temp_list) > 4:
                    # 不大于4，则全部改变，不用再赋值为原句
                    random.shuffle(ll_temp_list)
                    for s in ll_temp_list[4:ll_len]:
                        # 只需要改变ll_temp_list中没被选中的，因为不在ll_temp_list则长度为1，等价于原句
                        context_input[s] = [origin_sent_list[i][s]]
                    for s in ll_temp_list[0:4]:
                        context_input[s] = list(context_input[s])[0:4]

            # 对context_input中，最多4个句子改变，其他len-4个不变，随机取
            pro_context_list = itertools.product(*context_input)
        print("========")
        final_context_list = []
        for pro_context in pro_context_list:
            final_context_list.append(list(pro_context))

        # 读对应的那一条context的问题
        item_temp = copy.deepcopy(prediction_data[0]["paragraphs"][i])
        question_list = []
        ans_list = []
        for jj in item_temp["qas"]:
            question_list.append(jj["question"])
            ans_temp = [ans['text'] for ans in jj["answers"]]
            # ans_list.append(jj["answers"])
            ans_list.append(list(set(ans_temp)))
        print(question_list)

        test_set = []
        index = 0
        question_sent_li = question_sent_li_all[i]

        for pro_context_index in range(0, len(final_context_list)):
            context_for_input_li = final_context_list[pro_context_index]
            for question_sent_index in range(len(question_sent_li)):
                index_i = int(question_sent_li[question_sent_index])
                if index_i != -1:
                    ques_ans = ans_list[question_sent_index]
                    not_in_ctx = [ans for ans in ques_ans if ans not in context_for_input_li[index_i]]
                    if len(not_in_ctx) != 0:
                        for sent in context_input[index_i]:
                            not_in_ctx = [ans for ans in ques_ans if ans not in context_for_input_li[index_i]]
                            if len(not_in_ctx) == 0:
                                context_for_input_li[index_i] = sent
                                print("got!")
                                break
                        not_in_ctx = [ans for ans in ques_ans if ans not in context_for_input_li[index_i]]
                        if len(not_in_ctx) != 0:
                            context_for_input_li[index_i] = origin_sent_list[i][index_i]
                            print("Change!")
                    else:
                        print("do not need change")

                question_temp = item_temp["qas"][question_sent_index]
                question_temp["id"] = create_id()
                item_to_add = []
                item_to_add.append(question_temp)
                item_temp_modify = copy.deepcopy(item_temp)
                item_temp_modify["qas"] = item_to_add
                context_all_line = "".join(context_for_input_li)
                item_temp_modify["context"] = context_all_line
                valid_temp = context_all_line + str(question_list[question_sent_index])
                if valid_temp in test_set:
                    continue
                else:
                    test_set.append(valid_temp)
                    item_list.append(copy.deepcopy(item_temp_modify))
                    index += 1
            # print(index)
    print("success")
    prediction_data[0]["paragraphs"] = item_list
    file_name = "dev_modify_200.json"
    path = "./" + file_name
    with open(path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(prediction_json, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    # sst bert
    # file_name = "sst"
    # dataset = "sst"
    # label_path = "./comp_res/w_nsst_result_greedy.sents"
    # temp_list, adjunct_list, ner_list, for_list, hyp_words_list, comp_list = gen_sent_temp_main(file_name, label_path, 0, 1821, dataset)
    # pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    # all_masked_word, all_masked_adjunct = gen_mask_phrase(adjunct_list, pos_list, ner_list, for_list, hyp_words_list)
    # test_file_path = "./" + file_name + "_bert_test_2.txt"
    # adjunct_file_path = "./" + file_name + "_bert_adjunct_2.txt"
    # out_file = "./senta_sst_2.tsv"
    # sst_tests, sst_adjuncts = gen_tests_for_sst(comp_list, temp_list, all_masked_word, all_masked_adjunct)
    # save_new_tests_for_sst(test_file_path, sst_tests)
    # save_new_tests_for_sst(adjunct_file_path, sst_adjuncts)
    # gen_input_for_senta(out_file, sst_tests, sst_adjuncts)

    # sst_tests = []
    # test_file = open(test_file_path, "r")
    # tests = []
    # line = test_file.readline()
    # while line:
    #     if ("sent_id =" in line) & (line != "sent_id = 0\n"):
    #         sst_tests.append(tests)
    #         tests = []
    #     elif (line != "\n") & ("sent_id" not in line):
    #         line = line[:-1]
    #         tests.append(line)
    #     line = test_file.readline()
    # if len(tests) != 0:
    #     sst_tests.append(tests)
    #
    # sst_adjuncts = []
    # adjunct_file = open(adjunct_file_path, "r")
    # line = adjunct_file.readline()
    # adjuncts = []
    # while line:
    #     if ("sent_id =" in line) & (line != "sent_id = 0\n"):
    #         sst_adjuncts.append(adjuncts)
    #         adjuncts = []
    #     elif (line != "\n") & ("sent_id" not in line):
    #         line = line[:-1]
    #         adjuncts.append(line)
    #     line = adjunct_file.readline()
    # if len(adjuncts) != 0:
    #     sst_adjuncts.append(adjuncts)
    #
    #
    # gen_input_for_treelstm("./tree_input.txt", "./target_labels.txt", sst_tests, sst_adjuncts)

    ### squad bert
    # file_name = "context"
    # label_path = "./comp_res/ncontext_result_greedy.sents"
    # temp_list, adjunct_list, ner_list, for_list, hyp_words_list, comp_list = gen_sent_temp_main(file_name, label_path,
    #                                                                                             5828, 5829, "squad")
    # pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    # all_masked_word, all_masked_adjunct = gen_mask_phrase(adjunct_list, pos_list, ner_list, for_list, hyp_words_list)
    #
    # file_path = "./" + file_name + "_bert_test.txt"
    # all_tests, final_result = gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct)
    # print(final_result)

    # 获取context里句子个数的列表
    # context_sentence_len_list, origin_sent_list = get_sentence_len(file_name)
    # final_result_dic = mapping_context_sentence(context_sentence_len_list, final_result)
    # print(final_result_dic)
    # generate_final_json(context_sentence_len_list, origin_sent_list, final_result_dic)

    ### squad syn
    file_name = "context"
    label_path = "./comp_res/ncontext_result_greedy.sents"
    temp_list, adjunct_list, ner_list, for_list, hyp_words_list, comp_list = gen_sent_temp_main(file_name, label_path,
                                                                                                500, 502, "squad")
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    all_masked_word, all_masked_adjunct, all_masked_word_pos = gen_mask_phrase(adjunct_list, pos_list, ner_list,
                                                                               for_list, hyp_words_list)

    file_path = "./" + file_name + "_syn_200.txt"
    all_tests, final_result = gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct,
                                              all_masked_word_pos)
    print(final_result)

    # 获取context里句子个数的列表
    # s_idx = 0
    # e_idx = 200
    # context_sentence_len_list, origin_sent_list = get_sentence_len(file_name)
    # final_result_dic = mapping_context_sentence(context_sentence_len_list, final_result, s_idx, e_idx)
    # print(final_result_dic)
    # generate_final_json(context_sentence_len_list, origin_sent_list, final_result_dic, s_idx, e_idx)


