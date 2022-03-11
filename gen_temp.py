from grammar_check import *
import re
from nltk import CoreNLPParser
import spacy

## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")

eng_parser = CoreNLPParser('http://127.0.0.1:9000', tagtype='pos')


# 命名实体：
def extract_ner(sent):
    doc = nlp(sent)
    ner_list = []
    hyp_words, spill_words_list = get_hyphen_word(sent)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'ORG', 'NORP', 'PRODUCT', 'EVENT', 'LOC'] and len(ent.text.split()) >= 2:
            ner_list.append(ent.text)
            # print(ent.text, ent.label_)
    extract_ner_byAlpha(sent, ner_list)
    ## 检查是否包含了完整的连接词或者特殊词语
    for i in range(len(ner_list)):
        ner = ner_list[i]
        n_w = ner.split(" ")
        sent_words = sent.split(" ")
        s_idx = check_continuity(n_w, sent_words, -2)
        if s_idx == -1:
            for w in hyp_words:
                tmp_n_w = list(n_w)
                tmp_n_w[0] = w
                if check_continuity(tmp_n_w, sent_words, -2) != -1:
                    ner_list[i] = " ".join(tmp_n_w)
                    break
                tmp_n_w = list(n_w)
                tmp_n_w[-1] = w
                if check_continuity(tmp_n_w, sent_words, -2) != -1:
                    ner_list[i] = " ".join(tmp_n_w)
                    break
        ner = ner_list[i]
        n_w = ner.split(" ")
        s_idx = check_continuity(n_w, sent_words, -2)
        if s_idx == -1:
            s_idx = check_continuity(n_w[:-1], sent_words, -2)
            e_idx = s_idx + len(n_w) - 1
            if (sent_words[e_idx] != n_w[-1]) & (n_w[-1] in sent_words[e_idx]):
                ner_list[i] = " ".join(n_w[:-1]) + " " + sent_words[e_idx]

    return ner_list


def extract_ner_byAlpha(text, ner_list):
    arr = text.split()
    arr[0] = arr[0][0].lower() + arr[0][1:-1]
    for i in range(len(arr)):
        if not arr[i][0].isupper():
            arr[i] = '#'
    s = ''
    # print(arr)
    for word in arr:
        if word == '#':
            s = s.strip()
            if len(s.split()) > 1 and str_in_list(s, ner_list):
                list_in_str(s, ner_list)
                ner_list.append(s)
            s = ''
        else:
            s += word + ' '
    return ner_list


def str_in_list(s, li):
    for i in li:
        if s in i:
            return False
    return True


def list_in_str(s, li):
    for item in li:
        if item in s:
            li.remove(item)


def convert_label(orig_sent, comp_label):
    temp_words = []
    source_words = orig_sent.split(" ")
    for i in range(len(comp_label)):
        if comp_label[i] == 1:
            temp_words.append(source_words[i])
        else:
            if source_words[i] == ",":
                temp_words.append(source_words[i])
            else:
                temp_words.append("0")

    return temp_words


def judge_divide(sbar_words, pp_words):
    if " ".join(pp_words) in " ".join(sbar_words):
        s_idx = sbar_words.index(pp_words[0])
        e_idx = sbar_words.index(pp_words[len(pp_words) - 1])
        count = s_idx + len(sbar_words) - e_idx - 1
        if count < 4:
            return False
        else:
            return True


def get_correct_sidx(orig_words, key_words):
    w_count = orig_words.count(key_words[0])
    if w_count == 1:
        s_idx = orig_words.index(key_words[0])
        e_idx = s_idx + 1
        while orig_words[e_idx] != key_words[-1]:
            e_idx += 1
        return s_idx, e_idx
    else:
        if key_words.count(key_words[0]) == 1:
            s_idx = -1
            for i in range(w_count):
                s_idx = orig_words.index(key_words[0], s_idx + 1)
                e_idx = orig_words.index(key_words[-1], s_idx + 1)
                part_words = orig_words[s_idx:e_idx + 1]
                if part_words.count(key_words[0]) == 1:
                    return s_idx, e_idx
        else:
            print(key_words)
            search_key = 1
            while (orig_words.count(key_words[search_key]) != 1) & (search_key < len(key_words)):
                search_key += 1

            search_key_idx = orig_words.index(key_words[search_key])
            s_idx = search_key_idx - 1
            key_words_idx = search_key - 1
            while (s_idx >= 0) & (key_words_idx >= 0):
                if orig_words[s_idx] == key_words[key_words_idx]:
                    key_words_idx -= 1
                s_idx -= 1
            e_idx = search_key_idx + 1
            key_words_idx = search_key + 1
            while (e_idx < len(orig_words)) & (key_words_idx < len(key_words)):
                if orig_words[e_idx] == key_words[key_words_idx]:
                    key_words_idx += 1
                e_idx += 1
            return s_idx + 1, e_idx - 1



    # if (cut_s_idx + add_idx < cut_idx[0]) & (cut_idx[0] < cut_e_idx + add_idx):
    #     s_idx = cut_s_idx + add_idx
    #     e_idx = cut_e_idx + add_idx + cut_idx[1] - cut_idx[0] + 1
    #     add_idx += cut_idx[1] - cut_idx[0] + 1
    #     return True, s_idx, e_idx, add_idx
    # else:
    #     return False, -1, -1, add_idx


def gen_temp(orig_sents, cut_sents, comp_labels, start_idx, end_idx):
    sbar_pattern = re.compile(r's\d+')
    pp_pattern = re.compile(r'p\d+')
    temp_list = []
    adjunct_list = []
    all_ner = []
    comp_list = []
    dictionary = load_dictionary('./Dictionary.txt')
    for i in range(start_idx, end_idx):
        # sbar_list = all_sbar[i]
        # pp_list = all_pp[i]
        # for_list = all_formulations[i]
        orig_sent = orig_sents[i].replace("``", "\"").replace("''", "\"")
        cut_sent = cut_sents[i].replace("``", "\"").replace("''", "\"")
        s_words = orig_sent.split(" ")
        cut_s_words = cut_sent.split(" ")
        print("original sentences: ", orig_sent)
        ner_list = extract_ner(orig_sent)
        res_label, sbar_list, pp_list, conj_res, for_list = grammar_check_one_sent(i, orig_sent, cut_sent, comp_labels[i], dictionary)
        comp_res = get_res_by_label(s_words, res_label)
        temp_words = convert_label(orig_sent, res_label)
        cut_idx_list = search_cut_content(s_words)
        if len(sbar_list) != 0:
            for j in range(len(sbar_list)):
                if len(sbar_list[j].split(" ")) > 2:
                    if len(cut_idx_list) != 0:
                        s_idx, e_idx = get_correct_sidx(s_words, sbar_list[j].split(" "))
                    else:
                        s_idx = check_continuity(sbar_list[j].split(" "), s_words, -2)
                        e_idx = s_idx + len(sbar_list[j].split(" ")) - 1
                    exist_flag = True
                    for s in range(s_idx, e_idx + 1):
                        if temp_words[s] != "0":
                            exist_flag = False
                            break
                    if exist_flag:
                        for s in range(s_idx, e_idx + 1):
                            temp_words[s] = "s" + str(j)
        ## devide modifying prep
        if len(pp_list) != 0:
            for j in range(len(pp_list)):
                if len(cut_idx_list) != 0:
                    s_idx, e_idx = get_correct_sidx(s_words, pp_list[j][1].split(" "))
                else:
                    s_idx = check_continuity(pp_list[j][1].split(" "), s_words, -2)
                    e_idx = s_idx + len(pp_list[j][1].split(" ")) - 1
                exist_flag = True
                for p in range(s_idx, e_idx + 1):
                    result = sbar_pattern.findall(temp_words[p])
                    if (temp_words[p] != "0") & (len(result) == 0):
                        exist_flag = False
                        break
                if len(result) != 0:
                    index = int(result[0][-1])
                    sbar = sbar_list[index]
                    if (not judge_divide(sbar.split(" "), pp_list[j][1].split(" "))) | (pp_list[j][0] == "v"):
                        continue
                if exist_flag & (pp_list[j][0] == "p"):
                    if s_words[s_idx - 1] in ["and", "or", "but"]:
                        s_idx = s_idx - 1
                    for p in range(s_idx, e_idx + 1):
                        temp_words[p] = "p" + str(j)
        # ner need to maintan the same value
        if len(ner_list) != 0:
            print("ner_list:", ner_list)
            for j in range(len(ner_list)):
                s_idx = check_continuity(ner_list[j].split(" "), s_words, -2)
                e_idx = s_idx + len(ner_list[j].split(" ")) - 1
                s_flag = temp_words[s_idx]
                sbar_match = sbar_pattern.findall(s_flag)
                pp_match = pp_pattern.findall(s_flag)
                if (temp_words[j] == "0") | (len(sbar_match) != 0) | (len(pp_match) != 0):
                    for n in range(s_idx + 1, e_idx):
                        if temp_words[n] != s_flag:
                            temp_words[n] = s_flag
        if len(for_list) != 0:
            print("for_list:", ner_list)
            for j in range(len(for_list)):
                s_idx = check_continuity(for_list[j].split(" "), s_words, -2)
                e_idx = s_idx + len(for_list[j].split(" ")) - 1
                s_flag = temp_words[s_idx]
                sbar_match = sbar_pattern.findall(s_flag)
                pp_match = pp_pattern.findall(s_flag)
                if (temp_words[j] == "0") | (len(sbar_match) != 0) | (len(pp_match) != 0):
                    for n in range(s_idx + 1, e_idx):
                        if temp_words[n] != s_flag:
                            temp_words[n] = s_flag
        slot = 0
        adjuncts = []
        adjunct = []

        for j in range(0, len(temp_words)):
            sbar_match = sbar_pattern.findall(temp_words[j])
            pp_match = pp_pattern.findall(temp_words[j])
            if (temp_words[j] == "0") | (len(sbar_match) != 0) | (len(pp_match) != 0):
                flag = temp_words[j]
                temp_words[j] = "t" + str(slot)
                adjunct.append(s_words[j])
                if j + 1 < len(temp_words):
                    if temp_words[j + 1] != flag:
                        slot = slot + 1
                        adjuncts.append(" ".join(adjunct))
                        adjunct = []
        print("temp: ", " ".join(temp_words))
        print("adjuncts: ", adjuncts)
        temp_list.append(" ".join(temp_words))
        adjunct_list.append(adjuncts)
        all_ner.append(ner_list)
        comp_list.append(comp_res)
    return temp_list, adjunct_list, all_ner, comp_list


def save_temp_adjuncts(temp_list, adjunct_list, file_name):
    temp_file = open("./temp_adjunct/" + file_name + "_temp.txt", "w")
    adjunct_file = open("./temp_adjunct/" + file_name + "_adjunct.txt", "w")
    for i in range(len(temp_list)):
        temp_file.write(temp_list[i] + "\n")
        adjunct_file.write(";".join(adjunct_list[i]) + "\n")

    temp_file.close()
    adjunct_file.close()


def format_sent(sent):
    words = sent.split()
    for i in range(len(words)):
        words[i] = words[i].strip().rstrip()
    sent = " ".join(words)
    sent = sent.replace(", ,", ",")
    sent = sent.replace(", .", ".")
    sent = sent.replace(" - ", "-")
    return sent


def gen_step_sentence(temp_list, adjunct_list, comp_list, file_name):
    all_sents = []
    f = open("./temp_adjunct/" + file_name + "_step.txt", "w")
    for i in range(len(temp_list)):
        f.write("sent_id = " + str(i) + "\n")
        sbar_pattern = re.compile(r't\d+')
        sent_list = []
        temp = temp_list[i]
        f.write(comp_list[i] + "\n")
        sent_list.append(comp_list[i])
        for j in range(len(adjunct_list[i])):
            slot = ["t" + str(j)] * temp.count("t" + str(j))
            slot = " ".join(slot)
            sent = temp.replace(slot, adjunct_list[i][j])
            temp = sent
            result = set(sbar_pattern.findall(sent))
            sent_word = sent.split(" ")
            if len(result) != 0:
                for r in result:
                    slot = [r] * sent_word.count(r)
                    sent = sent.replace(" ".join(slot), "")
                    sent = format_sent(sent)
            f.write(sent + "\n")
            sent_list.append(sent)
        f.write("\n")
        all_sents.append(sent_list)
    f.close()
    return all_sents


def gen_sent_temp_main(file_name, start_idx, end_idx):
    orig_sent_path = "./comp_input/" + file_name + ".cln.sent"
    orig_sents = load_orig_sent(orig_sent_path)
    cut_sent_path = "./comp_input/n" + file_name + ".cln.sent"
    cut_sents = load_orig_sent(cut_sent_path)
    comp_labels = load_label("./ncontext_result_greedy.sents")
    #gen_temp(orig_sents, cut_sents, comp_labels, start_idx, end_idx)
    temp_list, adjunct_list, ner_list, comp_list = gen_temp(orig_sents, cut_sents, comp_labels, start_idx, end_idx)
    return temp_list, adjunct_list, comp_list, ner_list


if __name__ == '__main__':
    file_name = "context"
    start_idx = 6000
    end_idx = 6213
    gen_sent_temp_main(file_name, start_idx, end_idx)
    # sent_path = "./comp_input/" + file_name + ".cln.sent"
    # comp_label = load_label("./comp_label/slahan_w_syn/2_" + file_name + "_result_greedy.sents")
    # orig_sents = load_orig_sent(sent_path)
    # label_list, all_sbar, all_pp, all_conj, comp_list = check_grammar(orig_sents, comp_label)
    # temp_list, adjunct_list = gen_temp(orig_sents, label_list, all_sbar, all_pp)
    # gen_step_sentence(temp_list, adjunct_list, comp_list, file_name)
    # save_temp_adjuncts(temp_list, adjunct_list, file_name)
