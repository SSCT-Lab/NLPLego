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
    li = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'ORG', 'NORP', 'PRODUCT', 'EVENT', 'LOC'] and len(ent.text.split()) >= 2:
            li.append(ent.text)
            # print(ent.text, ent.label_)
    extract_ner_byAlpha(sent, li)
    return li


def extract_ner_byAlpha(text, li):
    arr = text.split()
    arr[0] = arr[0][0].lower()+arr[0][1:-1]
    for i in range(len(arr)):
        if not arr[i][0].isupper():
            arr[i] = '#'
    s = ''
    # print(arr)
    for word in arr:
        if word == '#':
            s = s.strip()
            if len(s.split()) > 1 and str_in_list(s, li):
                list_in_str(s, li)
                li.append(s)
            s = ''
        else:
            s += word + ' '
    return li


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
        e_idx = sbar_words.index(pp_words[len(pp_words)-1])
        count = s_idx + len(sbar_words) - e_idx - 1
        if count < 4:
            return False
        else:
            return True


def gen_temp(orig_sents, comp_labels, all_sbar, all_pp):
    sbar_pattern = re.compile(r's\d+')
    pp_pattern = re.compile(r'p\d+')
    temp_list = []
    adjunct_list = []
    all_ner = []
    for i in range(len(orig_sents)):
        sbar_list = all_sbar[i]
        pp_list = all_pp[i]
        ner_list = extract_ner(orig_sents[i])
        #print(pos_list)
        print("original sentences: ", orig_sents[i])
        s_words = orig_sents[i].split(" ")
        temp_words = convert_label(orig_sents[i], comp_labels[i])

        if len(sbar_list) != 0:
            for j in range(len(sbar_list)):
                if len(sbar_list[j].split(" ")) > 2:
                    s_idx = check_continuity(sbar_list[j].split(" "), s_words)
                    e_idx = s_idx + len(sbar_list[j].split(" "))
                    exist_flag = True
                    for s in range(s_idx, e_idx):
                        if temp_words[s] != "0":
                            exist_flag = False
                            break
                    if exist_flag:
                        for s in range(s_idx, e_idx):
                            temp_words[s] = "s" + str(j)
        ## devide modifying prep
        if len(pp_list) != 0:
             for j in range(len(pp_list)):
                s_idx = check_continuity(pp_list[j][1].split(" "), s_words)
                e_idx = s_idx + len(pp_list[j][1].split(" "))
                exist_flag = True
                for p in range(s_idx, e_idx):
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
                    for p in range(s_idx, e_idx):
                        temp_words[p] = "p" + str(j)
        ## ner need to maintan the same value
        if len(ner_list) != 0:
            for j in range(len(ner_list)):
                s_idx = check_continuity(ner_list[j].split(" "), s_words)
                e_idx = s_idx + len(ner_list[j].split(" "))
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
        words = orig_sents[i].split(" ")
        for j in range(0, len(temp_words)):
            sbar_match = sbar_pattern.findall(temp_words[j])
            pp_match = pp_pattern.findall(temp_words[j])
            if (temp_words[j] == "0")|(len(sbar_match) != 0)|(len(pp_match) != 0):
                flag = temp_words[j]
                temp_words[j] = "t" + str(slot)
                adjunct.append(words[j])
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
    return temp_list, adjunct_list, all_ner


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
            slot = ["t"+str(j)] * temp.count("t"+str(j))
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


def gen_sent_temp_main(file_name):
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    orig_sents = load_orig_sent(sent_path)
    label_list, all_sbar, all_pp, all_conj, comp_list = grammar_check_main(file_name)
    temp_list, adjunct_list, ner_list = gen_temp(orig_sents, label_list, all_sbar, all_pp)
    return temp_list, adjunct_list, comp_list, ner_list


if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    comp_label = load_label("./comp_label/slahan_w_syn/2_" + file_name + "_result_greedy.sents")
    orig_sents = load_orig_sent(sent_path)
    label_list, all_sbar, all_pp, all_conj, comp_list = check_grammar(orig_sents, comp_label)
    temp_list, adjunct_list = gen_temp(orig_sents, label_list, all_sbar, all_pp)
    gen_step_sentence(temp_list, adjunct_list, comp_list, file_name)
    save_temp_adjuncts(temp_list, adjunct_list, file_name)
