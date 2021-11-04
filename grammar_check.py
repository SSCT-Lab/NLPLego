import spacy
from nltk import CoreNLPParser
import numpy

## Stanford Corenlp constituency parser
eng_parser = CoreNLPParser('http://127.0.0.1:9000')
## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")

## load dictionary.txt (Saved some fixed collocations)
def load_dictionary(d_path):
    d = open(d_path, "r")
    line = d.readline()
    dictionary = {}
    while line:
        if "key " in line:
            key = line[:-1].split(" ")[-1]
            dictionary[key] = []
        elif len(line) > 1:
            l_words = line[:-1].split(" ")
            index = l_words.index(key) - 1
            dictionary[key].append(l_words[index])
        line = d.readline()
    return dictionary

## obtain constituency parser tree
def get_nlp_tree(sent):
    words = sent.split(" ")
    par_res = eng_parser.parse(words)
    for line in par_res:
        nlp_tree = line
    return nlp_tree

## check the continuity of phrase/clause
def check_continuity(key_words, words):
    flag = False
    s_idx = -1
    print("words: ", words)
    while not flag:
        s_idx = words.index(key_words[0], s_idx + 1)
        for i in range(s_idx, s_idx + len(key_words)):
            if words[i] == key_words[i - s_idx]:
                flag = True
            else:
                flag = False
                break
    return s_idx

## fill the flag_array by 1(the position of phrase or clause is 1, staring word is 2)
def fill_sent_flag(sent_flag, s_idx, e_idx):
    sent_flag[s_idx] = 2
    for i in range(s_idx + 1, e_idx):
        sent_flag[i] = 1
    return sent_flag

## load compress result from file, 1 is reserved
def load_label(label_path):
    labels = open(label_path, mode="r")
    label = labels.readline()
    label_list = []
    while label:
        label = label.split(" ")[1:-1]
        label = [int(x) for x in label]
        label_list.append(label)
        label = labels.readline()
    return label_list

## load the original sentences from file
def load_orig_sent(orig_path):
    orig_sents = open(orig_path, mode="r")
    sent = orig_sents.readline()
    sent_list = []
    while sent:
        sent = sent[:-1]
        s_words = sent.split(" ")[1:-1]
        sent = " ".join(s_words)
        sent_list.append(sent)
        sent = orig_sents.readline()
    return sent_list


# def exist_phrase(tree):
#     for st in tree.subtrees():
#         if st.leaves() != tree.leaves():
#             label = st.label()
#             if label == "PP":
#                 return True
#     return False


# def get_prep_list_by_constituency(sent):
#     nlp_tree = get_nlp_tree(sent)
#     print(nlp_tree)
#     words = sent.split(" ")
#     pp_flag = [0] * len(words)
#     pp_list = []
#     for st in nlp_tree.subtrees():
#         label = st.label()
#         if label == "PP":
#             pp_words = st.leaves()
#             pp = " ".join(pp_words)
#             s_idx = check_continuity(pp_words, words)
#             pp_flag = fill_pp_flag(pp_flag, s_idx, s_idx + len(st.leaves()))
#             pp_list.append(pp)
#     if pp_list:
#         return True, pp_list, pp_flag
#     else:
#         return False, pp_list, pp_flag


## When there are multiple prepositional phrases, cut and keep the first one
def check_pp_end(subtree, end_word, pw):
    s_idx = subtree.index(pw)
    for w in end_word:
        if w in subtree[s_idx+1:]:
            finish = w
            break
    e_idx = subtree.index(finish, s_idx+1)
    pp = subtree[:e_idx + 1]
    return pp


## obtain all prepositional phrases in one sentence by dependency relation
def get_prep_list_by_dependency(sent):
    print(sent)
    pp_list = []
    doc = nlp(sent)
    prep_of = []
    dictionary = load_dictionary("./Dictionary.txt")
    for token in doc:
        if (token.text == "of") & (token.dep_ == "prep"):
            if token.head.pos_ in ["ADJ", "VERB"]:
                prep_of.append(token.head.text)
            elif token.head.text.lower() in dictionary["of"]:
                prep_of.append(token.head.text)
            else:
                prep_of.append('X')
        # print('{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.pos_, token.dep_, token.head.text, token.head.pos_))
    noun_chunks = []
    for i in doc.noun_chunks:
        noun_chunks.append(i.text)
    for w in doc:
        if w.pos_ == "ADP":
            network = [t.text for t in list(w.children)]
            if len(network) != 0:
                subtree = [tok.orth_ for tok in w.subtree]
                pos_list = [tok.pos_ for tok in w.subtree]
                if pos_list.count('ADP') >= 2:
                    pp = check_pp_end(subtree, network, w.text)
                else:
                    pp = subtree
                if (w.head.pos_ == "VERB") & (w.dep_ == "prep"):
                    print(w.head.pos_, w.dep_, w.head.text)
                    alternative = list(pp)
                    pp.insert(0, w.head.text)
                pp = " ".join(pp)
                if "-" in pp:
                    pp = pp.replace(" - ", "-")
                if pp in sent:
                    pp_list.append(pp)
                else:
                    pp = " ".join(alternative).replace(" - ", "-")
                    if pp in sent:
                        pp_list.append(pp)
    #print(pp_list)
    new_pp_list = list(pp_list)
    of_count = 0
    for i in range(len(pp_list)):
        if "of " in pp_list[i]:
            of_flag = False
            for j in range(i-1, -1, -1):
                if prep_of[0] in pp_list[j]:
                    new_pp = pp_list[j] + " " + pp_list[i]
                    new_pp_list.insert(i + 1 - of_count, new_pp)
                    new_pp_list.pop(i-of_count)
                    new_pp_list.pop(j-of_count)
                    of_flag = True
                    prep_of.pop(0)
                    of_count += 1
                    break
            if not of_flag:
                if prep_of[0] != 'X':
                    new_pp = prep_of[0] + " " + pp_list[i]
                    if new_pp in sent:
                        new_pp_list[i - of_count] = new_pp.replace(" - ", "-")
                prep_of.pop(0)

    #print(new_pp_list)
    return new_pp_list, noun_chunks


def get_res_by_label(words, comp_label):
    res_words = []
    for i in range(len(words)):
        if comp_label[i] == 1:
            res_words.append(words[i])
    comp_res = " ".join(res_words)
    print("final result: ", comp_res)
    return comp_res


## Check the integrity of prepositional phrases in the compression results
def check_pp_integrity(words, comp_label, orig_pp, pp_flag, noun_chunks):
    s_idx = -1
    res_label = list(comp_label)
    print(comp_label)
    for i in range(len(orig_pp)):
        s_idx = pp_flag.index(2, s_idx + 1)
        if comp_label[s_idx] == 1:
            maintain = True
        else:
            maintain = False
        pp_len = len(orig_pp[i].split(" "))

        if not maintain:
            maintain_flag = True
            count = 0
            for j in range(s_idx + 1, s_idx + pp_len):
                if res_label[j] == 1:
                    count += 1

            if maintain_flag:
                if count > (pp_len - 1)/2:
                    res_label[s_idx] = 1
                    maintain = True

        for j in range(s_idx+1, s_idx + pp_len):
            if words[j] == ",":
                maintain = False
            if maintain:
                if res_label[j] != -1:
                    res_label[j] = 1
            else:
                res_label[j] = 0

        if maintain & (s_idx > 0):
            for n in noun_chunks:
                if words[s_idx - 1] in n.split(" ")[-1]:
                    res_label[s_idx - 1] = 1

    # res_words = []
    # s_words = sent.split(" ")
    # for i in range(len(s_words)):
    #     if res_label[i] == 1:
    #         res_words.append(s_words[i])
    # res_line = " ".join(res_words)
    # print(res_label)
    # print("modify result: ", res_line)
    return res_label


## Check the integrity of prepositional phrases in the compression results
def check_sbar_integrity(words, comp_label, sbar_list, sbar_flag):
    s_idx = -1
    res_label = list(comp_label)
    print(comp_label)
    for i in range(len(sbar_list)):
        s_idx = sbar_flag.index(2, s_idx + 1)
        sbar_len = len(sbar_list[i].split(" "))
        if sbar_len == 1:
            del_flag = True
            for j in range(s_idx):
                if comp_label[j] != 0:
                    del_flag = False
                    break
            if del_flag:
                res_label[s_idx] = -1
                continue
        else:
            if comp_label[s_idx] == 0:
                for j in range(s_idx, s_idx + sbar_len):
                    res_label[j] = -1
            else:
                for j in range(s_idx, s_idx + sbar_len):
                    res_label[j] = 1
    # res_words = []
    # for i in range(len(words)):
    #     if res_label[i] == 1:
    #         res_words.append(words[i])
    # comp_sent = " ".join(res_words)
    # print(res_label)
    # print("modify result: ", comp_sent)
    return res_label


def exist_sbar(nlp_tree):
    count = 0
    for s in nlp_tree.subtrees():
        pos_list = s.pos()
        if s.label() == "SBAR":
            count += 1
        if (s.label() == "PP") & (pos_list[0][0] in ["while", "when"]):
            count += 1
    if count >= 2:
        return True
    else:
        return False


def devide_sbar(pos_list, long_sbar, nlp_tree):
    count = 0
    for s in nlp_tree.subtrees():
        if s.label() == "SBAR":
            count += 1
        if (s.label() == "PP") & (pos_list[0][0] in ["while", "when"]):
            count += 1
        if count == 2:
            sub_string = " ".join(s.leaves()).replace(" - ", "-")
            break
    sbar = ""
    if pos_list[0][1] in ['IN', 'WDT', 'WP', 'WRB', "WP$"]:
        sbar = long_sbar.split(sub_string)[0]
        sbar = sbar.replace(" - ", "-").strip().rstrip()
        sbar = sbar.split(" , ")[0]

    return sbar
    # index_list = []
    # sbar = ""
    # for i in range(len(pos_list)):
    #     if pos_list[i][1] in ['IN', 'WDT']:
    #         index_list.append(i)
    #
    # if pos_list[0][1] in ['IN', 'WDT']:
    #     sbar = " ".join(words[0:index_list[0]])
    #     sbar = sbar.replace(" - ", "-")
    #
    # return sbar


def get_phrase_idx(words, phrase):
    count = words.count(phrase[0])
    if count == 1:
        idx = words.index(phrase[0])
        return idx
    else:
        i = 0
        s_idx = -1
        while i < count:
            idx = words.index(phrase[0], s_idx + 1)
            if (words[idx + 1] == phrase[1]) & (words[idx + 2] == phrase[2]):
                return idx
            else:
                count += 1
                s_idx = idx


def check_that_clause(s_words, sbar, pos_list, dictionary):
    s_idx = get_phrase_idx(s_words, sbar)
    if pos_list[s_idx-1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        sbar = s_words[s_idx - 1] + " " + "that"
        return False, sbar
    elif pos_list[s_idx-1][1] in ['JJ']:
        sbar = s_words[s_idx - 2] + " " + s_words[s_idx - 1] + " " + "that"
        return False, sbar
    elif pos_list[s_idx-1][0] in dictionary['that']:
        sbar = s_words[s_idx - 1] + " " + "that"
        return False, sbar
    else:
        return True, ""

    # if s_words[s_idx - 1] in dictionary["that"]:
    #     sbar = s_words[s_idx - 1] + " " + "that"
    #     return False, sbar
    # else:
    #     return True, ""

def check_comma(words, comp_label):
    comma_flag = False
    start_flag = False
    count = 0
    for i in range(len(words)):
        if comp_label[i] == 1:
            count += 1
        if comp_label[i] == 1:
            start_flag = True
        if (words[i] == ",") & start_flag:
            if count > 3:
                comma_flag = True
        if comma_flag & (comp_label[i] == 1) & (words[i] != "."):
                comp_label[i] = 0

    return comp_label


def extra_sbar(sent, dictionary):
    sbar_list = []
    words = sent.replace("-", " - ").split(" ")
    nlp_tree = get_nlp_tree(sent)
    all_pos_list = nlp_tree.pos()
    for s in nlp_tree.subtrees():
        label = s.label()
        pos_list = s.pos()
        if (label == "SBAR") | ((label == "PP") & (pos_list[0][0] in ["while", "when"])):
            s_idx = check_continuity(" ".join(s.leaves()).split(" "), sent.replace("-", " - ").split(" "))
            if s_idx >= 1:
                if (all_pos_list[s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ']) & (all_pos_list[s_idx][1] not in ["IN", "WDT", "WP", "WP$", "WRB"]):
                    sbar = all_pos_list[s_idx - 1][0]
                    if sbar in dictionary['that']:
                        sbar_list.append(sbar)
                    continue
            if not exist_sbar(s):
                if pos_list[0][1] in ["IN", "WDT", "WP", "WP$", "WRB"]:
                    sbar = " ".join(s.leaves())
                    sbar = sbar.replace(" - ", "-")
                    sbar_list.append(sbar)
            else:
                long_sbar = " ".join(s.leaves()).replace(" - ", "-")
                sbar = devide_sbar(pos_list, long_sbar, s)
                if len(sbar) > 0:
                    sbar_list.append(sbar)

    for i in range(len(sbar_list)):
        sbar_words = sbar_list[i].split(" ")
        if sbar_words[0] == "that":
            flag, sbar = check_that_clause(words, sbar_words, all_pos_list, dictionary)
            if not flag:
                sbar_list[i] = sbar

    return sbar_list


def filter_pp_in_sbar(sbar_list, pp_list):
    if len(sbar_list) > 0:
        res_pp = list(pp_list)
        for pp in pp_list:
            for sbar in sbar_list:
                if pp in sbar:
                    res_pp.remove(pp)
        return res_pp
    else:
        return pp_list


def write_list_in_txt(comp_list, orig_comp, file_path):
    f = open(file_path, "w")
    for i in range(len(comp_list)):
        f.write("i = " + str(i) + "\n")
        f.write("original: " + orig_comp[i] + "\n")
        f.write("modifiy: " + comp_list[i] + "\n")
        f.write("\n")

## check completeness of prep phrases, clause
def check_grammar(orig_sents, comp_label):
    comp_list = []
    orig_comp = []
    dictionary = load_dictionary('./Dictionary.txt')
    for i in range(len(orig_sents)):
        res_label = list(comp_label[i])
        sbar_list = extra_sbar(orig_sents[i], dictionary)
        pp_list, noun_chunks = get_prep_list_by_dependency(orig_sents[i])
        print("sbar: ", sbar_list)
        print("prep phrase: ", pp_list)
        words = orig_sents[i].split(" ")
        if len(sbar_list) > 0:
            sbar_flag = [0] * len(words)
            for sbar in sbar_list:
                sbar_words = sbar.split(" ")
                s_idx = check_continuity(sbar_words, words)
                sbar_flag = fill_sent_flag(sbar_flag, s_idx, s_idx + len(sbar_words))
            print("sbar_flag: ", sbar_flag)
            res_label = check_sbar_integrity(words, res_label, sbar_list, sbar_flag)
            print("after sbar process: ", res_label)
        res_pp = filter_pp_in_sbar(sbar_list, pp_list)
        if len(res_pp) > 0:
            pp_flag = [0] * len(words)
            for pp in res_pp:
                pp_words = pp.split(" ")
                s_idx = check_continuity(pp_words, words)
                pp_flag = fill_sent_flag(pp_flag, s_idx, s_idx + len(pp_words))
            print("pp_flag: ", pp_flag)
            res_label = check_pp_integrity(words, res_label, res_pp, pp_flag, noun_chunks)
            print("after prep process: ", res_label)
        #res_label = check_comma(words, res_label)
        empty_flag = True
        for j in range(0, len(res_label) - 1):
            if res_label[j] == 1:
                empty_flag = False
                break
        if empty_flag:
            res_label = comp_label[i]
        orig_comp.append(get_res_by_label(words, comp_label[i]))
        comp_res = get_res_by_label(words, res_label)
        comp_list.append(comp_res)
    write_list_in_txt(comp_list, orig_comp, "./modify_res.txt")


if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    comp_label = load_label("./comp_label/slahan_w_syn/2_" + file_name + "_result_greedy.sents")
    orig_sents = load_orig_sent(sent_path)
    check_grammar(orig_sents, comp_label)

