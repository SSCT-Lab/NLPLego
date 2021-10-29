import spacy
from nltk import CoreNLPParser

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
def check_continuity(pp_words, words):
    flag = False
    s_idx = -1
    while not flag:
        s_idx = words.index(pp_words[0], s_idx + 1)
        for i in range(s_idx, s_idx + len(pp_words)):
            if words[i] == pp_words[i - s_idx]:
                flag = True
            else:
                flag = False
                break

    return s_idx

## fill the flag_array by 1(the position of phrase or clause is 1, staring word is 2)
def fill_pp_flag(pp_flag, s_idx, e_idx):
    pp_flag[s_idx] = 2
    for i in range(s_idx + 1, e_idx):
        pp_flag[i] = 1
    return pp_flag

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
    # for i in doc.noun_chunks:
    #     print(i, [t.pos_ for t in i])
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
                if (w.head.pos_ == "VERB")  & (w.dep_ == "prep"):
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
    print(pp_list)
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
                        new_pp_list[i - of_count] = prep_of[0] + " " + pp_list[i]
                prep_of.pop(0)

    print(new_pp_list)
    return new_pp_list

## Check the integrity of prepositional phrases in the compression results
def check_pp_integrity(sent, comp_label):
    orig_pp = get_prep_list_by_dependency(sent)
    #print(orig_pp, pp_flag)
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
        for j in range(s_idx+1, s_idx + pp_len):
            if maintain:
                res_label[j] = 1
            else:
                res_label[j] = 0
    res_words = []
    s_words = sent.split(" ")
    for i in range(len(s_words)):
        if res_label[i] == 1:
            res_words.append(s_words[i])
    res_line = " ".join(res_words)
    print(res_label)
    print("modify result: ", res_line)


def exist_sbar(nlp_tree):
    count = 0
    for s in nlp_tree.subtrees():
        if s.label()=="SBAR":
            count += 1
    if count >= 2:
        return True
    else:
        return False


def devide_sbar(pos_list):
    index_list = []
    for i in range(len(pos_list)):
        if pos_list[i][1] == "SBAR":
            index_list.append(i)


def extra_sbar(sent):
    sbar_list = []
    nlp_tree = get_nlp_tree(sent)
    for s in nlp_tree.subtrees():
        label = s.label()
        if label == "SBAR":
            pos_list = s.pos()
            if not exist_sbar(s):
                if pos_list[0][1] in ["IN", "WDT"]:
                    sbar = " ".join(s.leaves())
                    sbar = sbar.replace(" - ", "-")
                    sbar_list.append(sbar)
            else:
                devide_sbar(pos_list)

    print(sbar_list)

## check completeness of prep phrases, clause
def check_grammar(orig_sents, comp_label):
    for i in range(len(orig_sents)):
        words = orig_sents[i].split(" ")
        pp_flag = [0] * len(words)
        pp_list = get_prep_list_by_dependency(orig_sents[i])
        for pp in pp_list:
            pp_words = pp.split(" ")
            s_idx = check_continuity(pp_words, words)
            pp_flag = fill_pp_flag(pp_flag, s_idx, s_idx + len(pp_words))
        #check_pp_integrity(orig_sents[i], comp_label[i])
        #extra_sbar(orig_sents[i])



if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    comp_label = load_label("./comp_res/2_" + file_name + "_result_greedy.sents")
    orig_sents = load_orig_sent(sent_path)
    check_grammar(orig_sents, comp_label)

