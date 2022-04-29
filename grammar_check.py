import re
from functools import reduce

import spacy
from nltk import CoreNLPParser, Tree

## Stanford Corenlp constituency parser
from preprocess import load_formulation, format_formulation, search_cut_content

eng_parser = CoreNLPParser('http://127.0.0.1:9000')
## SpaCy dependency parser
spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.add_pipe("merge_entities")

formulations = load_formulation('./formulation.txt')
key_formulations = []
for f in formulations:
    key_formulations.append(format_formulation(f))


## load dictionary.txt (Saved some fixed collocations)
def load_dictionary(d_path):
    d = open(d_path, "r")
    line = d.readline()
    dictionary = {}
    while line:
        if "key " in line:
            key = line[:-1].split(" ")[-1]
            if key not in ["comp", "start", "end"]:
                dictionary[key] = {}
            else:
                dictionary[key] = []
        elif len(line) > 1:
            if key not in ["comp", "start", "end"]:
                l_words = line[:-1].split(" ")
                index = l_words.index(key)
                if index != 0:
                    w_idx = index - 1
                else:
                    w_idx = index + 1
                dictionary[key][l_words[w_idx]] = line[:-1]
            else:
                dictionary[key].append(line[:-1])
        line = d.readline()
    return dictionary


## obtain constituency parser tree
def get_nlp_tree(sent):
    sent = re.sub(r'%(?![0-9a-fA-F]{2})', "%25", sent)
    sent = sent.replace("+", "%2B")
    words = sent.split(" ")
    par_res = eng_parser.parse(words)
    for line in par_res:
        nlp_tree = line
    return nlp_tree




## check the continuity of phrase/clause
def check_continuity(key_words, words, search_start):
    flag = False
    s_idx = search_start
    while not flag:
        if key_words[0] in words[s_idx + 1:]:
            s_idx = words.index(key_words[0], s_idx + 1)
        else:
            return -1
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
    orig_sents = open(orig_path, mode="r", encoding="utf-8")
    sent = orig_sents.readline()
    sent_list = []
    while sent:
        sent = sent[:-1]
        s_words = sent.split(" ")[1:-1]
        sent = " ".join(s_words)
        sent_list.append(sent)
        sent = orig_sents.readline()
    return sent_list


def exist_pp(pos_list, pp_word, dictionary, key_pp, to_flag, spill_words_list):
    count = 0
    key = -1
    sent = " ".join(pp_word).replace(" - ", "-").replace(" – ", "–")
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    i = 0
    p_idx = pp_word.index(key_pp)

    for pp in dictionary["comp"]:
        if (pp in " ".join(pp_word)) & (pp_word[0] == pp.split(" ")[0]):
            p_idx = check_continuity(pp.split(" "), pp_word, -1) + len(pp.split(" "))
            break
    # if 【"as well as" in sent:
    #     skip_idx = check_continuity("as well as".split(" "), pp_word, -1)
    # else:
    #     skip_idx = -1
    first_to = -1
    if key_pp == "from":
        if "to" in pp_word[p_idx:]:
            first_to = pp_word[p_idx:].index("to")
    for w in doc:
        if i > p_idx:
            # if skip_idx != -1:
            #     if i in range(skip_idx, skip_idx + 3):
            #         i += 1
            #     continue
            if (pos_list[i] == "ADP") & (w.text not in ["of", "v", "than"]):
                if (i - 1 >= 0) & (i + 1 < len(pp_word)) & (w.text in spill_words_list):
                    if (pp_word[i - 1] in ["-", "–", "−"]) | (pp_word[i + 1] in ["-", "–", "−"]):
                        i += 1
                        continue
                network = [t.text for t in list(w.children)]
                if len(network) != 0:
                    if w.head.pos_ == "VERB":
                        ## ing modify noun
                        if w.head.dep_ == "acl":
                            count += 1
                            key = pp_word.index(w.head.text)
                            ## already
                            if pos_list[key - 1] == "ADV":
                                key = key - 1
                            break
                        elif w.text in ["during", "after", "before"]:
                            count += 1
                        ## on Monday in March
                        elif (network[0] in dictionary["on"].keys()) | (network[0] in dictionary["in"].keys()):
                            count += 1
                        else:
                            count += 1
                    else:
                        if (key_pp == "from") & ((first_to + p_idx) == i):
                            i += 1
                            continue
                        count += 1
                        words = [tok.orth_ for tok in w.subtree]
                        if words[0] != w.text:
                            p_idx = words.index(w.text)
                            key = i - p_idx
                            break
            elif (pp_word[0] == "among") & (pos_list[i] == "VERB"):
                count += 1
            elif w.text in ["while", "when", "where", "which", "who", "that", "during", "according"]:
                if (i > 0) & (pp_word[i - 1] != "at"):
                    count += 1
            elif (w.text == "to") & (not to_flag):
                if (key_pp == "from") & ((first_to + p_idx) == i):
                    i += 1
                    continue
                count += 1
            elif i < len(pp_word) - 1:
                if (w.text == ",") & ((pp_word[i + 1] in ["called"]) | (key_pp in ["by"])):
                    count += 1
            if count == 1:
                key = i
                break
        i += 1
    if count >= 1:
        return True, key
    else:
        return False, key


def devide_pp(pp_word, key):
    to_prep = " ".join(pp_word[0:key])
    # to_prep = to_prep.replace(" - ", "-").replace(" – ", "–")
    to_prep = to_prep.split(" , ")[0]
    return to_prep


## Complement prep phrase
def get_the_complete_phrase(p_word, h_word, s_word, pp, pos_list, all_pos_list, pp_list, hyp_words, orig_sent, last_s_idx):
    new_pp = list(pp)
    new_pos_list = list(pos_list)
    if " ".join(pp) not in " ".join(s_word):
        p_idx = s_word.index(p_word)
    else:
        p_idx = check_continuity(pp, s_word, -1)
    h_idx = -1
    if h_word in s_word[0:p_idx]:
        h_idx = p_idx - 1
        while s_word[h_idx] != h_word:
            h_idx = h_idx - 1
        ## add adv (modify verb)
        if h_idx - 1 >= 0:
            if all_pos_list[h_idx - 1] == "ADV":
                ## no longer, no more
                if all_pos_list[h_idx - 2] == "ADV":
                    h_idx = h_idx - 2
                ## already
                else:
                    h_idx = h_idx - 1
            else:
                # ## be able to
                # if all_pos_list[h_idx - 1] == "VERB":
                while all_pos_list[h_idx - 1] == "VERB":
                    h_idx = h_idx - 1
        for w_tup in hyp_words:
            w = w_tup[1]
            if "-" in w:
                w_w = w.split("-")
            else:
                w_w = w.split("–")
            if s_word[h_idx] == w_w[-1]:
                h_idx = -1
                return new_pp, new_pos_list, h_idx
        idx = p_idx - 1
        while idx >= h_idx:
            new_pp.insert(0, s_word[idx])
            new_pos_list.insert(0, all_pos_list[idx])
            idx = idx - 1

        if len(pp_list) > 0:
            pp_str = process_hyp_words(" ".join(new_pp), hyp_words, orig_sent, last_s_idx)
            if pp_list[-1][1] in pp_str:
                h_idx = -1
                new_pp = pp
                new_pos_list = pos_list

    return new_pp, new_pos_list, h_idx

def get_prep_of(doc, dictionary, all_pos_list, s_word, spill_words_list):
    i = 0
    prep_of = []
    for token in doc:
        if token.text == "of":
            head_text = s_word[i - 1]
            if head_text[-3:] == "ies":
                of_type = 1
                of_key = head_text[:-3].lower() + "y"
            elif head_text[-1:] == "s":
                of_type = 2
                of_key = head_text[:-1].lower()
            else:
                of_type = 3
                of_key = head_text.lower()
            if (token.dep_ in ["prep", "cc"]) & (s_word[i - 1] != ","):
                if (token.head.pos_ in ["ADJ", "VERB"]) & (head_text in s_word[:i]):
                    if head_text == s_word[i - 1]:
                        prep_of.append(token.head.text)
                    else:
                        prefix = ""
                        j = i - 1
                        while s_word[j] != head_text:
                            prefix = s_word[j] + " " + prefix
                            j -= 1
                        prefix = head_text + " " + prefix
                        prep_of.append(prefix.strip().rstrip())
                elif of_key in dictionary["of"].keys():
                    comp_prep_word = dictionary["of"][of_key].split(" ")
                    if comp_prep_word[0] in ["a", "an"]:
                        if (of_type in [1, 2]) & (comp_prep_word[1] not in ["series"]):
                            if "NUM" in all_pos_list[0:i]:
                                search_idx = i - 1
                                while (all_pos_list[search_idx:i].count("NUM") == 0) & (
                                        s_word[search_idx] not in ["all", ","]):
                                    search_idx = search_idx - 1
                                if search_idx - i < 8:
                                    prep_of.append(" ".join(s_word[search_idx:i]))
                                else:
                                    if (s_word[i - 1] in spill_words_list) & (s_word[i - 2] in spill_words_list):
                                        prep_of.append(s_word[i - 2] + " " + s_word[i - 1])
                                    else:
                                        prep_of.append(s_word[i - 1])
                            else:
                                prep_of.append(s_word[i - 1])
                        elif "DET" in all_pos_list[0:i]:
                            search_idx = i - 1
                            noun_count = 0
                            if all_pos_list[search_idx] == "NOUN":
                                while all_pos_list[search_idx] == "NOUN":
                                    noun_count += 1
                                    search_idx -= 1
                            while all_pos_list[search_idx] not in ["DET", "PUNCT"]:
                                search_idx = search_idx - 1
                            if s_word[search_idx] == ",":
                                search_idx += 1
                            if (search_idx - i < 8) & \
                                    (all_pos_list[search_idx:i].count("NOUN") + all_pos_list[search_idx:i].count(
                                        "PROPN") - noun_count == 0):
                                prep_of.append(" ".join(s_word[search_idx:i]))
                            else:
                                if (s_word[i - 1] in spill_words_list) & (s_word[i - 2] in spill_words_list):
                                    prep_of.append(s_word[i - 2] + " " + s_word[i - 1])
                                else:
                                    prep_of.append(s_word[i - 1])
                        elif ("one" in s_word[0:i]) | ("One" in s_word[0:i]):
                            search_idx = i - 1
                            while all_pos_list[search_idx:i].count("NUM") == 0:
                                search_idx = search_idx - 1
                            if search_idx - i < 8:
                                prep_of.append(" ".join(s_word[search_idx:i]))
                            else:
                                if (s_word[i - 1] in spill_words_list) & (s_word[i - 2] in spill_words_list):
                                    prep_of.append(s_word[i - 2] + " " + s_word[i - 1])
                                else:
                                    prep_of.append(s_word[i - 1])
                        else:
                            if (s_word[i - 1] in spill_words_list) & (s_word[i - 2] in spill_words_list):
                                prep_of.append(s_word[i - 2] + " " + s_word[i - 1])
                            else:
                                prep_of.append(s_word[i - 1])

                    else:
                        if s_word[i - 1] in ["percent", "%"]:
                            search_idx = i - 2
                            while all_pos_list[search_idx:i].count("NUM") == 0:
                                search_idx = search_idx - 1
                            prep_of.append(" ".join(s_word[search_idx:i]))
                        elif (all_pos_list[i - 1] == "NOUN") & (i - 2 > 0):
                            search_idx = i - 2
                            noun_count = 0
                            if all_pos_list[search_idx] == "NOUN":
                                while all_pos_list[search_idx] == "NOUN":
                                    noun_count += 1
                                    search_idx -= 1
                            while (all_pos_list[search_idx] != "DET") & (s_word[search_idx] != ","):
                                search_idx = search_idx - 1
                                if search_idx < 0:
                                    break
                            if (search_idx > 0) & (search_idx - i < 8) & \
                                    (all_pos_list[search_idx:i].count("NOUN") + all_pos_list[search_idx:i].count(
                                        "PROPN") - noun_count == 0):
                                prep_of.append(" ".join(s_word[search_idx:i]))
                            else:
                                if (s_word[i - 1] in spill_words_list) & (s_word[i - 2] in spill_words_list):
                                    prep_of.append(s_word[i - 2] + " " + s_word[i - 1])
                                else:
                                    prep_of.append(s_word[i - 1])
                        else:
                            prep_of.append(s_word[i - 1])
                elif s_word[i - 1] in ["percent", "%"]:
                    search_idx = i - 2
                    while all_pos_list[search_idx:i].count("NUM") == 0:
                        search_idx = search_idx - 1
                    prep_of.append(" ".join(s_word[search_idx:i]))
                elif all_pos_list[i - 1] == "NUM":
                    prep_of.append(s_word[i - 1])
                else:
                    prep_of.append('X')
            elif token.dep_ == "conj":
                prep_of.append(s_word[i - 1])
            else:
                prep_of.append('X')

        i += 1
    return prep_of


def get_hyphen_word(sent):
    hyphen_words = []
    s_word = sent.split(" ")
    for i in range(len(s_word)):
        w = s_word[i]
        if (("-" in w) | ("–" in w) | ("−" in w)) & (len(w) != 1):
            hyphen_words.append((i, w))

    word_list = []
    for w in hyphen_words:
        word_list.extend(re.split("-|–|−", w[1]))
        # if "-" in w:
        #     word_list.extend(w.split("-"))
        # else:
        #     word_list.extend(w.split("–"))

    return hyphen_words, word_list


def get_abbr_word(sent):
    abbr_words = []
    s_word = sent.split(" ")
    for w in s_word:
        if ("." in w) & (len(w) != 1):
            abbr_words.append(w)
    return abbr_words


def complement_pp_word(pp_word, s_word, pos_list, all_pos_list, abbr_words, spill_words_list):
    comp_f = ""
    comp_abbr = ""
    s_idx = check_continuity(pp_word, s_word, -1)
    e_idx = s_idx + len(pp_word) - 1
    ## process punct adj
    if (s_word[e_idx] == "–") & (s_word[e_idx - 1] in spill_words_list):
        pp_word.append(s_word[e_idx + 1])
        pos_list.append(all_pos_list[e_idx + 1])
    if e_idx + 1 < len(s_word):
        if (s_word[e_idx + 1] == "–") & (s_word[e_idx] in spill_words_list):
            pp_word.extend(s_word[e_idx + 1:e_idx + 3])
            pos_list.extend(all_pos_list[e_idx + 1:e_idx + 3])
    ## process formulation
    for f in formulations:
        nf = f.replace(" ", "")
        if pp_word[-1] == nf[:len(pp_word[-1])]:
            pp_for = pp_word[-1]
            for idx in range(e_idx + 1, len(s_word)):
                pp_for = pp_for + s_word[idx]
                if pp_for == nf:
                    comp_f = f
                    break
            break
    ## process abbr word
    for abbr in abbr_words:
        if pp_word[-1] == abbr[:len(pp_word[-1])]:
            pp_abbr = pp_word[-1]
            for idx in range(e_idx + 1, len(s_word)):
                pp_abbr = pp_abbr + s_word[idx]
                if pp_abbr == abbr:
                    comp_abbr = abbr
                    break
            break
    return pp_word, pos_list, comp_f, comp_abbr


def get_complete_last_word(pp_word, s_word):
    if len(pp_word) >= 3:
        s_idx = check_continuity(pp_word[:-1], s_word, -1)
    else:
        s_idx = check_continuity(pp_word, s_word, -1)
    e_idx = s_idx + len(pp_word) - 1
    if (s_word[e_idx] != pp_word[-1]) & (pp_word[-1] in s_word[e_idx]):
        pp_str = " ".join(pp_word[:-1]) + " " + s_word[e_idx]
    else:
        pp_str = " ".join(pp_word)
    return pp_str


def fill_pp_flag(pp_str, s_word, pp_flag, s_idx):
    s = ""
    i = 0
    while s != pp_str.replace(" ", ""):
        s += s_word[s_idx + i]
        s = s.replace(" ", "")
        pp_flag[s_idx + i] = 1
        i += 1

    return pp_flag


def get_verb_phrases(sent, hyp_words, spill_words_list):
    basic_elements = []
    vp_list = []
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    s_word = [tok.text for tok in doc]
    pos_list = [tok.pos_ for tok in doc]
    i = 0
    dep_map = {}
    root_verb = ""
    root_idx = -1
    for token in doc:
        if (token.dep_ == "ROOT") & (token.pos_ in ["VERB", "AUX"]) & (token.text not in ["Said"]):
            basic_elements.append((i, token.dep_, token.text, token.pos_))
            root_verb = token.text
            root_idx = i
            break
        i += 1

    if (root_idx == -1) & (("VERB" in pos_list)|("AUX" in pos_list)):
        root_idx = 0
        while pos_list[root_idx] not in ["VERB", "AUX"]:
            root_idx += 1
        root_verb = doc[root_idx].text
        basic_elements.append((root_idx, "ROOT", root_verb, doc[root_idx].pos_))

    i = 0
    for token in doc:
        if ("subj" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (
                "ROOT" in [token.head.dep_, token.head.head.dep_]):
            subj_word = [tok.orth_ for tok in token.subtree]
            subj_str = process_hyp_words(" ".join(subj_word), hyp_words, sent, -1).replace(
                "et al .", "et al.")
            subj_str = re.split(' ; | – | — | , | who | which | that | where | when ', subj_str)[0]
            basic_elements.append((i, token.dep_, token.text, subj_str))
        if ("expl" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (
                "ROOT" in [token.head.dep_, token.head.head.dep_]):
            basic_elements.append((i, token.dep_, token.text, token.pos_))
        if ("advmod" in token.dep_) & (token.head.dep_ == "ROOT") & (token.text in ["here", "there"]):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
            else:
                vp_words = s_word[i: root_idx + 1]
            adv_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
            adv_str = re.split(' ; | – | — | who | which | that | where | when | why ', adv_str)[0]
            basic_elements.append((i, token.dep_, adv_str, token.pos_))
        if ("obj" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
            else:
                vp_words = s_word[i: root_idx + 1]
            obj_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
            obj_str = re.split(' ; | – | — | who | which | that | where | when | why ', obj_str)[0]
            basic_elements.append((i, token.dep_, obj_str, token.pos_))

        if ("comp" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
                comp_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
                if (basic_elements[-1][2] not in comp_str) | (basic_elements[-1][1] == "ROOT"):
                    basic_elements.append((i, token.dep_, comp_str, token.pos_))
                    # print("comp_str:", comp_str)
        if ("attr" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
            else:
                vp_words = s_word[i: root_idx + 1]
            attr_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
            basic_elements.append((i, token.dep_, attr_str, token.pos_))
            # print("attr_str:", attr_str)
        key = token.text + "-" + str(i)
        if key not in dep_map.keys():
            dep_map[key] = []
        dep_map[key].append((token.head.text, token.head.pos_, token.dep_, token.text))
        i += 1

    if len([elem for elem in basic_elements if (("subj" in elem[1]) | ("expl" in elem[1]))]) == 0:
        propn_idx = root_idx - 1
        # while (pos_list[propn_idx] not in ["PROPN"]) & (propn_idx > 0):
        #     propn_idx -= 1
        noun_idx = root_idx - 1
        while (pos_list[noun_idx] not in ["NOUN"]) & (noun_idx > 0):
            noun_idx -= 1
        # if (propn_idx > 0) & (pos_list[propn_idx] == "PROPN"):
        #     propn_words = [tok.orth_ for tok in doc[propn_idx].subtree]
        if (noun_idx > 0) & (pos_list[noun_idx] == "NOUN"):
            noun_words = [tok.orth_ for tok in doc[noun_idx].subtree]
            subj_str = process_hyp_words(" ".join(noun_words), hyp_words, sent, -1).replace(
                "et al .", "et al.")
            subj_str = re.split(' ; | – | — | , | who | which | that | where | when ', subj_str)[0]
            basic_elements.append((noun_idx, "nsubj", doc[noun_idx].text, subj_str))

    i = 0
    for w in doc:
        if (i > 0) & (i + 1 < len(s_word)):
            if ((s_word[i - 1] in ["-", "—"]) | (s_word[i + 1] in ["-", "—"])) & (w.text in spill_words_list):
                i += 1
                continue
        if (w.dep_ == "acl") & (w.pos_ == "VERB"):
            network = [t.text for t in list(w.children)]
            if len(network) != 0:
                acl_word = [tok.orth_ for tok in w.subtree]
                for f in formulations:
                    nf = f.replace(" ", "")
                    if acl_word[-1] == nf[:len(acl_word[-1])]:
                        vp_for = acl_word[-1]
                        for idx in range(i + len(acl_word), len(s_word)):
                            vp_for = vp_for + s_word[idx]
                            if vp_for == nf:
                                comp_f = f
                                acl_word[-1] = comp_f
                                break
                        break
                acl_str = process_hyp_words(" ".join(acl_word), hyp_words, sent, -1).replace("et al .", "et al.")
                save_flag = True
                if acl_word[0] == w.text:
                    for vp in vp_list:
                        if vp[1] in acl_str:
                            save_flag = False
                            break
                    if save_flag:
                        vp_list.append(("acl", acl_str))

        key = w.text + "-" + str(i)
        if key in dep_map.keys():
            if len(dep_map[key]) != 0:
                dep_list = dep_map[key]
                for dep_re in dep_list:
                    if dep_re[1] in ["VERB", "AUX"]:
                        if (dep_re[2] == "acomp") & (dep_re[0] in s_word[:i]):
                            s_idx = i - 1
                            while s_word[s_idx] != dep_re[0]:
                                s_idx -= 1
                            vp_word = s_word[s_idx:i + 1]
                            aco_str = process_hyp_words(" ".join(vp_word), hyp_words, sent, -1).replace(
                                "et al .", "et al.")
                            if len(vp_list) > 0:
                                if vp_list[-1][1] in aco_str:
                                    continue
                                if aco_str in vp_list[-1][1]:
                                    vp_list.pop()
                                else:
                                    temp = (vp_list[-1][0],
                                            vp_list[-1][1] + " " + process_hyp_words(" ".join(vp_word[1:]), hyp_words,
                                                                                     sent, -1).replace("et al .", "et al."))
                                    if temp[1] in sent:
                                        vp_list.pop()
                                        vp_list.append(temp)
                                        continue
                            save_flag = True
                            for vp in vp_list:
                                if vp[1] in aco_str:
                                    save_flag = False
                                    break
                            if save_flag:
                                vp_list.append(("acomp", aco_str))

                        if (dep_re[2] == "auxpass") & (dep_re[0] in s_word[i:]):
                            s_idx = i
                            e_idx = s_word.index(dep_re[0], i)
                            if s_word[e_idx + 1] == "/":
                                e_idx += 1
                                while pos_list[e_idx] != pos_list[s_word.index(dep_re[0], i)]:
                                    e_idx += 1
                            vp_word = s_word[s_idx:e_idx + 1]
                            pass_str = process_hyp_words(" ".join(vp_word), hyp_words, sent, -1).replace(
                                "et al .", "et al.")
                            if len(vp_list) > 0:
                                if (vp_list[-1][1] in pass_str) | (
                                        (s_word[s_idx] == "m") & (s_word[s_idx - 1] != "\'")):
                                    continue
                                if pass_str in vp_list[-1][1]:
                                    vp_list.pop()
                            save_flag = True
                            for vp in vp_list:
                                if pass_str in vp[1]:
                                    save_flag = False
                                    break
                            if save_flag:
                                vp_list.append(("auxpass", pass_str))

                        if (dep_re[2] == "oprd") & (dep_re[0] in s_word[:i]):
                            s_idx = i - 1
                            while s_word[s_idx] != dep_re[0]:
                                s_idx -= 1
                            vp_word = s_word[s_idx:i + 1]
                            oprd_str = process_hyp_words(" ".join(vp_word), hyp_words, sent, -1).replace(
                                "et al .", "et al.")
                            ## acl and oprd can be same or included
                            if len(vp_list) > 0:
                                if vp_list[-1][1] in oprd_str:
                                    continue
                                if oprd_str in vp_list[-1][1]:
                                    vp_list.pop()
                                else:
                                    last_vp_words = vp_list[-1][1].split(" ")
                                    if (vp_word[0] in last_vp_words) & (vp_word[0] != last_vp_words[0]):
                                        c_idx = last_vp_words.index(vp_word[0])
                                        last_vp = " ".join(last_vp_words[:c_idx])
                                        temp = (
                                        vp_list[-1][0], last_vp + " " + process_hyp_words(" ".join(vp_word), hyp_words,
                                                                                          sent, -1).replace("et al .",
                                                                                                        "et al."))
                                        if temp[1] in sent:
                                            vp_list.pop()
                                            vp_list.append(temp)
                                            continue
                            save_flag = True
                            for vp in vp_list:
                                if vp[1] in oprd_str:
                                    save_flag = False
                                    break
                            if save_flag:
                                vp_list.append(("oprd", oprd_str))
        i += 1

    return vp_list, basic_elements, root_verb, root_idx


## obtain all prepositional phrases in one sentence by dependency relation
def get_prep_list_by_dependency(sent, hyp_words, spill_words_list, abbr_words, basic_elems):
    # print(sent)
    pp_list = []
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    dictionary = load_dictionary("./Dictionary.txt")
    # noun_chunks = []
    all_pos_list = [tok.pos_ for tok in doc]
    s_word = [tok.text for tok in doc]
    pp_flag = [0] * len(all_pos_list)
    ## save the word before "of"
    prep_of = get_prep_of(doc, dictionary, all_pos_list, s_word, spill_words_list)
    i = 0
    s_idx = -1
    last_s_idx = -1
    ##Traversal preposition
    for w in doc:
        if w.text == "v":
            i += 1
            continue
        if (i > 0) & (i + 1 < len(s_word)):
            if ((s_word[i - 1] in ["-", "—"]) | (s_word[i + 1] in ["-", "—"])) & (w.text in spill_words_list):
                i += 1
                continue
        vp_flag = False
        if (w.pos_ == "ADP") | (w.text == "to"):
            network = [t.text for t in list(w.children)]
            if len(network) != 0:
                pp_word = [tok.orth_ for tok in w.subtree]
                pos_list = [tok.pos_ for tok in w.subtree]
                if pp_word[-1] == "and":
                    pp_word.pop(len(pp_word) - 1)
                    pos_list.pop(len(pos_list) - 1)

                pp_str = process_hyp_words(" ".join(pp_word), hyp_words, sent, last_s_idx)
                if pp_str == "with whom":
                    i += 1
                    continue
                if pp_str in sent:
                    pp_word, pos_list, comp_f, comp_abbr = complement_pp_word(pp_word, s_word, pos_list, all_pos_list,
                                                                              abbr_words, spill_words_list)
                elif pp_str.split(",")[0] in sent:
                    c_idx = pp_word.index(",")
                    pp_word = pp_word[:c_idx]
                    pos_list = pos_list[:c_idx]
                    pp_word, pos_list, comp_f, comp_abbr = complement_pp_word(pp_word, s_word, pos_list, all_pos_list,
                                                                              abbr_words, spill_words_list)
                else:
                    comp_f = ""
                    comp_abbr = ""
                    pp_word = [w.text]
                alternative = list(pp_word)
                old_pos_list = list(pos_list)
                if w.text == "of":
                    if prep_of[0] != 'X':
                        if len(pp_list) > 0:
                            last_pp = pp_list[-1][1].replace("-", " - ").replace("–", " – ")
                            if " ".join(pp_word) in last_pp:
                                prep_of.pop(0)
                                i += 1
                                continue
                            last_pp_word = last_pp.split(" ")
                            prefix = prep_of[0].split(" ")
                            for j in range(len(prefix) - 1, -1, -1):
                                pp_word.insert(0, prefix[j])
                                pos_list.insert(0, all_pos_list[i - j])
                            if prefix[-1] == last_pp_word[-1]:
                                s_idx = check_continuity(last_pp_word, s_word, last_s_idx)
                                if s_word[s_idx + len(last_pp_word)] == "of":
                                    for i in range(len(last_pp_word)):
                                        pp_flag[s_idx + i] = 0
                                    for i in range(len(prefix)):
                                        if len(last_pp_word) > 0:
                                            last_pp_word.pop()
                                        else:
                                            break
                                    last_pp_word.extend(pp_word)
                                    last_pos_list = list(all_pos_list[s_idx:s_idx + len(last_pp_word)])
                                    last_pos_list.extend(pos_list)
                                    pp_word = last_pp_word
                                    pos_list = last_pos_list
                                    if pp_list[-1][0] == "v":
                                        vp_flag = True
                                    else:
                                        vp_flag = False
                                    pp_list.pop()
                        else:
                            prefix = prep_of[0].split(" ")
                            for j in range(len(prefix) - 1, -1, -1):
                                pp_word.insert(0, prefix[j])
                                pos_list.insert(0, all_pos_list[i - j])

                    prep_of.pop(0)
                    if (w.head.head.pos_ == "VERB") & (w.head.dep_ in ["prep", "dobj", "advcl"]):
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word,
                                                                           pos_list, all_pos_list, pp_list, hyp_words,
                                                                           sent, last_s_idx)
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                        if h_idx != -1:
                            vp_flag = True
                else:
                    ## save verb phrase
                    if (w.head.pos_ == "VERB") & (w.dep_ in ["prep", "agent"]):
                        if (pp_word[0] not in ["during", "after", "before", "via", "due"]) & (
                                "in order " not in " ".join(pp_word)):
                            pp_word, pos_list, h_idx = get_the_complete_phrase(w.text, w.head.text, s_word, pp_word,
                                                                               pos_list, all_pos_list, pp_list,
                                                                               hyp_words,
                                                                               sent, last_s_idx)
                            if pp_flag[h_idx] == 1:
                                pp_word = alternative
                            if h_idx != -1:
                                vp_flag = True
                        else:
                            if w.head.dep_ == "acl":
                                tmp_str = w.head.text + " " + w.text
                                if tmp_str in sent:
                                    v_idx = check_continuity(tmp_str.split(" "), s_word, -1)
                                    pp_word.insert(0, w.head.text)
                                    pos_list.insert(0, all_pos_list[v_idx])
                                    pp_flag[v_idx] = 0
                                    vp_flag = False
                    # if (w.head.pos_ == "ADV") & (w.head.dep_ == "advmod") & (w.head.head.pos_ == "VERB"):
                    if (w.head.pos_ == "ADV") & (w.head.dep_ == "advmod") & (w.head.head.pos_ == "VERB"):
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word,
                                                                           pos_list, all_pos_list, pp_list, hyp_words,
                                                                           sent, last_s_idx)
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                        if h_idx != -1:
                            vp_flag = True

                    ## process on Monday...in March
                    if (alternative[0] in ["on", "in"]) & (len(alternative) == 2):
                        if (alternative[1] in dictionary["on"].keys()) | (alternative[1] in dictionary["in"].keys()):
                            pp_str = " ".join(alternative)
                            pp_list.append(('p', pp_str, w.text))
                            s_idx = check_continuity(alternative, s_word, last_s_idx)
                            pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
                            i += 1
                            continue
                    if (w.text == "as") & (alternative[0] in dictionary["as"].keys()):
                        pp_word = alternative
                        pos_list = old_pos_list
                        if h_idx != -1:
                            vp_flag = False
                ## cut long prep
                flag, key = exist_pp(pos_list, pp_word, dictionary, w.text, False, spill_words_list)
                if flag & (key > 1):
                    if pp_word[key - 1] in ["and", "or"]:
                        pp_word = pp_word[:key - 1]
                    else:
                        pp_word = pp_word[:key]
                if pp_str == "of which":
                    i += 1
                    continue
                ## del ","
                if pp_word[-1] in [",", "."]:
                    pp_word.pop(len(pp_word) - 1)
                    pos_list.pop(len(pp_word) - 1)
                ##Supplement of phrase

                if pp_word[0] in ["+"]:
                    pp_word.pop(0)
                    pos_list.pop(0)

                if (pp_word[-1] in ["which", "what", "that"]) | (pp_word[0] == "v"):
                    i += 1
                    continue

                if len(comp_f) != 0:
                    pp_str = " ".join(pp_word[:-1]) + " " + comp_f
                elif len(comp_abbr) != 0:
                    pp_str = " ".join(pp_word[:-1]) + " " + comp_abbr
                else:
                    pp_str = " ".join(pp_word)

                if len(pp_str.split(" ")) == 1:
                    i += 1
                    continue

                pp_str = process_hyp_words(pp_str, hyp_words, sent, last_s_idx)
                pp_str = get_complete_last_word(pp_str.split(" "), sent.split(" "))

                if len(pp_list) > 0:
                    if pp_str in pp_list[-1]:
                        i += 1
                        continue
                    if pp_list[-1][1] in pp_str:
                        pp_list.pop()
                        last_s_idx = -1

                if pp_str in sent:
                    s_idx = check_continuity(pp_word, s_word, last_s_idx)
                    if pp_flag[s_idx] != 1:
                        if len(pp_list) > 0:
                            if pp_list[-1][1] in pp_str:
                                pp_list.pop()
                        if vp_flag:
                            pp_list.append(('v', pp_str, w.text))
                        else:
                            pp_list.append(('p', pp_str, w.text))
                        pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
                last_s_idx = s_idx

            elif w.text == "to":
                if (w.head.pos_ == "VERB") & (w.dep_ == "aux"):
                    if w.head.dep_ != "ROOT":
                        pp_word = [tok.orth_ for tok in w.head.subtree]
                        pos_list = [tok.pos_ for tok in w.head.subtree]

                    else:
                        dobj_elem = [ele for ele in basic_elems if (ele[1] == "dobj") & (w.head.text in ele[2])][0]
                        pp_word = dobj_elem[2].split()
                        pp_word.insert(0, w.text)

                    pp_str = process_hyp_words(" ".join(pp_word), hyp_words, sent, last_s_idx)
                    if pp_str in sent:
                        pp_word, pos_list, comp_f, comp_abbr = complement_pp_word(pp_word, s_word, pos_list,
                                                                                  all_pos_list,
                                                                                  abbr_words, spill_words_list)
                    elif pp_str.split(",")[0] in sent:
                        c_idx = pp_word.index(",")
                        pp_word = pp_word[:c_idx]
                        pos_list = pos_list[:c_idx]
                        pp_word, pos_list, comp_f, comp_abbr = complement_pp_word(pp_word, s_word, pos_list,
                                                                                  all_pos_list,
                                                                                  abbr_words, spill_words_list)
                    else:
                        comp_f = ""
                        comp_abbr = ""
                        pp_word = [w.text]
                    alternative = list(pp_word)
                    ## 状语从句修饰词 从句补充
                    if w.head.dep_ in ["ccomp", "xcomp", "advcl"]:
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word, pos_list, all_pos_list, pp_list,
                                                                           hyp_words, sent, last_s_idx)
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                        vp_flag = True
                    ## cut long prep
                    flag, key = exist_pp(pos_list, pp_word, dictionary, w.text, True, spill_words_list)
                    if flag & (key > 1):
                        pp_word = pp_word[:key]
                        pp_str = devide_pp(pp_word, key)
                    else:
                        if len(comp_f) != 0:
                            pp_str = " ".join(pp_word[:-1]) + " " + comp_f
                        elif len(comp_abbr) != 0:
                            pp_str = " ".join(pp_word[:-1]) + " " + comp_abbr
                        else:
                            pp_str = " ".join(pp_word)
                    pp_str = process_hyp_words(pp_str, hyp_words, sent, last_s_idx)
                    if (len(pp_str.split(" ")) == 1) | (pp_str.split(" ")[0] == "v") | (pp_str.split(" ")[-1] == "to"):
                        i += 1
                        continue
                    pp_str = get_complete_last_word(pp_str.split(" "), sent.split(" "))
                    if len(pp_list) > 0:
                        if pp_list[-1][1] in pp_str:
                            pp_list.pop()
                            last_s_idx = -1
                    if pp_str in sent:
                        s_idx = check_continuity(pp_word, s_word, last_s_idx)
                        if s_word[s_idx - 1] in dictionary["to"].keys():
                            pp_str = " ".join(s_word[s_idx - 2:s_idx]) + " " + pp_str
                            for i in range(s_idx - 2, s_idx):
                                pp_flag[i] = 0
                            s_idx = s_idx - 2
                            if len(pp_list) > 0:
                                if pp_list[-1][1] in pp_str:
                                    pp_list.pop()

                        if pp_flag[s_idx] != 1:
                            if len(pp_list) > 0:
                                if pp_list[-1][1] in pp_str:
                                    pp_list.pop()
                            if vp_flag:
                                pp_list.append(('v', pp_str, w.text))
                            else:
                                pp_list.append(('p', pp_str, w.text))
                            pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
                last_s_idx = s_idx
        i += 1

    return pp_list


def get_res_by_label(words, comp_label):
    res_words = []
    for i in range(len(words)):
        if comp_label[i] == 1:
            res_words.append(words[i])
    comp_res = " ".join(res_words)
    # print("final result: ", comp_res)
    return comp_res


def check_vp_integrity(res_label, cut_words, vp_list, vp_flag):
    s_idx = -1
    for vp in vp_list:
        s_idx = vp_flag.index(2, s_idx + 1)
        vp_words = vp[1].split()
        if vp[0] in ["acomp", "auxpass"]:
            if res_label[s_idx:s_idx + len(vp_words)].count(1) > 0:
                maintain = True
            else:
                maintain = False
        else:
            if res_label[s_idx] == 1:
                maintain = True
            else:
                maintain = False

        for i in range(s_idx, s_idx + len(vp_words)):
            if cut_words[i] == ",":
                maintain = False
            if maintain:
                res_label[i] = 1
            else:
                res_label[i] = 0

    return res_label


## Check the integrity of prepositional phrases in the compression results
def check_pp_integrity(words, res_label, orig_pp, pp_flag, ner_list, sbar_list):
    s_idx = -1
    last_maintain = True
    last_sidx = -1
    first_verb_pp = -1
    first_comma = -1
    if "," in words:
        first_comma = words.index(",")
    for i in range(len(orig_pp)):
        if orig_pp[i][0] == "v":
            first_verb_pp = i
            break
    for i in range(len(orig_pp)):
        sbar_flag = False
        for sbar in sbar_list:
            if sbar[1] in orig_pp[i][1]:
                sbar_flag = True
                break
        s_idx = pp_flag.index(2, s_idx + 1)
        if (res_label[s_idx] == 1) & (words[s_idx - 1] != ":"):
            maintain = True
        else:
            maintain = False
        pp_words = orig_pp[i][1].split(" ")
        pp_len = len(pp_words)
        if not maintain:
            maintain_flag = True
            count = res_label[s_idx:s_idx + pp_len].count(1)
            if maintain_flag:
                if count > pp_len / 2:
                    res_label[s_idx] = 1
                    maintain = True
            ## of位于重要内容部分
            sym_count = words[:s_idx].count(":") + words[:s_idx].count(",") + words[:s_idx].count("–") \
                        + words[:s_idx].count("and") + words[:s_idx].count("or")

            ## how to find "of" to be subject
            if ("of" in pp_words) & (not maintain):
                if (count != 0) & ((sym_count == 0) | (res_label[:s_idx].count(1) == 0)):
                    maintain = True
                else:
                    maintain = False
            if (i == first_verb_pp) & (count != 0):
                maintain = True
            if s_idx > 2:
                last_comma = s_idx - 1
                while words[last_comma] != ",":
                    last_comma -= 1
                    if last_comma < 0:
                        break
                if last_comma != first_comma:
                    one_count = res_label[last_comma + 1:s_idx].count(1)
                    if (one_count == 0) & (s_idx != last_comma + 1) & (i != first_verb_pp):
                        maintain = False
        else:
            if "and" in words[:s_idx]:
                and_idx = s_idx - 1
                while words[and_idx] != "and":
                    and_idx -= 1
                if not ("," in words[and_idx:s_idx]):
                    if res_label[and_idx:s_idx].count == 0:
                        maintain = False

        for j in range(s_idx, s_idx + pp_len):
            if ((words[j] == ",") & (j != first_comma) & ("as" not in pp_words)) | (words[j] == ":"):
                maintain = False
            if maintain:
                if (res_label[j] != -1) | sbar_flag:
                    res_label[j] = 1
            else:
                res_label[j] = 0

        if maintain & (s_idx > 1) & (s_idx + pp_len < len(words)):
            for n in ner_list:
                n_words = n.split(" ")
                if (words[s_idx] == n_words[-1]) & (words[s_idx - 1] == n_words[-2]):
                    n_idx = s_idx - 1
                    for j in range(len(n_words) - 1):
                        res_label[n_idx] = 1
                        n_idx -= 1
                if (words[s_idx + pp_len - 1] == n_words[0]) & (words[s_idx + pp_len] == n_words[1]):
                    n_idx = s_idx + 1
                    for j in range(len(n_words) - 1):
                        res_label[n_idx] = 1
                        n_idx += 1
        if i > 1:
            count = res_label[last_sidx + len(orig_pp[i - 1][1].split(" ")):s_idx].count(1)
            temp = (s_idx - last_sidx - len(orig_pp[i - 1][1].split(" "))) / 2
            if count < temp:
                if (not maintain) & (not last_maintain):
                    for j in range(last_sidx + len(orig_pp[i - 1][1].split(" ")), s_idx):
                        res_label[j] = 0
            else:
                if last_maintain & maintain:
                    for j in range(last_sidx + len(orig_pp[i - 1][1].split(" ")), s_idx):
                        res_label[j] = 1
        last_maintain = maintain
        last_sidx = s_idx
    return res_label


def check_symbols_integrity(res_label, sym_list, sem_flag, pp_list, sbar_list, s_words, root_idx):
    s_idx = -1
    new_res_label = list(res_label)
    max_value = 0
    maintain_idx = -1
    for i in range(0, len(sym_list)):
        s_idx = sem_flag.index(2, s_idx + 1)
        e_idx = s_idx + len(sym_list[i].split(" "))
        one_count = res_label[s_idx:e_idx].count(1)
        if one_count > max_value:
            if i == 0:
                if (root_idx > s_idx) & (root_idx < e_idx):
                    max_value = one_count
                    maintain_idx = i
            else:
                if max_value == 0:
                    max_value = one_count
                    maintain_idx = i
    s_idx = -1
    for i in range(0, len(sym_list)):
        s_idx = sem_flag.index(2, s_idx + 1)
        sym_words = sym_list[i].split(" ")
        if (i == maintain_idx) & (max_value < len(sym_words)/2):
            for j in range(s_idx, s_idx + len(sym_words)):
                new_res_label[j] = 1
            for sbar in sbar_list:
                if (sbar[1] in sym_list[i]) & (sbar[0] == "s"):
                    sbar_idx = check_continuity(sbar[1].split(" "), s_words, -1)
                    for j in range(sbar_idx, sbar_idx + len(sbar[1].split(" "))):
                        new_res_label[j] = 0
            for pp in pp_list:
                if (pp[1] in sym_list[i]) & (pp[0] == "p"):
                    pp_idx = check_continuity(pp[1].split(" "), s_words, -1)
                    for j in range(pp_idx, pp_idx + len(pp[1].split(" "))):
                        new_res_label[j] = 0
        if i != maintain_idx:
            if s_idx - 1 > 0:
                s_idx -= 1
            if ('—' in s_words) | ('–' in s_words):
                if i != 0:
                    for j in range(s_idx, s_idx + len(sym_words)):
                        new_res_label[j] = 0
            else:
                for j in range(s_idx, s_idx + len(sym_words)):
                    new_res_label[j] = 0

    if new_res_label.count(1) == 1:
        s_idx = check_continuity(sym_list[0], s_words, -1)
        e_idx = s_idx + len(sym_list[0].split(" "))
        for j in range(s_idx, e_idx):
            new_res_label[j] = 1
        for sbar in sbar_list:
            if (sbar[1] in sym_list[0]) & (sbar[0] == "s"):
                sbar_idx = check_continuity(sbar[1].split(" "), s_words, -1)
                for j in range(sbar_idx, sbar_idx + len(sbar[1].split(" "))):
                    new_res_label[j] = 0
        for pp in pp_list:
            if (pp[1] in sym_list[0]) & (pp[0] == "p"):
                pp_idx = check_continuity(pp[1].split(" "), s_words, -1)
                for j in range(pp_idx, pp_idx + len(pp[1].split(" "))):
                    new_res_label[j] = 0


    return new_res_label, maintain_idx


## Check the integrity of sbar in the compression results
def check_sbar_integrity(res_label, sbar_list, sbar_flag, cut_words, pp_list, basic_elements):
    s_idx = -1
    for i in range(len(sbar_list)):
        sbar_type = sbar_list[i][0]
        sbar = sbar_list[i][1]
        s_idx = sbar_flag.index(2, s_idx + 1)
        sbar_len = len(sbar.split(" "))
        if sbar_len == 1:
            del_flag = True
            for j in range(s_idx):
                if res_label[j] != 0:
                    del_flag = False
                    break
            if del_flag:
                res_label[s_idx] = -1
                continue
        else:
            symbols = cut_words[s_idx: s_idx + sbar_len].count("\"") + cut_words[s_idx: s_idx + sbar_len].count(",")
            count = res_label[s_idx: s_idx + sbar_len].count(1)
            pp_len = 0
            for pp in pp_list:
                if (pp[0] == "p") & (pp[1] in sbar):
                    pp_len += len(pp[1].split(" "))
            if (sbar_type == "t") & (count > 0):
                if "that" in sbar:
                    t_idx = s_idx + sbar.split(" ").index("that")
                else:
                    t_idx = s_idx + len(sbar.split(" "))
                if (sbar.split(" ")[0] != "that") & (res_label[:t_idx].count(1) == 0):
                    for j in range(s_idx, t_idx + 1):
                        res_label[j] = -1
                    for j in range(t_idx + 1, s_idx + sbar_len):
                        res_label[j] = 1
                else:
                    for j in range(s_idx, s_idx + sbar_len):
                        res_label[j] = 1
            else:
                e_idx = s_idx + sbar_len
                if count >= (sbar_len - symbols) / 2:
                    for ele in basic_elements:
                        if ("obj" in ele[1]) | ("attr" in ele[1]):
                            if ele[2] in " ".join(cut_words[:s_idx]):
                                obj_idx = check_continuity(ele[2].split(" "), cut_words, -1)
                                if obj_idx + len(ele[2].split(" ")) == s_idx:
                                    s_idx = obj_idx
                    for j in range(s_idx, e_idx):
                        res_label[j] = 1
                else:
                    if count < (sbar_len - symbols - pp_len) / 2:
                        for j in range(s_idx, e_idx):
                            res_label[j] = -1
    return res_label


def check_formulation_intergrity(res_label, for_list, for_flag):
    s_idx = -1
    for i in range(len(for_list)):
        s_idx = for_flag.index(2, s_idx + 1)
        for_len = len(for_list[i].split(" "))
        count = res_label[s_idx: s_idx + for_len].count(1)
        for j in range(s_idx, s_idx + for_len):
            if (count > for_len / 2) & (res_label[j] != -1):
                res_label[j] = 1
            else:
                res_label[j] = 0

    return res_label


def check_ner_intergrity(res_label, ner_list, ner_flag):
    s_idx = -1
    for i in range(len(ner_list)):
        s_idx = ner_flag.index(2, s_idx + 1)
        ner_len = len(ner_list[i].split(" "))
        maintain = False
        count = res_label[s_idx:s_idx + ner_len].count(1)
        if count > 0:
            maintain = True

        for j in range(s_idx, s_idx + ner_len):
            if maintain:
                res_label[j] = 1
            else:
                res_label[j] = 0

    return res_label


def exist_sbar(nlp_tree):
    count = 0
    for s in nlp_tree.subtrees():
        pos_list = s.pos()
        if s.label() == "SBAR":
            count += 1
        if (s.label() == "PP") & (pos_list[0][0] in ["while", "when"]):
            count += 1
        if count == 2:
            break
    if count >= 2:
        return True
    else:
        return False


def devide_sbar(pos_list, long_sbar, nlp_tree, hyp_words, orig_sent):
    count = 0
    for s in nlp_tree.subtrees():
        if s.label() == "SBAR":
            count += 1
        elif (s.label() == "PP") & (s.leaves()[0] in ["while", "when"]):
            count += 1
        if count == 2:
            sub_string = process_hyp_words(" ".join(s.leaves()), hyp_words, orig_sent, -1)
            sbar_word = long_sbar.split(sub_string)[0].strip().rstrip().split(" ")
            if len(sbar_word) == 1:
                long_sbar = process_hyp_words(long_sbar, hyp_words, orig_sent, -1)
                sbar = long_sbar.strip().rstrip().split(" , ")[0]
                return sbar
            else:
                break
    # sbar = ""
    # if pos_list[0][1] in ['IN', 'WDT', 'WP', 'WRB', "WP$"]:
    sbar = process_hyp_words(long_sbar.split(sub_string)[0], hyp_words, orig_sent, -1).strip().rstrip()
    if sbar == "that":
        sbar = sbar + " " + sub_string
    sbar = sbar.split(" , ")[0]

    if len(sbar.split(" ")) <= 3:
        sbar = long_sbar

    return sbar


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
            if len(phrase) > 2:
                if (words[idx + 1] == phrase[1]) & (words[idx + 2] == phrase[2]):
                    return idx
            else:
                if words[idx + 1] == phrase[1]:
                    return idx
                else:
                    count += 1
                    s_idx = idx


def judge_that_in_start(s_words, sbar_idx, pp_list):
    if "," in s_words[0:sbar_idx]:
        common_idx = s_words[0:sbar_idx].index(",")
        start_idx = common_idx + 1
    else:
        start_idx = 0
    remain_w = sbar_idx - start_idx
    before_sbar = " ".join(s_words[start_idx:sbar_idx])
    new_pp_list = list(pp_list)
    for i in range(len(pp_list)):
        p = pp_list[i]
        if (p[1] in before_sbar) & (p[0] == "p"):
            len_p = len(p[1].split(" "))
            remain_w = remain_w - len_p
            new_pp_list.pop(new_pp_list.index(p))
    return remain_w, start_idx, new_pp_list


def check_that_clause(s_words, sbar_words, pos_list, dictionary, pp_list, hyp_words, orig_sent):
    # s_idx = get_phrase_idx(s_words, sbar)
    s_idx = check_continuity(sbar_words, s_words, -1)
    remain_w, start_idx, new_pp_list = judge_that_in_start(s_words, s_idx, pp_list)
    if pos_list[s_idx][0] != sbar_words[0]:
        count = 0
        for w_tup in hyp_words:
            w = w_tup[1]
            if w in " ".join(s_words[:s_idx]):
                count += 1
        p_s_idx = count * 2 + s_idx
    else:
        p_s_idx = s_idx
    if pos_list[p_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        if remain_w <= 6:
            # sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
            sbar = ("t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    if (pos_list[p_s_idx - 2][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (pos_list[p_s_idx - 1][0] in ['"']):
        if remain_w <= 6:
            # sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
            sbar = ("t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif (pos_list[p_s_idx - 1][1] in ['JJ']) & (
            pos_list[p_s_idx - 2][0] in ["is", "are", "am", "been", "'s", "'re", "be", "'m"]):
        if remain_w <= 6:
            # sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
            sbar = ("t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(s_words[s_idx - 2] + " " + s_words[s_idx - 1] + " " + " ".join(sbar_words),
                                           hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif (pos_list[p_s_idx - 1][0] in dictionary['that'].keys()) & (pos_list[p_s_idx][0] not in dictionary['that'].keys()):
        if remain_w <= 6:
            # sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
            sbar = ("t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif pos_list[p_s_idx][0] in dictionary['that'].keys():
        if remain_w <= 6:
            # sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
            sbar = ("t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(" ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    else:
        sbar = ("s", " ".join(sbar_words))
        return False, sbar, pp_list


def check_comma(words, res_label):
    comma_flag = False
    start_flag = False
    count = 0
    for i in range(len(words)):
        if res_label[i] == 1:
            count += 1
        if res_label[i] == 1:
            start_flag = True
        if (words[i] == ",") & start_flag:
            if count > 3:
                comma_flag = True
        if comma_flag & (res_label[i] == 1) & (words[i] != "."):
            res_label[i] = 0

    return res_label


def inter(a, b):
    return list(set(a) & set(b))


def process_wrong_formulation(sbar):
    sbar = sbar.replace("7n2 15n 40", "7n2 + 15n + 40")
    sbar = sbar.replace("( n + 1 ) 2", "(n + 1)2")
    sbar = sbar.replace("( p − 1 ) ! 1", "(p − 1)! + 1")
    sbar = sbar.replace("1 m · s − 2", "1 m·s−2")
    sbar = sbar.replace("np ≡ n", "np≡n")
    sbar = sbar.replace("[ 256kn 1, 256 k ( n + 1) − 1 ]", "[256kn + 1, 256k(n + 1) − 1]")
    sbar = sbar.replace("Sky + HD", "Sky+HD")
    sbar = sbar.replace("Sky +", "Sky+")
    sbar = sbar.replace("CD4 +", "CD4+")
    sbar = sbar.replace("CD8 +", "CD8+")
    sbar = sbar.replace("− F", "−F")
    sbar = sbar.replace("73 / 173 / EEC", "73/173/EEC")
    sbar = sbar.replace("- i -", "-i-")
    sbar = sbar.replace("HDLC / LAPD / LAPB", "HDLC/LAPD/LAPB")
    # sbar = sbar.replace(" ° ", "° ")
    sbar = sbar.replace("° F", "°F")
    sbar = sbar.replace("° C", "°C")
    sbar = sbar.replace(" ° E ", "°E ")
    sbar = sbar.replace(" ° ", "° ")
    sbar = sbar.replace("£ ", "£")
    sbar = sbar.replace("€ ", "€")
    sbar = sbar.replace("§ ", "§")
    sbar = sbar.replace("X. 25", "X.25")
    sbar = sbar.replace(" )", ")")
    if sbar[-3:] == "° E":
        sbar = sbar[:-3] + "°E"
    return sbar


def process_hyp_words(sent, hyp_words, orig_sent, search_s_idx):
    wrong_hyp_words = []
    word_list = []
    for w_tup in hyp_words:
        w = w_tup[1]
        if w[0] in ["-", "–", "−"]:
            w = w.replace("-", "- ").replace("–", "– ").replace("−", "− ")
        elif w[-1] in ["-", "–", "−"]:
            w = w.replace("-", " -").replace("–", " –").replace("−", " −")
        else:
            w = w.replace("-", " - ").replace("–", " – ").replace("−", " − ")
        word_list.append(w.split())
        wrong_hyp_words.append(w)

    for i in range(len(hyp_words)):
        sent = sent.replace(wrong_hyp_words[i], hyp_words[i][1])
        idx = hyp_words[i][0]
        if (len(sent.split(" ")) >= 2) & (idx > search_s_idx):
            if (sent.split(" ")[-1] in ["-", "–"]) & (sent.split(" ")[-2] in word_list[i]):
                sent = " ".join(sent.split(" ")[:-2]) + " " + hyp_words[i][1]
            elif (sent.split(" ")[-1] in word_list[i]) & (sent.split(" ")[-2] == orig_sent.split(" ")[idx - 1]):
                if sent.split(" ")[-1] == word_list[i][0]:
                    sent = " ".join(sent.split(" ")[:-1]) + " " + hyp_words[i][1]

    sent = process_wrong_formulation(sent)
    if (" / " in sent) & (" / " not in orig_sent):
        sent = sent.replace(" / ", "/")

    if (" +" in sent) & (" +" not in orig_sent):
        sent = sent.replace(" +", "+")

    if (" m " in sent) & (" m " not in orig_sent):
        sent = sent.replace(" m ", "m ")

    if (sent[-2:] == " m") & (" m " not in orig_sent):
        sent = sent[:-2] + "m"

    return sent


def format_tree_sent(key_words, hyp_words, sent, sent_words, last_s_idx):
    key_sent = " ".join(key_words).replace("-LRB-", "(").replace("-RRB-", ")")
    ## uncomplete formulation should not in the starting index
    for i in range(len(key_formulations)):
        if key_formulations[i] in key_sent:
            if formulations[i][0] == "(":
                key_sent = key_sent.replace(key_formulations[i], " " + formulations[i])
            else:
                key_sent = key_sent.replace(key_formulations[i], formulations[i])
    key_sent = key_sent.replace("( ", "(").replace(" )", ")")
    key_sent = process_hyp_words(key_sent, hyp_words, sent, last_s_idx)
    key_sent = get_complete_last_word(key_sent.split(" "), sent.split(" "))
    key_words = key_sent.split(" ")
    orig_s_idx = -1
    if len(key_words) > 1:
        orig_s_idx = check_continuity(key_words[:-1], sent_words, -1)
        for f in formulations:
            f_w = f.split(" ")
            if f_w[0] == sent_words[orig_s_idx + len(key_words) - 1]:
                if len(f_w) == 1:
                    key_words[-1] = f_w[0]
                else:
                    if f_w[1] == sent_words[orig_s_idx + len(key_words)]:
                        key_words.extend(f_w[1:])
                break
    return key_sent, key_words, orig_s_idx


def extra_sbar(sent, hyp_words):
    sbar_list = []
    nlp_tree = get_nlp_tree(sent)
    cc_sent_list, np_sbar_list, np_pp_list = extract_sent_np(nlp_tree, sent, hyp_words)
    tree_words = nlp_tree.leaves()
    all_pos_list = nlp_tree.pos()
    sent_words = sent.split(" ")
    orig_s_idx = -1
    for s in nlp_tree.subtrees():
        label = s.label()
        pos_list = s.pos()
        if (label == "SBAR") | ((label == "PP") & (pos_list[0][0] in ["while", "when"])):
            if (s.leaves()[0] in [">", "<", "can", "Bolinopsis", "I"]) | (len(s.leaves()) == 1):
                continue
            key_words = s.leaves()
            tree_s_idx = check_continuity(key_words, tree_words, -1)
            key_sent, key_words, orig_s_idx = format_tree_sent(key_words, hyp_words, sent, sent_words, orig_s_idx)
            ### 's
            if len(key_words) == 1:
                continue
            if orig_s_idx > 0:
                if sent_words[orig_s_idx - 1].lower() in ["there", "here"]:
                    continue
            if orig_s_idx >= 2:
                if key_words[0] == "as":
                    if sent_words[orig_s_idx - 2] == "as":
                        key_words = sent_words[orig_s_idx - 2:orig_s_idx] + key_words
                        pos_list = all_pos_list[tree_s_idx - 2:tree_s_idx] + pos_list
            if not exist_sbar(s):
                if (pos_list[0][1] in ["IN", "WDT", "WP", "WP$", "WRB", 'JJ', "PRP", "TO"]) | (
                        (pos_list[0][0] == "as") & (pos_list[0][1] == "RB")):
                    ## including what ...
                    if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (
                            pos_list[0][0] == "what"):
                        key_words.insert(0, all_pos_list[tree_s_idx - 1][0])

                    orig_s_idx = check_continuity(key_words, sent_words, -1)
                    for hw in hyp_words:
                        if orig_s_idx + len(key_words) - 1 == hw[0]:
                            key_words[-1] = hw[1]
                            break
                    sbar = process_hyp_words(" ".join(key_words), hyp_words, sent, orig_s_idx)
                    sbar = process_wrong_formulation(sbar)
                    if len(sbar_list) > 0:
                        if sbar not in sbar_list[-1]:
                            sbar_list.append(sbar)
                    else:
                        sbar_list.append(sbar)
            else:
                long_sbar = process_hyp_words(" ".join(key_words), hyp_words, sent, orig_s_idx)
                if " as to " not in long_sbar:
                    sbar = devide_sbar(pos_list, long_sbar, s, hyp_words, sent)
                else:
                    sbar = long_sbar
                if len(sbar) > 1:
                    if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (
                            pos_list[0][0] == "what"):
                        sbar = all_pos_list[tree_s_idx - 1][0] + " " + sbar
                    sbar = process_wrong_formulation(sbar)
                    if len(sbar_list) > 0:
                        if sbar not in sbar_list[-1]:
                            sbar_list.append(sbar)
                    else:
                        sbar_list.append(sbar)

    return sbar_list, all_pos_list, cc_sent_list, np_sbar_list, np_pp_list


def using_pp_update_sbar(sent, sbar_list, all_pos_list, dictionary, pp_list, hyp_words):
    for i in range(len(sbar_list)):
        sbar = process_hyp_words(sbar_list[i], hyp_words, sent, -1)
        sbar_words = sbar.split(" ")
        if (sbar_words[0] == "that") | (sbar_words[0] in dictionary['that'].keys()):
            update_flag, sbar, pp_list = check_that_clause(sent.split(" "),
                                                           sbar_words, all_pos_list,
                                                           dictionary, pp_list, hyp_words, sent)
            sbar_list[i] = sbar
        else:
            sbar_list[i] = ("s", sbar)

    for i in range(len(sbar_list) - 1):
        if sbar_list[i][1] in sbar_list[i + 1][1]:
            sbar_list.pop(i)
            break

    return sbar_list, pp_list


def filter_pp_in_sbar(sbar_list, pp_list):
    if len(sbar_list) > 0:
        res_pp = list(pp_list)
        for pp in pp_list:
            for sbar in sbar_list:
                if pp[1] in sbar[1]:
                    if pp in res_pp:
                        res_pp.remove(pp)
                else:
                    if " ".join(pp[1].split(" ")[-2:]).istitle():
                        if " ".join(pp[1].split(" ")[:-1]) in sbar[1]:
                            if pp in res_pp:
                                res_pp.remove(pp)
        return res_pp
    else:
        return pp_list


def group_by_level(tree):
    tree_positions = {}
    for p in tree.treepositions():
        if len(p) not in tree_positions.keys():
            tree_positions[len(p)] = []
        tree_positions[len(p)].append(p)
    position_labels = {}
    for key in tree_positions.keys():
        tree_list = tree_positions[key]
        for t in tree_list:
            if not isinstance(tree[t], str):
                if key not in position_labels.keys():
                    position_labels[key] = []
                position_labels[key].append(tree[t].label())
            else:
                if key not in position_labels.keys():
                    position_labels[key] = []
                position_labels[key].append(tree[t])
    return tree_positions, position_labels


def extract_sent_np(tree, sent, hyp_words):
    sent_list = []
    np_sbar_list = []
    np_pp_list = []
    sent_words = sent.split(" ")
    tree_positions, position_labels = group_by_level(tree)
    position_labels = sorted(position_labels.items(), key=lambda x: x[0], reverse=False)
    for item in position_labels:
        key = item[0]
        if "CC S" in " ".join(item[1]):
            tree_list = tree_positions[key]
            for i in range(len(item[1])):
                if item[1][i] == "S":
                    s_tree = tree[tree_list[i]]
                    s_words = s_tree.leaves()
                    if i > 0:
                        if item[1][i - 1] == "CC":
                            s_words = list(tree[tree_list[i - 1]].leaves()) + s_words
                    if len(s_words) < 2:
                        continue
                    key_sent, key_words, orig_s_idx = format_tree_sent(s_words, hyp_words, sent, sent_words, -1)
                    if len(sent_list) > 0:
                        if " ".join(key_words) in sent_list[-1]:
                            continue
                    if len(key_words) < 2:
                        continue
                    key_sent = " ".join(key_words)
                    sent_list.append(key_sent)
        if ("NP SBAR" in " ".join(item[1])) | (("NP , SBAR" in " ".join(item[1]))):
            tree_list = tree_positions[key - 1]
            for i in range(len(position_labels[key - 1][1])):
                if position_labels[key - 1][1][i] == "NP":
                    np_tree = tree[tree_list[i]]
                    if not isinstance(np_tree, str):
                        np_tree_positions, np_position_labels = group_by_level(np_tree)
                        if " ".join(np_position_labels[1]) in ["NP SBAR", "NP , SBAR"]:
                            key_sent, key_words, orig_s_idx = format_tree_sent(np_tree.leaves(), hyp_words, sent, sent_words, -1)
                            if len(key_words) < 3:
                                continue
                            if len(np_sbar_list) > 0:
                                if key_sent in np_sbar_list[-1]:
                                    continue
                            #print("original: ", sent)
                            np_sbar_list.append(key_sent)
                            #print("np sbar: ", key_sent)
                            break
                    else:
                        print("Not match condition")
        if "NP PP" in " ".join(item[1]):
            tree_list = tree_positions[key - 1]
            for i in range(len(position_labels[key - 1][1])):
                if position_labels[key - 1][1][i] == "NP":
                    np_tree = tree[tree_list[i]]
                    if not isinstance(np_tree, str):
                        np_tree_positions, np_position_labels = group_by_level(np_tree)
                        if " ".join(np_position_labels[1]) in ["NP PP"]:
                            key_words = np_tree.leaves()
                            key_sent, key_words, orig_s_idx = format_tree_sent(key_words, hyp_words, sent, sent_words, -1)
                            if len(np_pp_list) > 0:
                                if key_sent in np_pp_list[-1]:
                                    continue
                            #print("original: ", sent)
                            np_pp_list.append(key_sent)
                            #print("np pp: ", key_sent)
                            break
                    else:
                        print("Not match condition")
    return sent_list, np_sbar_list, np_pp_list

def extract_conj(text):
    #print("conj_str:", text)
    res = []
    # as well as单独处理
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(text)
    ans = []
    min = 0
    j = 0
    while j < len(doc):
        if doc[j].dep_ == 'preconj':
            j = two_conj(j, doc, ans)
            min = j + 1
            if not len(ans) == 0:
                res.append(ans)
                ans = []
            continue
        elif doc[j].dep_ == 'conj':
            j = single_conj(min, j, doc, ans)
            min = j
            if not len(ans) == 0:
                s_temp = ans[0].split(" ")
                for i in range(1, len(s_temp) - 1):
                    if s_temp[i] == '-' and "".join(s_temp[i - 1:i + 2]) in text:
                        ans[0] = " ".join(s_temp[0:i - 1]) + " " + "".join(s_temp[i - 1:i + 2]) + " " + " ".join(
                            s_temp[i + 2:len(s_temp)])
                for i in range(len(s_temp)):
                    if len(s_temp) >= 5 and s_temp[len(s_temp) - 2] == 'and' and "," in s_temp:
                        if s_temp[len(s_temp) - 1].endswith("s") and s_temp[len(s_temp) - 3].endswith("s"):
                            if not s_temp[s_temp.index(',') - 1].endswith("s"):
                                ans[0] = " ".join(s_temp[s_temp.index(',') + 1:])
                if ans[0] == 'had no choice but , and gradually lost':
                    ans[0] = 'had no choice and lost'
                if ans[0] == 'Sophomore , junior , and senior undergraduates':
                    ans[0] = 'Sophomore , junior , and undergraduates'
                res.append(ans)
                ans = []
        else:
            pass
        j += 1
    return res


def valid(start, end, doc, j):
    if j != end:
        for i in doc[j + 1:end]:
            if i.text == "and" or i.text == "or":
                return 1
    else:
        for i in doc[start + 1:j]:
            if i.text == "and" or i.text == "or":
                return 1
    return 0


def single_conj(min, j, doc, ans):
    flag = 0
    str = ''
    choose_flag = 0
    # 找后面还有没有
    end = j
    start = 0
    if j < len(doc) - 1:
        for i in range(len(doc) - 1, j, -1):
            if doc[i].head == doc[j] and doc[i].dep_ == 'conj':
                end = i
                break

    # 找前面
    for i in range(min, j + 1):
        if doc[i] == doc[j].head:
            flag == 1
            start = i
            choose_flag = valid(start, end, doc, j)
    if choose_flag:
        str = ' '.join([x.text for x in doc[start:end + 1]])
        # print(str)
        ans.append(str.strip())
        ans.append(1)
    return end


def two_conj(j, doc, ans):
    str = ''
    end = 0
    for i in range(j, len(doc)):
        str += ' ' + doc[i].text
        if doc[i].dep_ == 'conj' and doc[i].head == doc[j].head:
            end = i
            break
    if not str.isspace():
        ans.append(str.strip())
        ans.append(2)
    return i + 1


def write_list_in_txt(orig_sents, comp_list, orig_comp, file_path):
    f = open(file_path, "w", encoding="utf-8")
    for i in range(len(comp_list)):
        f.write("i = " + str(i) + "\n")
        f.write(orig_sents[i] + "\n")
        f.write("orig: " + orig_comp[i] + "\n")
        f.write("modifiy: " + comp_list[i] + "\n")
        f.write("\n")


def juede_word_is_formulation(word):
    if (len(word) == 1) & (word not in [".", ":"]):
        return True
    elif word.replace(".", "").isdigit():
        return True
    elif word.replace("/", "").isdigit():
        return True
    elif word in ["<", ">", "+", "−", "·", "T(n)", "f(n)", "...", "(n", "(p", "bi", "xO"]:
        return True
    elif len(re.sub("\D", "", word)) != 0:
        return True
    elif word.isupper():
        return True
    elif ("(" in word) | ("log" in word) | (")" in word):
        return True
    else:
        return False


def extra_formulation(cut_sent):
    for_list = []
    cut_words = cut_sent.split(" ")
    for i in range(len(cut_words)):
        if (cut_words[i] in ["=", "<", ">", "+", "−"]) | \
                (("/" in cut_words[i]) & (juede_word_is_formulation(cut_words[i])) & (len(cut_words[i]) != 1)) \
                | (("(n" in cut_words[i]) & (juede_word_is_formulation(cut_words[i]))) \
                | ("m·s−2" in cut_words[i]):
            j = i - 1
            while j >= 0:
                if juede_word_is_formulation(cut_words[j]):
                    j = j - 1
                else:
                    break
            s_idx = j + 1
            j = i + 1
            while j < len(cut_words):
                if juede_word_is_formulation(cut_words[j]):
                    j = j + 1
                else:
                    break
            e_idx = j - 1
            formulation = " ".join(cut_words[s_idx:e_idx + 1])
            if formulation[-1] == ",":
                formulation = formulation[:-1]
            if formulation[0] == ",":
                formulation = formulation[1:]
            if formulation.strip().rstrip() not in for_list:
                for_list.append(formulation.strip().rstrip())

    return for_list


def str_in_list(s, li):
    for i in li:
        if s in i:
            return False
    return True


def list_in_str(s, li):
    new_list = list(li)
    for item in li:
        if item in s:
            new_list.remove(item)
    return new_list


def correct_hyp_word_in_ner(hyp_words, n_w, sent_words, last_s_idx):
    if (n_w[-1] in ["-", "–", "−"]) & (len(n_w) == 2):
        n_w = [n_w[0] + n_w[1]]
    new_ner = " ".join(n_w)
    for w_tup in hyp_words:
        w = w_tup[1]
        tmp_n_w = list(n_w)
        tmp_n_w[0] = w
        if check_continuity(tmp_n_w, sent_words, last_s_idx) != -1:
            new_ner = " ".join(tmp_n_w)
            break
        tmp_n_w = list(n_w)
        tmp_n_w[-1] = w
        if check_continuity(tmp_n_w, sent_words, last_s_idx) != -1:
            new_ner = " ".join(tmp_n_w)
            break

    n_w = new_ner.split(" ")
    s_idx = check_continuity(n_w, sent_words, last_s_idx)
    if s_idx == -1:
        s_idx = check_continuity(n_w[:-1], sent_words, last_s_idx)
        if s_idx != -1:
            e_idx = s_idx + len(n_w) - 1
            if (sent_words[e_idx] != n_w[-1]) & (n_w[-1] in sent_words[e_idx]):
                new_ner = " ".join(n_w[:-1]) + " " + sent_words[e_idx]
        else:
            tmp_n_w = list(n_w)
            for w_tup in hyp_words:
                w = w_tup[1]
                w_w = re.split('-|–|−', w)
                if tmp_n_w[0] == w_w[-1]:
                    tmp_n_w[0] = w
                    continue
                if tmp_n_w[-1] == w_w[0]:
                    tmp_n_w[-1] = w
                    continue
            new_ner = " ".join(tmp_n_w)
            s_idx = check_continuity(tmp_n_w, sent_words, last_s_idx)

    return new_ner, s_idx


def supplement_ner_list(ner_list, alpha_ner_list, s_word):
    new_ner_list = []
    alpha_ner_idx = 0
    new_ner = ""
    new_ner_words = []
    if len(alpha_ner_list) > 0:
        for i in range(len(ner_list)):
            ner_words = ner_list[i].split()
            upper_words = [w for w in ner_words if w[0].isupper()]
            new_ner_words.extend(upper_words)
            if (set(new_ner_words).issubset(set(alpha_ner_list[alpha_ner_idx].split(" ")))) \
                    & (len(new_ner_words) <= len(alpha_ner_list[alpha_ner_idx].split(" "))) \
                    & (len(new_ner_words) != 0) & (len(upper_words) != 0):
                new_ner = new_ner + " " + ner_list[i]
            else:
                if new_ner != "":
                    new_ner_list.append(new_ner.strip().rstrip())
                while (not set(ner_words).issubset(set(alpha_ner_list[alpha_ner_idx].split(" ")))) & (
                        alpha_ner_idx < len(alpha_ner_list) - 1):
                    alpha_ner_idx += 1
                new_ner = ner_list[i]
                new_ner_words = ner_words
        if new_ner != "":
            new_ner_list.append(new_ner.strip().rstrip())

    for i in range(len(alpha_ner_list)):
        for j in range(len(new_ner_list)):
            ner_words = new_ner_list[j].split()
            ner_words = [w for w in ner_words if w[0].isupper()]
            if set(ner_words).issubset(set(alpha_ner_list[i].split(" "))):
                if len(new_ner_list[j].split()) > len(alpha_ner_list[i].split(" ")):
                    alpha_ner_list[i] = new_ner_list[j]
                new_ner_list.pop(j)
                break

    if len(new_ner_list) != 0:
        alpha_ner_list.extend(new_ner_list)

    sort_alpha_ner_list = sorted(alpha_ner_list, key=lambda i : len(i.split(" ")), reverse=True)
    del_ner = set()
    for i in range(len(sort_alpha_ner_list)):
        for j in range(i + 1, len(sort_alpha_ner_list)):
            if sort_alpha_ner_list[j] in sort_alpha_ner_list[i]:
                del_ner.add(sort_alpha_ner_list[j])
    for ner in del_ner:
        alpha_ner_list.remove(ner)
    for i in range(len(alpha_ner_list)):
        if isinstance(alpha_ner_list[i], str):
            if alpha_ner_list.count(alpha_ner_list[i]) == 1:
                s_idx = check_continuity(alpha_ner_list[i].split(" "), s_word, -1)
                alpha_ner_list[i] = (s_idx, alpha_ner_list[i])
            else:
                n_idx = list(filter(lambda x: alpha_ner_list[x] == alpha_ner_list[i], list(range(len(alpha_ner_list)))))
                s_idx = -1
                for idx in n_idx:
                    s_idx = check_continuity(alpha_ner_list[idx].split(" "), s_word, s_idx)
                    alpha_ner_list[idx] = (s_idx, alpha_ner_list[idx])

    alpha_ner_list.sort(key=lambda tup: tup[0])
    ner_list = []
    for i in range(0, len(alpha_ner_list)):
        if i >= 1:
            if alpha_ner_list[i][0] == alpha_ner_list[i - 1][0]:
                if len(alpha_ner_list[i][1].split(" ")) > len(alpha_ner_list[i - 1][1].split(" ")):
                    ner_list.pop()
                    ner_list.append(alpha_ner_list[i][1])
            else:
                    ner_list.append(alpha_ner_list[i][1])
        else:
                ner_list.append(alpha_ner_list[i][1])

    return ner_list

def format_ner(ner, sent):
    if (" . " not in sent) & ("." in ner) & (". " not in sent):
        ner = ner.replace(" . ", ".")
    elif (". " in sent) & ("." in ner):
        ner = ner.replace(" .", ".")
    elif (" + " not in sent) & ("+" in ner)& ("+ " in sent):
        ner = ner.replace(" + ", "+ ")
    puncts = ["/", "-", "–", "−"]
    for p in puncts:
        if (" " + p + " " not in sent) & (p in ner) & (p in sent):
            ner = ner.replace(" " + p + " ", p)
    ner = ner.replace("St .", "St.")
    return ner

# 命名实体：
def extract_ner(sent):
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    source_words = [tok.text for tok in doc]
    sent_words = sent.split(" ")
    spacy_nlp.enable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    ner_list = []
    hyp_words, spilt_words_list = get_hyphen_word(sent)
    for ent in doc.ents:
        if (ent.label_ in ['PERSON', 'GPE', 'ORG', 'NORP', 'PRODUCT', 'EVENT', 'LOC', "LAW"]) & (
                len(ent.text.split()) >= 2):
            new_ner = ent.text
            if check_continuity(ent.text.split(" "), source_words, -1) == -1:
                new_ner = new_ner.replace("-", " - ").replace("–", " – ").replace("−", " − ") \
                    .replace("/", " / ").replace(".", " . ").replace("§", " § ").replace("'s", " 's")
                new_ner = new_ner.replace("St .", "St.")
                new_ner = " ".join(new_ner.split())
            ner_list.append(new_ner.strip().rstrip())
    alpha_ner_list = extract_ner_byAlpha(source_words, spilt_words_list)
    if (len(alpha_ner_list) > 0) | (len(ner_list) > 0):
        ner_list = supplement_ner_list(ner_list, alpha_ner_list, source_words)
        new_ner_list = []
        s_idx_list = []
        i = 0
        while i < len(ner_list):
            if (i == 0) | (len(s_idx_list) == 0):
                last_s_idx = -1
            else:
                last_s_idx = s_idx_list[-1]
            ner = ner_list[i].replace(" § ", " §")
            ner = format_ner(ner, sent)
            n_w = ner.split()
            s_idx = check_continuity(n_w, sent_words, last_s_idx)
            if s_idx == -1:
                new_ner, s_idx = correct_hyp_word_in_ner(hyp_words, n_w, sent_words, last_s_idx)
                if s_idx != -1:
                    if len(new_ner.split(" ")) > 1:
                        new_ner_list.append(new_ner)
                        s_idx_list.append(s_idx)
                    i += 1
                else:
                    i = i - 1
                    if len(s_idx_list) > 0:
                        s_idx_list.pop()
                        new_ner_list.pop()
                    if (i == 0) | (len(s_idx_list) == 0):
                        last_s_idx = -1
                    else:
                        last_s_idx = s_idx_list[-1]
                    new_ner, s_idx = correct_hyp_word_in_ner(hyp_words, ner_list[i].split(), sent_words, last_s_idx)
                    if s_idx != -1:
                        if len(new_ner.split(" ")) > 1:
                            new_ner_list.append(new_ner)
                            s_idx_list.append(s_idx)
                        i += 1
                        last_s_idx = s_idx
                        new_ner, s_idx = correct_hyp_word_in_ner(hyp_words, ner_list[i].split(), sent_words, last_s_idx)
                        if s_idx == -1:
                            new_ner, s_idx = correct_hyp_word_in_ner(hyp_words, ner_list[i].split(), sent_words,
                                                                     last_s_idx - 1)
                            if s_idx != -1:
                                new_ner = new_ner_list[-1] + " " + " ".join(ner_list[i].split()[1:])
                                new_ner_list[-1] = new_ner
                                i += 1
                        else:
                            if len(new_ner.split(" ")) > 1:
                                new_ner_list.append(new_ner)
                                s_idx_list.append(s_idx)
                            i += 1

            else:
                if len(ner.split(" ")) > 1:
                    new_ner_list.append(ner)
                    s_idx_list.append(s_idx)
                i += 1

        if "Millingen aan de Rijn" in sent:
            for i in range(len(new_ner_list)):
                if new_ner_list[i] in "Millingen aan de Rijn":
                    new_ner_list[i] = "Millingen aan de Rijn"
        return new_ner_list, s_idx_list
    else:
        return [], []


def extract_ner_byAlpha(words, split_hyp_words):
    tmp_words = list(words)
    ner_list = []
    # tmp_words[0] = tmp_words[0][0].lower() + tmp_words[0][1:-1]
    for i in range(len(tmp_words)):
        if not tmp_words[i][0].isupper():
            if (tmp_words[i - 1] != '#') & (tmp_words[i] in ["-", "–", "−", "/", ".", "s", "m"]):
                continue
            if (tmp_words[i - 1] != '#') & (tmp_words[i] in split_hyp_words):
                continue
            tmp_words[i] = '#'
        else:
            if i > 2:
                if (tmp_words[i - 2] != "#") & (len(words[i - 1]) == 1) & (words[i - 1] not in [":", ";", ","]):
                    tmp_words[i - 1] = words[i - 1]
    tmp_words[-1] = '#'
    if tmp_words[-2] == ".":
        tmp_words[-2] = '#'
    s = ''
    for word in tmp_words:
        if word == '#':
            s = s.strip()
            if s != "":
                if s.split()[-1] == "/":
                    s = " ".join(s.split()[:-1])
            if len(s.split()) > 1:
                ner_list.append(s)
            s = ''
        else:
            s += word + ' '
    return ner_list

def del_sbar_phrase(res_label, sbar_list, rep_cut_words, res_pp, vp_list):
    # res_label = [1] * len(res_label)
    if len(sbar_list) > 0:
        for sbar in sbar_list:
            sbar_words = sbar[1].split(" ")
            s_idx = check_continuity(sbar_words, rep_cut_words, -1)
            if sbar[0] == "s":
                e_idx = s_idx + len(sbar_words)
                for j in range(s_idx, e_idx):
                    res_label[j] = 0
    if len(res_pp) > 0:
        for pp in res_pp:
            if (pp[0] == "p") & (pp[2] != "of"):
                pp_words = pp[1].split(" ")
                s_idx = check_continuity(pp_words, rep_cut_words, -1)
                for j in range(s_idx, s_idx + len(pp_words)):
                    res_label[j] = 0

    if len(vp_list) > 0:
        for vp in vp_list:
            if vp[0] == "acl":
                vp_words = vp[1].split(" ")
                s_idx = check_continuity(vp_words, rep_cut_words, -1)
                for j in range(s_idx, s_idx + len(vp_words)):
                    res_label[j] = 0
    return res_label


def create_seed_sent(comp_label, res_label, cut_words, sbar_list, rep_cut_words, res_pp, vp_list):
    res_label = [0] * len(res_label)
    common_flag = False
    if res_label.count(1) != len(res_label):
        if "," in cut_words:
            common_sent = " ".join(cut_words[:cut_words.index(",")])
            for sbar in sbar_list:
                if (common_sent in sbar[1]) & (sbar[0] == "s"):
                    common_flag = True
                    break

        for i in range(len(res_label)):
            if cut_words[i] not in [";", ":", "–", "—"]:
                res_label[i] = 1
            elif (cut_words[i] not in [";", ":", "–", "—", ","]) & (not common_flag):
                res_label[i] = 1
            else:
                break

    res_label = del_sbar_phrase(res_label, sbar_list, rep_cut_words, res_pp, vp_list)

    if res_label.count(1) < 4:
        res_label = comp_label

    if rep_cut_words[res_label.index(1)] == ",":
        res_label[res_label.index(1)] = 0

    return res_label


def process_final_result(comp_label, res_label, cut_words, rep_cut_words, sbar_list, res_pp, root_verb, basic_elements, vp_list, sym_sent, dictionary):
    if (cut_words[-1] in [".", "?", "!"]) & (res_label[-1] != 1):
        res_label[-1] = 1

    if (cut_words[-2] in [".", "?", "!"]) & (res_label[-2] != 1):
        res_label[-2] = 1

    comp_res = get_res_by_label(cut_words[:-1], res_label[:-1])
    if (res_label.count(1) < 4) | (res_label.count(1) == len(res_label)) | (len([sbar for sbar in sbar_list if comp_res in sbar])):
        res_label = create_seed_sent(comp_label, res_label, cut_words, sbar_list, rep_cut_words, res_pp, vp_list)

    first_idx = res_label.index(1)
    for i in range(first_idx + 1, len(res_label) - 4):
        if i >= 4:
            if (res_label[i] == 1) & (res_label[i - 4:i].count(1) == 0) & (res_label[i + 1:i + 5].count(1) == 0) & (
                    rep_cut_words[i] != root_verb):
                res_label[i] = 0

    comp_res = get_res_by_label(cut_words, res_label)
    search_idx = max(int(len(comp_res.split(" ")) / 3), 4) + 1
    if "but" in comp_res.split(" ")[-search_idx:-1]:
        one_indexs = list(filter(lambda x: res_label[x] == 1, list(range(len(res_label)))))[-search_idx:-1]
        change_idx = -1
        for idx in one_indexs:
            if cut_words[idx] == "but":
                change_idx = idx
            if change_idx != -1:
                res_label[idx] = 0
        comp_res = get_res_by_label(cut_words, res_label)

    if comp_res.split(" ")[0] == ",":
        res_label[res_label.index(1)] = 0

    if comp_res.split(" ")[-2] in dictionary["end"]:
        orig_one_idx = len(res_label) - 2
        while res_label[orig_one_idx] != 1:
            orig_one_idx -= 1
        s_idx = orig_one_idx + 1
        while (s_idx < len(rep_cut_words) - 1) & (rep_cut_words[s_idx] not in [";", ":", "–"]):
            if (s_idx - orig_one_idx > 5) & (rep_cut_words[s_idx] == ","):
                break
            res_label[s_idx] = 1
            s_idx += 1
        comp_res = get_res_by_label(cut_words, res_label)
        if "very little to do with" not in comp_res:
            if len(sbar_list) > 0:
                for sbar in sbar_list:
                    sbar_words = sbar[1].split(" ")
                    s_idx = check_continuity(sbar_words, rep_cut_words, -1)
                    if s_idx > orig_one_idx + 1:
                        e_idx = s_idx + len(sbar_words)
                        for i in range(s_idx, e_idx):
                            res_label[i] = 0
            if len(res_pp) > 0:
                for pp in res_pp:
                    if pp[0] == "p":
                        pp_words = pp[1].split(" ")
                        s_idx = check_continuity(pp_words, rep_cut_words, -1)
                        if s_idx not in [orig_one_idx + 1, orig_one_idx + 2]:
                            for j in range(s_idx, s_idx + len(pp_words)):
                                res_label[j] = 0

    comp_res = get_res_by_label(cut_words, res_label)
    comp_res_words = comp_res.split(" ")

    for elem in basic_elements:
        if "," in elem[2]:
            elem_words = elem[2].split(" , ")[0].split(" ")
        else:
            elem_words = elem[2].split(" ")
        if len(elem_words) > 1:
            if comp_res_words[-2] in elem_words[:-1]:
                e_idx = check_continuity(elem_words, rep_cut_words, -1)
                for j in range(e_idx, e_idx + len(elem_words)):
                    res_label[j] = 1
            else:
                elem_str = " ".join(elem_words)
                pp_obj = [pp for pp in res_pp if elem_str in pp[1]]
                if (elem[1] in ["dobj", "attr", "advmod"]) & (elem_str not in comp_res) & (len(pp_obj) == 0) & (elem_str in sym_sent):
                    e_idx = check_continuity(elem_words, rep_cut_words, -1)
                    for j in range(e_idx, e_idx + len(elem_words)):
                        res_label[j] = 1

    comp_res = get_res_by_label(cut_words, res_label)
    comp_res_words = comp_res.split(" ")
    if comp_res_words[0] in dictionary["start"]:
        idx = 0
        while res_label[idx] != 1:
            res_label[idx] = 1
            idx += 1

    change_flag = False
    for sbar in sbar_list:
        tmp = [j for j in sbar[1].split(" ") if j in ["which", "who", "where", "when", "while"]]
        comp_res_words = get_res_by_label(cut_words[:-1], res_label[:-1]).split(" ")
        if (len(tmp) != 0) & (set(comp_res_words).issubset(set(sbar[1].split(" ")))):
            first_idx = res_label.index(1)
            if first_idx != 0:
                idx = 0
                while res_label[idx] != 1:
                    res_label[idx] = 1
                    idx += 1
                change_flag = True

    if change_flag:
        for sbar in sbar_list:
            sbar_words = sbar[1].split(" ")
            s_idx = check_continuity(sbar_words, rep_cut_words, -1)
            if s_idx > idx:
                e_idx = s_idx + len(sbar_words)
                for i in range(s_idx, e_idx):
                    res_label[i] = 0

    first_idx = res_label.index(1)
    if cut_words[first_idx] == "that":
        res_label[first_idx] = 0
        first_idx = res_label.index(1)

    last_idx = first_idx
    for j in range(first_idx + 1, len(res_label) - 1):
        if (rep_cut_words[j] in [":", ";", ","]) & (res_label[j] == 0):
            next_idx = j + 1
            while rep_cut_words[next_idx] not in [":", ";", ","]:
                next_idx += 1
                if next_idx > len(res_label) - 2:
                    next_idx -= 1
                    break
            if (res_label[j + 1:next_idx].count(1) != 0) & (rep_cut_words[last_idx] not in [":", ";", ","]):
                res_label[j] = 1
        if res_label[j] == 1:
            last_idx = j

    one_indexs = list(filter(lambda x: res_label[x] == 1, list(range(len(res_label)))))
    if cut_words[one_indexs[-2]] in ["that", "about", ":", ";", ",", "``", "but"]:
        res_label[one_indexs[-2]] = 0

    if cut_words[one_indexs[-3]] in [":", ";", ",", "``"]:
        if (cut_words[one_indexs[-3]] != ":") | (cut_words[one_indexs[-4]] not in ["include", "included", "are"]):
            res_label[one_indexs[-2]] = 0
            res_label[one_indexs[-3]] = 0

    comp_res = get_res_by_label(cut_words, res_label)

    if "``" in cut_words:
        dquot_index = list(filter(lambda x: cut_words[x] == "``", list(range(len(cut_words)))))
        for idx in dquot_index:
            if res_label[idx] == 0:
                if (res_label[idx - 1] == 1) & (res_label[idx + 1] == 1):
                    res_label[idx] = 1
            if res_label[idx] == 1:
                n_idx = idx + 1
                while (cut_words[n_idx] != "''") & (n_idx < len(cut_words)):
                    n_idx += 1
                res_label[n_idx] = 1

    if "\'" in cut_words:
        quot_index = list(filter(lambda x: cut_words[x] == "\'", list(range(len(cut_words)))))
        for j in range(len(quot_index)):
            if res_label[quot_index[j]] == 0:
                if (res_label[quot_index[j] - 1] == 1) & (res_label[quot_index[j] + 1] == 1):
                    res_label[quot_index[j]] = 1
            if (res_label[quot_index[j]] == 1) & (j + 1 < len(quot_index)):
                res_label[quot_index[j + 1]] = 1

    if "%" in cut_words:
        per_index = list(filter(lambda x: rep_cut_words[x] == "%", list(range(len(rep_cut_words)))))
        for idx in per_index:
            if res_label[idx] == 1:
                res_label[idx - 1] = 1

    if "'s" in cut_words:
        per_index = list(filter(lambda x: rep_cut_words[x] == "'s", list(range(len(rep_cut_words)))))
        for idx in per_index:
            if res_label[idx - 1] == 1:
                res_label[idx] = 1

    if "but" in cut_words:
        but_idx = cut_words.index("but")
        if res_label[but_idx + 1] == 1:
            res_label[but_idx] = 1

    if (res_label.count(1) < 4) | (res_label.count(1) == len(res_label)):
        res_label = create_seed_sent(comp_label, res_label, cut_words, sbar_list, rep_cut_words, res_pp, vp_list)
        comp_res = get_res_by_label(cut_words, res_label)

    if ", ," in comp_res:
        one_indexs = list(filter(lambda x: res_label[x] == 1, list(range(len(res_label)))))
        for j in range(len(one_indexs) - 1):
            if (rep_cut_words[one_indexs[j]] == ",") & (rep_cut_words[one_indexs[j + 1]] == ","):
                res_label[one_indexs[j + 1]] = 0

    return res_label

def handle_included(conj_res, res_label):
    add_index = [-1, -1]
    remove_index = [-1, -1]
    if conj_res == []:
        return
    if conj_res[0][0] == 'complexity , circuit complexity , and decision tree complexity':
        add_index = [9, 17]
    if conj_res[0][0] == 'Wedge , Huntington Beach , and Malibu':
        add_index = [15, 27]
        remove_index = [29, 30]
    if conj_res[0][0] == 'Swedes , and Anglo-Danes ':
        conj_res[0][0] = 'Swedes , and Anglo-Danes'
        add_index = [15, 29]
    if conj_res[0][0] == 'include 5 University campuses ; 12 California State University campuses ; and private institutions':
        add_index = [0, 1]
    if add_index != [-1, -1]:
        res_label[add_index[0]:add_index[1]+1] = [1]*(add_index[1]-add_index[0]+1)
    if remove_index != [-1, -1]:
        res_label[remove_index[0]:remove_index[1]+1] = [0]*(remove_index[1]-remove_index[0]+1)


def process_conj(rep_cut_words, temp_res_label, res_label, pp_flag):
    conj_str = get_res_by_label(rep_cut_words, temp_res_label)
    conj_res = extract_conj(conj_str.strip().rstrip())
    # print(conj_res)
    # # print(conj_res)
    handle_included(conj_res, res_label)
    for conj_li in conj_res:
        if conj_li[1] == 1:
            conj_word = conj_li[0].split(" ")
            conj_index = -1
            index_conj = 0
            index_start_conj = -1
            conj_is_exist = 0
            conj_mapping_cut = []
            conj_word_index = -1
            for temp in range(len(rep_cut_words) - 1, -1, -1):
                if res_label[temp] != -1 and pp_flag[temp] == 0:
                    if rep_cut_words[temp] == conj_word[index_conj]:
                        conj_mapping_cut.append([index_conj, temp])
                        if conj_word[index_conj] == 'and' or conj_word[index_conj] == 'or':
                            conj_word_index = len(conj_mapping_cut) - 1
                        if index_conj == 0:
                            index_start_conj = temp
                        index_conj += 1
                        if index_conj == len(conj_word):
                            conj_is_exist = 1
                            # conj_index = index_start_conj
                            break
                        temp += 1
            conj_mapping_cut = conj_mapping_cut[::-1]
            if conj_is_exist:
                # 三种情况，a and/or b
                # a and存在，填补b
                # b and存在，填补a
                # a b 存在,填补连接词
                left_check = 1
                special_comma_list = []
                left_check_num = 0
                for check_conj in conj_mapping_cut[0:conj_word_index]:
                    if res_label[check_conj[1]] == 0:
                        if rep_cut_words[check_conj[1]] == ',':
                            special_comma_list.append(check_conj[1])
                        # else:
                        #     left_check = 0
                        #     break
                    else:
                        left_check_num += 1
                special_comma_list_left = len(special_comma_list)
                if left_check_num < (conj_word_index - special_comma_list_left) / 2:
                    left_check = 0
                right_check = 1
                right_check_num = 0
                for check_conj in conj_mapping_cut[conj_word_index + 1:len(conj_mapping_cut)]:
                    if res_label[check_conj[1]] == 0:
                        if rep_cut_words[check_conj[1]] == ',':
                            special_comma_list.append(check_conj[1])
                        # else:
                        #     right_check = 0
                        #     break
                    else:
                        right_check_num += 1
                special_comma_list_right = len(special_comma_list) - special_comma_list_left
                if right_check_num < (len(conj_mapping_cut) - conj_word_index - 1 - special_comma_list_right) / 2:
                    right_check = 0

                conj_check = 1 if res_label[conj_mapping_cut[conj_word_index][1]] == 1 else 0
                if left_check and right_check and conj_check:
                    pass
                    # pass
                    for check_conj in conj_mapping_cut:
                        if check_conj[1] not in special_comma_list:
                            res_label[check_conj[1]] = 1
                elif left_check == 0 and right_check and conj_check:
                    # a不全,补全a
                    # for check_conj in conj_mapping_cut[0:conj_word_index]:
                    for check_conj in conj_mapping_cut:
                        if check_conj[1] not in special_comma_list:
                            res_label[check_conj[1]] = 1
                elif left_check and right_check == 0 and conj_check:
                    # b不全,补全b
                    # for check_conj in conj_mapping_cut[conj_word_index + 1:len(conj_mapping_cut)]:
                    for check_conj in conj_mapping_cut:
                        if check_conj[1] not in special_comma_list:
                            res_label[check_conj[1]] = 1
                elif left_check and right_check and conj_check == 0:
                    res_label[conj_mapping_cut[conj_word_index][1]] = 1
                else:
                    if right_check == 1:
                        # 把连接词及后面去掉
                        for check_conj in conj_mapping_cut[0:conj_word_index + 1]:
                            res_label[check_conj[1]] = 0
                    else:
                        for check_conj in conj_mapping_cut[conj_word_index:len(conj_mapping_cut)]:
                            res_label[check_conj[1]] = 0
            else:
                # 没找到，把连接词及后面去掉
                index_conj = 0
                is_end = 0
                for temp in range(len(res_label)):
                    if res_label[temp] != -1 and pp_flag[temp] == 0:
                        if rep_cut_words[temp] == conj_word[index_conj]:
                            if conj_word[index_conj] == 'and' or conj_word[index_conj] == 'or':
                                is_end = 1
                            if is_end:
                                res_label[temp] = 0
                            index_conj += 1
                            if index_conj == len(conj_word):
                                break
                            temp += 1
        elif conj_li[1] == 2:
            conj_word = conj_li[0].split(" ")
            conj_index = -1
            for temp in range(len(rep_cut_words)):
                if conj_word[0] == rep_cut_words[temp] and conj_word == rep_cut_words[temp:temp + len(conj_word)]:
                    conj_index = temp
                    break
            if not conj_index == -1:
                # print("conj_index: ", conj_index, conj_index + len(conj_word),
                # cut_words[conj_index:conj_index + len(conj_word)])
                check_index = False
                for check_conj in range(conj_index, conj_index + len(conj_word)):
                    if res_label[check_conj] == 1:
                        check_index = True
                        break
                if check_index:
                    for check_conj in range(conj_index, conj_index + len(conj_word)):
                        res_label[check_conj] = 1
    return res_label, conj_res


## process root verb and subject
def modify_basic_elements(basic_elements, rep_cut_words, res_label, sbar_list, sym_list):
    subj_list = [elem for elem in basic_elements if (("subj" in elem[1]) | ("expl" in elem[1]))]
    root_verb = [elem for elem in basic_elements if elem[1] == "ROOT"]
    subj_flag = False
    for elem in root_verb:
        if elem[1] == "ROOT":
            not_root = False
            for sent in sym_list[1:]:
                if elem[2] in sent:
                    not_root = True
                    break
            if not not_root:
                e_idx = check_continuity(elem[2].split(" "), rep_cut_words, -1)
                res_label[e_idx] = 1

    idx_diff = -1
    if len(subj_list) == 0:
        return res_label
    subj_elem = subj_list[0]
    if len(root_verb) != 0:
        for subj in subj_list:
            if subj[0] < root_verb[0][0]:
                if idx_diff == -1:
                    idx_diff = root_verb[0][0] - subj[0]
                    subj_elem = subj
                else:
                    if root_verb[0][0] - subj[0] < idx_diff:
                        idx_diff = root_verb[0][0] - subj[0]
                        subj_elem = subj

    subj_str = subj_elem[3]
    not_subj = False
    for sbar in sbar_list:
        if (" " + subj_str + " " in sbar[1]) & (sbar[0] == "s"):
            not_subj = True
            break
    for sent in sym_list[1:]:
        if subj_str in sent:
            not_subj = True
            break
    if ((not not_subj) | (len(subj_list) == 1)) & (not subj_flag):
        e_idx = check_continuity(subj_elem[2].split(" "), rep_cut_words, -1)
        if ("of" in subj_str) | (len(subj_str.split(" ")) <= 3) | (res_label[e_idx] == 0):
            e_idx = check_continuity(subj_str.split(" "), rep_cut_words, -1)
            for j in range(e_idx, e_idx + len(subj_str.split(" "))):
                res_label[j] = 1
    return res_label


def grammar_check_one_sent(orig_sent, cut_sent, comp_label, dictionary):
    rep_cut_sent = cut_sent.replace("``", "\"").replace("''", "\"")
    res_label = list(comp_label)
    hyp_words, spill_words_list = get_hyphen_word(rep_cut_sent)
    abbr_words = get_abbr_word(rep_cut_sent)
    vp_list, basic_elements, root_verb, root_idx = get_verb_phrases(rep_cut_sent, hyp_words, spill_words_list)
    for_list = extra_formulation(rep_cut_sent)
    ner_list, ner_sidx_list = extract_ner(rep_cut_sent)
    sbar_list, pos_list, cc_sent_list, np_sbar_list, np_pp_list = extra_sbar(rep_cut_sent, hyp_words)
    pp_list = get_prep_list_by_dependency(rep_cut_sent, hyp_words, spill_words_list, abbr_words, basic_elements)
    sym_list = []
    if (";" in rep_cut_sent) | (" – " in rep_cut_sent) | (" — " in rep_cut_sent):
        sym_list = re.split(' ; | – | — ', rep_cut_sent)
    elif ":" in rep_cut_sent:
        if (':', 'SYM') in pos_list:
            idx = pos_list.index((':', 'SYM'))
        else:
            idx = pos_list.index((':', ':'))
        if ("NN" in pos_list[idx - 1][1]) | ("VBG" in pos_list[idx - 1][1]):
            sym_list = re.split(' : ', rep_cut_sent)
    else:
        sym_list = []
    sbar_list, new_pp_list = using_pp_update_sbar(rep_cut_sent, sbar_list, pos_list, dictionary, pp_list, hyp_words)
    rep_cut_words = rep_cut_sent.split(" ")
    res_label = modify_basic_elements(basic_elements, rep_cut_words, res_label, sbar_list, sym_list)
    sym_sent = cut_sent
    if len(vp_list) > 0:
        vp_flag = [0] * len(rep_cut_words)
        s_idx = -1
        for vp in vp_list:
            vp_words = vp[1].split(" ")
            s_idx = check_continuity(vp_words, rep_cut_words, s_idx)
            vp_flag = fill_sent_flag(vp_flag, s_idx, s_idx + len(vp_words))
        res_label = check_vp_integrity(res_label, rep_cut_words, vp_list, vp_flag)
        print("vp modify:", get_res_by_label(rep_cut_words, res_label))

    if len(sbar_list) > 0:
        sbar_flag = [0] * len(rep_cut_words)
        s_idx = -1
        for sbar in sbar_list:
            sbar_words = sbar[1].split(" ")
            s_idx = check_continuity(sbar_words, rep_cut_words, s_idx)
            sbar_flag = fill_sent_flag(sbar_flag, s_idx, s_idx + len(sbar_words))
        res_label = check_sbar_integrity(res_label, sbar_list, sbar_flag, rep_cut_words, pp_list, basic_elements)
        print("sbar modify:", get_res_by_label(rep_cut_words, res_label))

    res_pp = filter_pp_in_sbar(sbar_list, new_pp_list)
    if len(res_pp) > 0:
        pp_flag = [0] * len(rep_cut_words)
        s_idx = -1
        for pp in res_pp:
            pp_words = pp[1].split(" ")
            s_idx = check_continuity(pp_words, rep_cut_words, s_idx)
            pp_flag = fill_sent_flag(pp_flag, s_idx, s_idx + len(pp_words))
        res_label = check_pp_integrity(rep_cut_words, res_label, res_pp, pp_flag, ner_list, sbar_list)
        print("pp modify:", get_res_by_label(rep_cut_words, res_label))
    else:
        pp_flag = [0] * len(rep_cut_words)

    if len(for_list) > 0:
        for_flag = [0] * len(rep_cut_words)
        for f in for_list:
            f_words = f.split(" ")
            s_idx = check_continuity(f_words, rep_cut_words, -1)
            for_flag = fill_sent_flag(for_flag, s_idx, s_idx + len(f_words))
        res_label = check_formulation_intergrity(res_label, for_list, for_flag)

    if len(ner_list) > 0:
        ner_flag = [0] * len(rep_cut_words)
        for j in range(len(ner_list)):
            ner_words = ner_list[j].split(" ")
            ner_flag = fill_sent_flag(ner_flag, ner_sidx_list[j], ner_sidx_list[j] + len(ner_words))
        res_label = check_ner_intergrity(res_label, ner_list, ner_flag)
        print("ner modify:", get_res_by_label(rep_cut_words, res_label))

    if len(sym_list) > 0:
        sem_flag = [0] * len(rep_cut_words)
        for j in range(0, len(sym_list)):
            sem_words = sym_list[j].strip().rstrip().split(" ")
            s_idx = check_continuity(sem_words, rep_cut_words, -1)
            sem_flag = fill_sent_flag(sem_flag, s_idx, s_idx + len(sem_words))
        res_label, sym_idx = check_symbols_integrity(res_label, sym_list, sem_flag, res_pp, sbar_list, rep_cut_words, root_idx)
        sym_sent = sym_list[sym_idx]
        s_idx = check_continuity(sym_sent.strip().rstrip().split(" "), rep_cut_words, -1)
        temp_res_label = [0] * len(res_label)
        for j in range(s_idx, s_idx + len(sym_sent.strip().rstrip().split(" "))):
            temp_res_label[j] = 1
        temp_res_label[-1] = 1
        print("sym modify:", get_res_by_label(rep_cut_words, res_label))
    else:
        temp_res_label = [1] * len(res_label)
    temp_res_label = del_sbar_phrase(temp_res_label, sbar_list, rep_cut_words, res_pp, vp_list)
    res_label, conj_res = process_conj(rep_cut_words, temp_res_label, res_label, pp_flag)
    print("conj modify:", get_res_by_label(rep_cut_words, res_label))
    res_label = process_final_result(comp_label, res_label, cut_sent.split(" "), rep_cut_words, sbar_list, pp_list, root_verb,
                                     basic_elements, vp_list, sym_sent, dictionary)
    orig_words = orig_sent.split(" ")
    cut_idx = search_cut_content(orig_words)
    if len(cut_idx) != 0:
        for tup in cut_idx:
            count = tup[1] - tup[0] + 1
            for j in range(count):
                res_label.insert(tup[0], -2)

    return res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, cc_sent_list, np_sbar_list, np_pp_list


## check completeness of prep phrases, clause
def grammar_check_all_sents(cut_sents, comp_label, orig_sents, start_idx, end_idx):
    comp_list = []
    orig_comp = []
    label_list = []
    all_sbar = []
    all_pp = []
    all_conj = []
    all_formulations = []
    all_ners = []
    all_vps = []
    all_cc_sent = []
    all_np_sbar = []
    all_np_pp = []
    dictionary = load_dictionary('./Dictionary.txt')
    for i in range(start_idx, end_idx):
        res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, cc_sent_list, np_sbar_list, np_pp_list = grammar_check_one_sent(orig_sents[i], cut_sents[i], comp_label[i], dictionary)
        cut_res = get_res_by_label(cut_sents[i].split(" "), comp_label[i])
        orig_comp.append(cut_res)
        print("original result: ", cut_res)
        orig_res = get_res_by_label(orig_sents[i].split(" "), res_label)
        print("modify result: ", orig_res)
        comp_list.append(orig_res)
        label_list.append(res_label)
        all_sbar.append(sbar_list)
        all_pp.append(pp_list)
        all_conj.append(conj_res)
        all_formulations.append(for_list)
        all_ners.append(ner_list)
        all_vps.append(vp_list)
        all_cc_sent.append(cc_sent_list)
        all_np_sbar.append(np_sbar_list)
        all_np_pp.append(np_pp_list)
    write_list_in_txt(orig_sents, comp_list, orig_comp, "./modify_res.txt")
    return label_list, all_sbar, all_pp, all_conj, comp_list, all_formulations, all_ners, all_vps, all_cc_sent, all_np_sbar, all_np_pp


def grammar_check_main(start_idx, end_idx):
    cut_sent_path = "./comp_input/ncontext.cln.sent"
    orig_sent_path = "./comp_input/context.cln.sent"
    comp_label = load_label("./ncontext_result_greedy.sents")
    cut_sents = load_orig_sent(cut_sent_path)
    orig_sents = load_orig_sent(orig_sent_path)
    label_list, all_sbar, all_pp, all_conj, comp_list, all_for, all_ners, all_vps = grammar_check_all_sents(cut_sents, comp_label, orig_sents, start_idx, end_idx)
    return label_list, all_sbar, all_pp, all_conj, comp_list, all_for, all_ners, all_vps


if __name__ == '__main__':
    file_name = "context"
    cut_sent_path = "./comp_input/ncontext.cln.sent"
    orig_sent_path = "./comp_input/context.cln.sent"
    comp_label = load_label("./ncontext_result_greedy.sents")
    cut_sents = load_orig_sent(cut_sent_path)
    orig_sents = load_orig_sent(orig_sent_path)
    start_idx = 0
    end_idx = len(cut_sents)
    # end_idx = 6000
    grammar_check_all_sents(cut_sents, comp_label, orig_sents, start_idx, end_idx)


