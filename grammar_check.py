import re
import spacy
from nltk import CoreNLPParser

## Stanford Corenlp constituency parser
from preprocess import load_formulation, format_formulation, search_cut_content

eng_parser = CoreNLPParser('http://127.0.0.1:9000')
## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")
sub_nlp = spacy.load("en_core_web_sm")
sub_nlp.add_pipe("merge_noun_chunks")
sub_nlp.add_pipe("merge_entities")

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
            if key != "comp":
                dictionary[key] = {}
            else:
                dictionary[key] = []
        elif len(line) > 1:
            if key != "comp":
                l_words = line[:-1].split(" ")
                index = l_words.index(key)
                if index != 0:
                    w_idx = index - 1
                else:
                    w_idx = index + 1
                dictionary[key][l_words[w_idx]] = line[:-1]
            else:
                dictionary[key].append(line[:-1])
            # l_words = line[:-1].split(" ")
            # if key == "comp":
            #     dictionary[key].append(line[:-1])
            # else:
            #     index = l_words.index(key) - 1
            #     dictionary[key].append(l_words[index])
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


#
# def get_subject_phrase(doc):
#     for token in doc:
#         if "subj" in token.dep_:
#             subtree = list(token.subtree)
#             start = subtree[0].i
#             end = subtree[-1].i + 1
#             return doc[start:end]
#     return []
#
#
# def get_object_phrase(doc):
#     for token in doc:
#         if "dobj" in token.dep_:
#             subtree = list(token.subtree)
#             start = subtree[0].i
#             end = subtree[-1].i + 1
#             return doc[start:end]
#     return []

## check the continuity of phrase/clause
def check_continuity(key_words, words, last_s_idx):
    flag = False
    s_idx = -1
    while not flag & (s_idx != last_s_idx):
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


def exist_pp(pos_list, pp_word, dictionary, key_pp, to_flag, spill_words_list):
    count = 0
    key = -1
    sent = " ".join(pp_word).replace(" - ", "-").replace(" – ", "–")
    #sent = process_hyp_words(" ".join(pp_word), hyp_words, orig_sent)
    doc = nlp(sent)
    i = 0
    p_idx = pp_word.index(key_pp)
    for pp in dictionary["comp"]:
        if pp in " ".join(pp_word):
            p_idx = pp_word.index(pp.split(" ")[-1])
            break
    for w in doc:
        if i > p_idx:
            if (pos_list[i] == "ADP") & (w.text not in ["of", "v"]):
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
                        if w.text in ["during", "after", "before"]:
                            count += 1
                        ## on Monday in March
                        if (network[0] in dictionary["on"].keys()) | (network[0] in dictionary["in"].keys()):
                            count += 1
                    else:
                        count += 1
                        words = [tok.orth_ for tok in w.subtree]
                        if words[0] != w.text:
                            p_idx = words.index(w.text)
                            key = i - p_idx
                            break
            elif w.text in ["while", "when", "where", "which", "who", "that", "during", "according"]:
                count += 1
            elif (w.text == "to") & (not to_flag) :
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


# def get_to_prep_by_constituency(sent):
#     nlp_tree = get_nlp_tree(sent)
#     pp_list = []
#     for st in nlp_tree.subtrees():
#         label = st.label()
#         pp_words = st.leaves()
#         pp = " ".join(pp_words)
#         pos_list = st.pos()
#         if (label in ["VP"]) & (pp.split(" ")[0] == "to"):
#             flag, sub_string = exist_pp(st)
#             if flag:
#                 to_prep = devide_to_pp(pos_list, pp, sub_string)
#             else:
#                 to_prep = pp
#             pp_list.append(to_prep)
#     if pp_list:
#         return True, pp_list
#     else:
#         return False, pp_list

## When there are multiple prepositional phrases, cut and keep the first one
# def check_pp_end(subtree, end_word, pw, ):
# s_idx = subtree.index(pw)
# for w in end_word:
#     if w in subtree[s_idx + 1:]:
#         finish = w
#         break
# e_idx = subtree.index(finish, s_idx + 1)
# pp = subtree[:e_idx + 1]
# return pp

## Complement prep phrase
def get_the_complete_phrase(p_word, h_word, s_word, pp, pos_list, all_pos_list, pp_list, hyp_words, orig_sent):
    new_pp = list(pp)
    new_pos_list = list(pos_list)
    if " ".join(pp) not in " ".join(s_word):
        p_idx = s_word.index(p_word)
    else:
        p_idx = check_continuity(pp, s_word, -2)
    h_idx = -1
    if h_word in s_word[0:p_idx]:
        h_idx = p_idx - 1
        while s_word[h_idx:p_idx].count(h_word) == 0:
            h_idx = h_idx - 1
        # h_idx = s_word.index(h_word, search_idx, p_idx)
        if p_idx - h_idx < 5:
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
            for w in hyp_words:
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
                pp_str = process_hyp_words(" ".join(new_pp), hyp_words, orig_sent)
                if pp_list[-1][1] in pp_str:
                    h_idx = -1
                    new_pp = pp
                    new_pos_list = pos_list
        else:
            h_idx = -1
    return new_pp, new_pos_list, h_idx


def get_prep_of(doc, dictionary, all_pos_list, s_word):
    i = 0
    prep_of = []
    for token in doc:
        if token.text == "of":
            if (token.dep_ in ["prep", "cc"]) & (s_word[i - 1] != ","):
                head_text = token.head.text
                if token.head.pos_ in ["ADJ", "VERB"]:
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
                elif (head_text.lower() in dictionary["of"].keys()) | (
                        (head_text[:-1].lower() in dictionary["of"].keys()) & (head_text[-1] == "s")):
                    if head_text[:-1].lower() in dictionary["of"].keys():
                        comp_prep_word = dictionary["of"][head_text[:-1].lower()].split(" ")
                    else:
                        comp_prep_word = dictionary["of"][head_text.lower()].split(" ")
                    if comp_prep_word[0] in ["a", "an"]:
                        if (head_text[:-1].lower() in dictionary["of"].keys()) & (head_text[-1] == "s"):
                            if "NUM" in all_pos_list[0:i]:
                                search_idx = i - 1
                                while (all_pos_list[search_idx:i].count("NUM") == 0) & (s_word[search_idx] not in ["all"]):
                                    search_idx = search_idx - 1
                                prep_of.append(" ".join(s_word[search_idx:i]))
                            else:
                                prep_of.append(s_word[i - 1])
                        elif "DET" in all_pos_list[0:i]:
                            search_idx = i - 1
                            while (all_pos_list[search_idx:i].count("DET") == 0) & (s_word[search_idx] != ","):
                                search_idx = search_idx - 1
                            #s_idx = all_pos_list[search_idx:i].index("DET")
                            if s_word[search_idx] == ",":
                                search_idx += 1
                            prep_of.append(" ".join(s_word[search_idx:i]))
                        elif ("one" in s_word[0:i]) | ("One" in s_word[0:i]):
                            search_idx = i - 1
                            while all_pos_list[search_idx:i].count("NUM") == 0:
                                search_idx = search_idx - 1
                            #s_idx = all_pos_list[search_idx:i].index("NUM")
                            prep_of.append(" ".join(s_word[search_idx:i]))
                        else:
                            prep_of.append(s_word[i - 1])

                    else:
                        if s_word[i - 1] == "percent":
                            search_idx = i - 2
                            while all_pos_list[search_idx:i].count("NUM") == 0:
                                search_idx = search_idx - 1
                            s_idx = all_pos_list[search_idx:i].index("NUM")
                            prep_of.append(" ".join(s_word[s_idx + search_idx:i]))
                        else:
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
    for w in s_word:
        if (("-" in w) | ("–" in w) | ("−" in w)) & (len(w) != 1):
            hyphen_words.append(w)

    word_list = []
    for w in hyphen_words:
        if "-" in w:
            word_list.extend(w.split("-"))
        else:
            word_list.extend(w.split("–"))

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
    s_idx = check_continuity(pp_word, s_word, -2)
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
                # pp_word.append(s_word[idx])
                # pos_list.append(all_pos_list[idx])
                if pp_for == nf:
                    comp_f = f
                    break
                # pp_word.extend(s_word[e_idx + 1:idx + 1])
                # pos_list.extend(all_pos_list[e_idx + 1:idx + 1])
            break
    ## process abbr word
    for abbr in abbr_words:
        if pp_word[-1] == abbr[:len(pp_word[-1])]:
            pp_abbr = pp_word[-1]
            for idx in range(e_idx + 1, len(s_word)):
                pp_abbr = pp_abbr + s_word[idx]
                # pp_word.append(s_word[idx])
                # pos_list.append(all_pos_list[idx])
                if pp_abbr == abbr:
                    comp_abbr = abbr
                    break
            break
    return pp_word, pos_list, comp_f, comp_abbr


def get_complete_last_word(pp_word, s_word):
    if len(pp_word) >= 3:
        s_idx = check_continuity(pp_word[:-1], s_word, -2)
    else:
        s_idx = check_continuity(pp_word, s_word, -2)
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


## obtain all prepositional phrases in one sentence by dependency relation
def get_prep_list_by_dependency(sent, hyp_words, spill_words_list, abbr_words):
    print(sent)
    pp_list = []
    doc = nlp(sent)
    dictionary = load_dictionary("./Dictionary.txt")
    noun_chunks = []
    all_pos_list = [tok.pos_ for tok in doc]
    s_word = [tok.text for tok in doc]
    pp_flag = [0] * len(all_pos_list)
    ## save the word before "of"
    prep_of = get_prep_of(doc, dictionary, all_pos_list, s_word)
    ## save noun chunks
    for i in doc.noun_chunks:
        noun_chunks.append(i.text)
    i = 0
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
                pp_str = process_hyp_words(" ".join(pp_word), hyp_words, sent)
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
                ## save verb phrase
                if (w.head.pos_ == "VERB") & (w.dep_ == "prep"):
                    if pp_word[0] not in ["during", "after", "before", "via", "from"]:
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.text, w.head.text, s_word, pp_word,
                                                                           pos_list, all_pos_list, pp_list, hyp_words,
                                                                           sent)
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                        if h_idx != -1:
                            vp_flag = True
                    else:
                        if w.head.dep_ == "acl":
                            tmp_str = w.head.text + " " + w.text
                            if tmp_str in sent:
                                v_idx = check_continuity(tmp_str.split(" "), s_word, -2)
                                pp_word.insert(0, w.head.text)
                                pos_list.insert(0, all_pos_list[v_idx])
                                pp_flag[v_idx] = 0
                                vp_flag = False
                ## process on Monday...in March
                if (alternative[0] in ["on", "in"]) & (len(alternative) == 2):
                    if (alternative[1] in dictionary["on"].keys()) | (alternative[1] in dictionary["in"].keys()):
                        pp_str = " ".join(alternative)
                        pp_list.append(('p', pp_str))
                        s_idx = check_continuity(alternative, s_word, -2)
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
                    pp_word = pp_word[:key]
                if pp_str == "of which":
                    i += 1
                    if len(prep_of) != 0:
                        prep_of.pop(0)
                    continue
                ## del ","
                if pp_word[-1] in [",", "."]:
                    pp_word.pop(len(pp_word) - 1)
                    pos_list.pop(len(pp_word) - 1)
                ##Supplement of phrase
                if w.text == "of":
                    if (prep_of[0] != 'X') & (len(pp_list) > 0) & (pp_word[0] == "of"):
                            # last_pp = process_hyp_words(pp_list[-1][1], hyp_words)
                        last_pp = pp_list[-1][1].replace("-", " - ").replace("–", " – ")
                        if " ".join(pp_word) in last_pp:
                            prep_of.pop(0)
                            i += 1
                            continue
                        last_pp_word = last_pp.split(" ")
                        prefix = prep_of[0].split(" ")
                        for i in range(len(prefix) - 1, -1, -1):
                            pp_word.insert(0, prefix[i])
                        if prefix[-1] == last_pp_word[-1]:
                            s_idx = check_continuity(last_pp_word, s_word, -2)
                            if s_word[s_idx + len(last_pp_word)] == "of":
                                for i in range(len(last_pp_word)):
                                    pp_flag[s_idx + i] = 0
                                for i in range(len(prefix)):
                                    if len(last_pp_word) > 0:
                                        last_pp_word.pop()
                                    else:
                                        break
                                last_pp_word.extend(pp_word)
                                pp_word = last_pp_word
                                if pp_list[-1][0] == "v":
                                    vp_flag = True
                                else:
                                    vp_flag = False
                                pp_list.pop()
                    prep_of.pop(0)

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
                pp_str = process_hyp_words(pp_str, hyp_words, sent)
                pp_str = get_complete_last_word(pp_str.split(" "), sent.split(" "))

                if len(pp_list) > 0:
                    if pp_str in pp_list[-1]:
                        i += 1
                        continue

                if pp_str in sent:
                    s_idx = check_continuity(pp_word, s_word, -2)
                    if pp_flag[s_idx] != 1:
                        if vp_flag:
                            pp_list.append(('v', pp_str))
                        else:
                            pp_list.append(('p', pp_str))
                        pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
            elif w.text == "to":
                if (w.head.pos_ == "VERB") & (w.dep_ == "aux"):
                    pp_word = [tok.orth_ for tok in w.head.subtree]
                    pos_list = [tok.pos_ for tok in w.head.subtree]
                    pp_str = process_hyp_words(" ".join(pp_word), hyp_words, sent)
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
                    if w.head.dep_ in ["ccomp", "xcomp"]:
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word, pos_list, all_pos_list, pp_list,
                                                                           hyp_words, sent)
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
                    pp_str = process_hyp_words(pp_str, hyp_words, sent)
                    if (len(pp_str.split(" ")) == 1) | (pp_str.split(" ")[0] == "v") | (pp_str.split(" ")[-1] == "to"):
                        i += 1
                        continue
                    pp_str = get_complete_last_word(pp_str.split(" "), sent.split(" "))
                    if pp_str in sent:
                        s_idx = check_continuity(pp_word, s_word, -2)
                        if s_word[s_idx - 1] in dictionary["to"].keys():
                            pp_str = " ".join(s_word[s_idx - 2:s_idx]) + " " + pp_str
                            pp_word = s_word[s_idx - 2:s_idx] + pp_word
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
                                pp_list.append(('v', pp_str))
                            else:
                                pp_list.append(('p', pp_str))
                            pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
        i += 1

    return pp_list, noun_chunks


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
        pp_len = len(orig_pp[i][1].split(" "))

        if not maintain:
            maintain_flag = True
            count = 0
            for j in range(s_idx + 1, s_idx + pp_len):
                if res_label[j] == 1:
                    count += 1

            if maintain_flag:
                if count > (pp_len - 1) / 2:
                    res_label[s_idx] = 1
                    maintain = True

        for j in range(s_idx + 1, s_idx + pp_len):
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
    return res_label


## Check the integrity of sbar in the compression results
def check_sbar_integrity(comp_label, sbar_list, sbar_flag):
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
            count = 0
            for j in range(s_idx, s_idx + sbar_len):
                if comp_label[j] == 1:
                    count += 1
            if count > sbar_len / 2:
                for j in range(s_idx, s_idx + sbar_len):
                    res_label[j] = 1
            else:
                for j in range(s_idx, s_idx + sbar_len):
                    res_label[j] = -1
    return res_label

def check_formulation_intergrity(comp_label, for_list, for_flag):
    s_idx = -1
    res_label = list(comp_label)
    for i in range(len(for_list)):
        s_idx = for_flag.index(2, s_idx + 1)
        for_len = len(for_list[i].split(" "))
        count = 0
        for j in range(s_idx, s_idx + for_len):
            if comp_label[j] == 1:
                count += 1

        for j in range(s_idx, s_idx + for_len):
            if (count > for_len / 2) & (res_label[j] != -1):
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
        if (s.label() == "PP") & (s.leaves()[0] in ["while", "when"]):
            count += 1
        if count == 2:
            sub_string = process_hyp_words(" ".join(s.leaves()), hyp_words, orig_sent)
            sbar_word = long_sbar.split(sub_string)[0].strip().rstrip().split(" ")
            if len(sbar_word) == 1:
                long_sbar = process_hyp_words(long_sbar, hyp_words, orig_sent)
                sbar = long_sbar.strip().rstrip().split(" , ")[0]
                return sbar
            else:
                break
    sbar = ""
    if pos_list[0][1] in ['IN', 'WDT', 'WP', 'WRB', "WP$"]:
        sbar = process_hyp_words(long_sbar.split(sub_string)[0], hyp_words, orig_sent).strip().rstrip()
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
    remain_w = sbar_idx
    before_sbar = " ".join(s_words[0:sbar_idx])
    new_pp_list = list(pp_list)
    for i in range(len(pp_list)):
        p = pp_list[i]
        if p[1] in before_sbar:
            len_p = len(p[1].split(" "))
            remain_w = remain_w - len_p
            new_pp_list.pop(new_pp_list.index(p))
    return remain_w, new_pp_list


def check_that_clause(s_words, sbar, pos_list, dictionary, pp_list, hyp_words, orig_sent):
    # s_idx = get_phrase_idx(s_words, sbar)
    s_idx = check_continuity(sbar, s_words, -2)
    remain_w, new_pp_list = judge_that_in_start(s_words, s_idx, pp_list)
    if pos_list[s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        if remain_w < 6:
            sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
        else:
            sbar = process_hyp_words(s_words[s_idx - 1] + " " + "that", hyp_words, orig_sent)
        return True, sbar, new_pp_list
    elif (pos_list[s_idx - 1][1] in ['JJ']) & (
            pos_list[s_idx - 2][0] in ["is", "are", "am", "been", "'s", "'re", "be", "'m"]):
        if remain_w < 6:
            sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
        else:
            sbar = process_hyp_words(s_words[s_idx - 2] + " " + s_words[s_idx - 1] + " " + "that", hyp_words, orig_sent)
        return True, sbar, new_pp_list
    elif (pos_list[s_idx - 1][0] in dictionary['that'].keys()) & (pos_list[s_idx][0] not in dictionary['that'].keys()):
        if remain_w < 6:
            sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
        else:
            sbar = process_hyp_words(s_words[s_idx - 1] + " " + "that", hyp_words, orig_sent)

        return True, sbar, new_pp_list
    elif pos_list[s_idx][0] in dictionary['that'].keys():
        if remain_w < 6:
            sbar = process_hyp_words(" ".join(s_words[0:s_idx + 1]), hyp_words, orig_sent)
        else:
            sbar = process_hyp_words(s_words[s_idx], hyp_words, orig_sent)
        return True, sbar, new_pp_list
    else:
        return False, "", pp_list


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


def process_hyp_words(sent, hyp_words, orig_sent):
    wrong_hyp_words = []
    word_list = []
    for w in hyp_words:
        if w[0] in ["-", "–", "−"]:
            w = w.replace("-", "- ").replace("–", "– ").replace("−", "− ")
        elif w[-1] in ["-", "–", "−"]:
            w = w.replace("-", " -").replace("–", " –").replace("−", " −")
        else:
            w = w.replace("-", " - ").replace("–", " – ").replace("−", " − ")
        word_list.append(w.split())
        wrong_hyp_words.append(w)

    for i in range(len(hyp_words)):
        sent = sent.replace(wrong_hyp_words[i], hyp_words[i])
        idx = orig_sent.split(" ").index(hyp_words[i])
        if len(sent.split(" ")) >= 2:
            if (sent.split(" ")[-1] in ["-", "–"]) & (sent.split(" ")[-2] in word_list[i]):
                sent = " ".join(sent.split(" ")[:-1]) + hyp_words[i].split(" ")[-1]
            elif (sent.split(" ")[-1] in word_list[i]) & (sent.split(" ")[-2] == orig_sent.split(" ")[idx - 1]):
                if sent.split(" ")[-1] == word_list[i][0]:
                    sent = " ".join(sent.split(" ")[:-1]) + " " + hyp_words[i]

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


def extra_sbar(sent, dictionary, hyp_words):
    sbar_list = []
    nlp_tree = get_nlp_tree(sent)
    tree_words = nlp_tree.leaves()
    all_pos_list = nlp_tree.pos()
    sent_words = sent.split(" ")
    for s in nlp_tree.subtrees():
        label = s.label()
        pos_list = s.pos()
        if (label == "SBAR") | ((label == "PP") & (pos_list[0][0] in ["while", "when"])):
            tree_s_idx = check_continuity(s.leaves(), tree_words, -2)
            key_sent = " ".join(s.leaves()).replace("-LRB-", "(").replace("-RRB-", ")")
            ## uncomplete formulation should not in the starting index
            if (s.leaves()[0] in [">", "<"]) | (len(s.leaves()) == 1):
                continue
            for i in range(len(key_formulations)):
                if key_formulations[i] in key_sent:
                    if formulations[i][0] == "(":
                        key_sent = key_sent.replace(key_formulations[i], " " + formulations[i])
                    else:
                        key_sent = key_sent.replace(key_formulations[i], formulations[i])
            key_sent = key_sent.replace("( ", "(").replace(" )", ")")
            key_sent = process_hyp_words(key_sent, hyp_words, sent)
            key_sent = get_complete_last_word(key_sent.split(" "), sent.split(" "))
            key_words = key_sent.split(" ")
            ### 's
            if len(key_words) == 1:
                continue
            orig_s_idx = check_continuity(key_words[:-1], sent_words, -2)
            for f in formulations:
                f_w = f.split(" ")
                if f_w[0] == sent_words[orig_s_idx + len(key_words) - 1]:
                    if len(f_w) == 1:
                        key_words[-1] = f_w[0]
                    else:
                        if f_w[1] == sent_words[orig_s_idx + len(key_words)]:
                            key_words.extend(f_w[1:])
                    break

            if orig_s_idx >= 1:
                if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ']) & (
                        all_pos_list[tree_s_idx][1] not in ["IN", "WDT", "WP", "WP$", "WRB"]):
                    sbar = all_pos_list[tree_s_idx - 1][0]
                    if sbar in dictionary['that'].keys():
                        sbar_list.append(sbar)
                    continue
            if orig_s_idx >= 2:
                if key_words[0] == "as":
                    if sent_words[orig_s_idx - 2] == "as":
                        key_words = sent_words[orig_s_idx - 2:orig_s_idx] + key_words
                        pos_list = all_pos_list[tree_s_idx - 2:tree_s_idx] + pos_list

            if not exist_sbar(s):
                if (pos_list[0][1] in ["IN", "WDT", "WP", "WP$", "WRB", 'JJ', "PRP"]) | (
                        (pos_list[0][0] == "as") & (pos_list[0][1] == "RB")):
                    ## including what ...
                    if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (
                            pos_list[0][0] == "what"):
                        key_words.insert(0, all_pos_list[tree_s_idx - 1][0])

                    orig_s_idx = check_continuity(key_words, sent_words, -2)
                    for hw in hyp_words:
                        if sent_words[orig_s_idx + len(key_words) - 1] == hw:
                            key_words[-1] = hw
                            break

                    sbar = process_hyp_words(" ".join(key_words), hyp_words, sent)
                    sbar = process_wrong_formulation(sbar)
                    if len(sbar_list) > 0:
                        if sbar not in sbar_list[-1]:
                            sbar_list.append(sbar)
                    else:
                        sbar_list.append(sbar)
            else:
                long_sbar = process_hyp_words(" ".join(key_words), hyp_words, sent)
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

    return sbar_list, all_pos_list


def using_pp_update_sbar(sent, sbar_list, all_pos_list, dictionary, pp_list, hyp_words):
    for i in range(len(sbar_list)):
        sbar = process_hyp_words(sbar_list[i], hyp_words, sent)
        sbar_words = sbar.split(" ")
        if (sbar_words[0] == "that") | (sbar_words[0] in dictionary['that'].keys()):
            update_flag, sbar, pp_list = check_that_clause(sent.split(" "),
                                                           sbar_words, all_pos_list,
                                                           dictionary, pp_list, hyp_words, sent)
            if update_flag:
                sbar_list[i] = sbar

    for i in range(len(sbar_list) - 1):
        if sbar_list[i] in sbar_list[i + 1]:
            sbar_list.pop(i)
            break

    return sbar_list, pp_list


def filter_pp_in_sbar(sbar_list, pp_list):
    if len(sbar_list) > 0:
        res_pp = list(pp_list)
        for pp in pp_list:
            for sbar in sbar_list:
                if pp[1] in sbar:
                    if pp in res_pp:
                        res_pp.remove(pp)
        return res_pp
    else:
        return pp_list


def extract_conj(text):
    res = []
    # as well as单独处理
    doc = nlp(text)
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
                res.append(ans)
                ans = []
        else:
            pass
        j += 1
    return res


def single_conj(min, j, doc, ans):
    flag = 0
    str = ''
    choose_flag = 0
    if min != 0 and doc[min] == doc[j].head:
        for i in doc[min + 1:j + 1]:
            if i.text == 'and' or i.text == 'or':
                choose_flag = 1
            str += ' ' + i.text
        return j + 1
    for i in range(min, j + 1):
        if doc[i] == doc[j].head:
            flag = 1
        if flag == 1:
            if doc[i].text == 'and' or doc[i].text == 'or':
                choose_flag = 1
            str += ' ' + doc[i].text
    if choose_flag:
        print(str)
        ans.append(str.strip())
        ans.append(1)
    return j


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


def write_list_in_txt(comp_list, orig_comp, file_path):
    f = open(file_path, "w")
    for i in range(len(comp_list)):
        f.write("i = " + str(i) + "\n")
        f.write("original: " + orig_comp[i] + "\n")
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
                | (("(n" in cut_words[i]) & (juede_word_is_formulation(cut_words[i])))\
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

    if len(for_list) > 0:
        print(for_list)
    return for_list


## check completeness of prep phrases, clause
def check_grammar(cut_sents, comp_label, orig_sents):
    comp_list = []
    orig_comp = []
    label_list = []
    all_sbar = []
    all_pp = []
    all_conj = []
    dictionary = load_dictionary('./Dictionary.txt')
    for i in range(4623, len(cut_sents)):
        for_list = extra_formulation(cut_sents[i])
        cut_sents[i] = cut_sents[i].replace("``", "\"").replace("''", "\"")
        res_label = list(comp_label[i])
        hyp_words, spill_words_list = get_hyphen_word(cut_sents[i])
        abbr_words = get_abbr_word(cut_sents[i])
        sbar_list, pos_list = extra_sbar(cut_sents[i], dictionary, hyp_words)
        pp_list, noun_chunks = get_prep_list_by_dependency(cut_sents[i], hyp_words, spill_words_list, abbr_words)
        print("prep phrase: ", pp_list)
        sbar_list, new_pp_list = using_pp_update_sbar(cut_sents[i], sbar_list, pos_list, dictionary, pp_list,
                                                      hyp_words)
        print("sbar: ", sbar_list)
        cut_words = cut_sents[i].split(" ")
        if len(sbar_list) > 0:
            sbar_flag = [0] * len(cut_words)
            for sbar in sbar_list:
                sbar_words = sbar.split(" ")
                s_idx = check_continuity(sbar_words, cut_words, -2)
                if sbar_flag[s_idx] == 2:
                    s_idx = check_continuity(sbar_words, cut_words, s_idx)
                sbar_flag = fill_sent_flag(sbar_flag, s_idx, s_idx + len(sbar_words))
            # print("sbar_flag: ", sbar_flag)
            res_label = check_sbar_integrity(res_label, sbar_list, sbar_flag)
            # print("after sbar process: ", res_label)
        res_pp = filter_pp_in_sbar(sbar_list, new_pp_list)
        if len(res_pp) > 0:
            pp_flag = [0] * len(cut_words)
            for pp in res_pp:
                pp_words = pp[1].split(" ")
                s_idx = check_continuity(pp_words, cut_words, -2)
                pp_flag = fill_sent_flag(pp_flag, s_idx, s_idx + len(pp_words))
            # print("pp_flag: ", pp_flag)
            res_label = check_pp_integrity(cut_words, res_label, res_pp, pp_flag, noun_chunks)
            # print("after prep process: ", res_label)
        # res_label = check_comma(words, res_label)
        if len(for_list) > 0:
            for_flag = [0] * len(cut_words)
            for f in for_list:
                f_words = f.split(" ")
                s_idx = check_continuity(f_words, cut_words, -2)
                for_flag = fill_sent_flag(for_flag, s_idx, s_idx + len(f_words))
            res_label = check_formulation_intergrity(res_label, for_list, for_flag)

        conj_str = ''
        if len(res_pp) == 0:
            pp_flag = [0] * len(cut_words)
        for conj_i in range(len(res_label)):
            conj_str = conj_str + ' ' + cut_words[conj_i] if (
                    res_label[conj_i] != -1 and pp_flag[conj_i] == 0) else conj_str
        # print("conj_Str: ", conj_str)
        conj_res = extract_conj(conj_str.strip().rstrip())
        # print("conj_res: ", conj_res)
        for conj_li in conj_res:
            if conj_li[1] == 1:
                conj_word = conj_li[0].split(" ")
                conj_index = -1
                for temp in range(len(cut_words)):
                    if conj_word[0] == cut_words[temp] and conj_word == cut_words[temp:temp + len(conj_word)]:
                        conj_index = temp
                        break
                if not conj_index == -1:
                    #print("conj_index: ", conj_index, conj_index + len(conj_word),
                          # cut_words[conj_index:conj_index + len(conj_word)])
                    check_index = 0
                    for check_conj in range(conj_index, conj_index + len(conj_word)):
                        check_index += 1 if res_label[check_conj] == 1 else 0
                    if check_index == len(conj_word) - 1:
                        for check_conj in range(conj_index, conj_index + len(conj_word)):
                            res_label[check_conj] = 1
            elif conj_li[1] == 2:
                conj_word = conj_li[0].split(" ")
                conj_index = -1
                for temp in range(len(cut_words)):
                    if conj_word[0] == cut_words[temp] and conj_word == cut_words[temp:temp + len(conj_word)]:
                        conj_index = temp
                        break
                if not conj_index == -1:
                    #print("conj_index: ", conj_index, conj_index + len(conj_word),
                          # cut_words[conj_index:conj_index + len(conj_word)])
                    check_index = False
                    for check_conj in range(conj_index, conj_index + len(conj_word)):
                        if res_label[check_conj] == 1:
                            check_index = True
                            break
                    if check_index:
                        for check_conj in range(conj_index, conj_index + len(conj_word)):
                            res_label[check_conj] = 1
        # print("after conj process: ", res_label)

        if res_label.count(1) < 3:
            res_label = comp_label[i]

        orig_words = orig_sents[i].split(" ")
        cut_idx = search_cut_content(orig_words)
        if len(cut_idx) != 0:
            for tup in cut_idx:
                count = tup[1] - tup[0] + 1
                for j in range(count):
                    res_label.insert(tup[0], 0)
        if len(res_label) != len(orig_words):
            print("orig context: ", orig_words)


        orig_comp.append(get_res_by_label(cut_words, comp_label[i]))
        orig_res = get_res_by_label(orig_words, res_label)
        comp_list.append(orig_res)
        label_list.append(res_label)
        all_sbar.append(sbar_list)
        all_pp.append(pp_list)
        all_conj.append(conj_res)
    write_list_in_txt(comp_list, orig_comp, "./modify_res.txt")
    return label_list, all_sbar, all_pp, all_conj, comp_list


def grammar_check_main():
    cut_sent_path = "./comp_input/ncontext.cln.sent"
    orig_sent_path = "./comp_input/context.cln.sent"
    comp_label = load_label("./ncontext_result_greedy.sents")
    cut_sents = load_orig_sent(cut_sent_path)
    orig_sents = load_orig_sent(orig_sent_path)
    label_list, all_sbar, all_pp, all_conj, comp_list = check_grammar(cut_sents, comp_label, orig_sents)
    return label_list, all_sbar, all_pp, all_conj, comp_list


if __name__ == '__main__':
    file_name = "context"
    cut_sent_path = "./comp_input/ncontext.cln.sent"
    orig_sent_path = "./comp_input/context.cln.sent"
    comp_label = load_label("./ncontext_result_greedy.sents")
    cut_sents = load_orig_sent(cut_sent_path)
    orig_sents = load_orig_sent(orig_sent_path)
    check_grammar(cut_sents, comp_label, orig_sents)
