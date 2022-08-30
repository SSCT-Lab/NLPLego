from process_utils import *
from nltk import CoreNLPParser

eng_parser = CoreNLPParser('http://127.0.0.1:9000')

## obtain constituency parser tree
def get_nlp_tree(sent):
    sent = re.sub(r'%(?![0-9a-fA-F]{2})', "%25", sent)
    sent = sent.replace("+", "%2B")
    words = sent.split(" ")
    par_res = eng_parser.parse(words)
    for line in par_res:
        nlp_tree = line
    return nlp_tree

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

def devide_sbar(long_sbar, nlp_tree, hyp_words, orig_sent, all_pos_list, dictionary):
    count = 0
    for s in nlp_tree.subtrees():
        if s.label() == "SBAR":
            count += 1
        elif (s.label() == "PP") & (s.leaves()[0] in ["while", "when"]):
            count += 1
        if count == 2:
            #long_sbar = process_hyp_words(long_sbar, hyp_words, orig_sent, -1)
            sub_string = process_hyp_words(" ".join(s.leaves()), hyp_words, orig_sent, -1)
            if len(s.leaves()) == 1:
                sbar = long_sbar.strip().rstrip().split(" , ")[0]
                return sbar
            if not check_clause_type(orig_sent.split(" "), sub_string.split(" "), all_pos_list, dictionary, hyp_words):
                sbar_word = long_sbar.split(sub_string)[0].strip().rstrip().split(" ")
            else:
                sbar_word = long_sbar.split(" ")
            if sbar_word[-1] == ",":
                sbar_word = sbar_word[:-1]
            if len(sbar_word) == 1:
                sbar = long_sbar.strip().rstrip().split(" , ")[0]
                return sbar
            else:
                break
    # sbar = ""
    # if pos_list[0][1] in ['IN', 'WDT', 'WP', 'WRB', "WP$"]:
    sbar = " ".join(sbar_word).strip().rstrip()
    if sbar == "that":
        sbar = sbar + " " + sub_string
    # sbar = sbar.split(" , ")[0]
    if len(sbar.split(" ")) < 3:
        sbar = long_sbar

    return sbar

def extra_sbar(sent, nlp_tree, hyp_words, dictionary):
    sbar_list = []
    tree_words = nlp_tree.leaves()
    all_pos_list = nlp_tree.pos()
    sent_words = sent.split(" ")
    orig_s_idx = -1
    for s in nlp_tree.subtrees():
        label = s.label()
        pos_list = s.pos()
        if (label == "SBAR") | ((label == "PP") & (pos_list[0][0] in ["while", "when", "before", "after", "during"])):
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
            ## judge sbar
            sbar_flag = False
            if pos_list[0][1] in ["IN", "WDT", "WP", "WP$", "WRB", 'JJ', "PRP", "TO", "RB"]:
                sbar_flag = True
            elif (pos_list[0][0] == "as") & (pos_list[0][1] == "RB"):
                sbar_flag = True
            elif (pos_list[0][0] == "that") & (pos_list[0][1] == "DT"):
                sbar_flag = True
            elif "of" in key_words:
                start_index = key_words.index("of") + 1
                if pos_list[start_index][1] in ["IN", "WDT", "WP", "WP$", "WRB", 'JJ', "PRP", "TO"]:
                    sbar_flag = True

            if not exist_sbar(s):
                if sbar_flag:
                    ## including what ...
                    if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (
                            pos_list[0][0] in ["what", "how"]):
                        key_words.insert(0, all_pos_list[tree_s_idx - 1][0])

                    orig_s_idx = check_continuity(key_words, sent_words, -1)
                    for hw in hyp_words:
                        if orig_s_idx + len(key_words) - 1 == hw[0]:
                            key_words[-1] = hw[1]
                            break
                    sbar = process_hyp_words(" ".join(key_words), hyp_words, sent, orig_s_idx)
                    sbar = process_wrong_formulation(sbar)
                    if (key_words[0].lower() == "although") & (" , " in sbar):
                        sbar = sbar.split(" , ")[0]
                    sbar = cut_sub_sent_in_pp_sbar(sbar, sbar.split(" "), sbar.split(" ")[0])
                    if len(sbar_list) > 0:
                        if (sbar not in sbar_list[-1]) & (len(sbar.split(" ")) > 1):
                            sbar_list.append(sbar)
                    else:
                        if len(sbar.split(" ")) > 1:
                            sbar_list.append(sbar)
            else:
                if sbar_flag:
                    long_sbar = process_hyp_words(" ".join(key_words), hyp_words, sent, orig_s_idx)
                    if " as to " not in long_sbar:
                        sbar = devide_sbar(long_sbar, s, hyp_words, sent, all_pos_list, dictionary)
                    else:
                        sbar = long_sbar
                    if len(sbar.split(" ")) > 1:
                        if (all_pos_list[tree_s_idx - 1][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (
                                pos_list[0][0] == "what"):
                            sbar = all_pos_list[tree_s_idx - 1][0] + " " + sbar
                        sbar = process_wrong_formulation(sbar)
                        if (key_words[0].lower() == "although") & (" , " in sbar):
                            sbar = sbar.split(" , ")[0]
                        sbar = cut_sub_sent_in_pp_sbar(sbar, sbar.split(" "), sbar.split(" ")[0])
                        if len(sbar_list) > 0:
                            if (sbar not in sbar_list[-1]) & (len(sbar.split(" ")) > 1):
                                sbar_list.append(sbar)
                        else:
                            if len(sbar.split(" ")) > 1:
                                sbar_list.append(sbar)

    return sbar_list, all_pos_list


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


def extract_cc_by_constituent(tree, sent, orig_sent, hyp_words):
    print("sent: ", sent)
    cc_list = []
    sent_words = sent.split(" ")
    tree_positions, position_labels = group_by_level(tree)
    position_labels = sorted(position_labels.items(), key=lambda x: x[0], reverse=False)
    for item in position_labels:
        key = item[0]
        if " CC " in " ".join(item[1]):
            tree_list = tree_positions[key - 1]
            for i in range(len(position_labels[key - 1][1])):
                if position_labels[key - 1][1][i] != "S":
                    cc_tree = tree[tree_list[i]]
                    if not isinstance(cc_tree, str):
                        cc_tree_positions, cc_position_labels = group_by_level(cc_tree)
                        if "CC" in cc_position_labels[1]:
                            key_sent, key_words, orig_s_idx = format_tree_sent(cc_tree.leaves(), hyp_words, orig_sent,
                                                                                   sent_words, -1)
                            #print("CC: ", key_sent)
                            if key_sent != sent:
                                if " , " in key_sent:
                                    phrase_list = key_sent.split(" , ")
                                    if ("and" not in phrase_list[-1].split()) & ("or" not in phrase_list[-1].split()):
                                        key_sent = " , ".join(phrase_list[:-1])
                                if " : " in key_sent:
                                    phrase_list = key_sent.split(" : ")
                                    if ("and" not in phrase_list[-1].split()) & ("or" not in phrase_list[-1].split()):
                                        key_sent = " : ".join(phrase_list[:-1])
                                cc_list.append(key_sent)

    return cc_list


def del_adjuncts_in_cc(cc_list, sbar_list, res_pp, vp_list):
    for i in range(len(cc_list)):
        key_sent = cc_list[i]
        if len(sbar_list) > 0:
            for sbar in sbar_list:
                if (sbar[0] == "s") & (sbar[1] in key_sent):
                    key_sent = key_sent.replace(sbar[1], "").strip().rstrip()

        if len(res_pp) > 0:
            for pp in res_pp:
                if (pp[0] == "p") & (pp[2] not in ["of", "than"]) & (pp[1] in key_sent):
                    key_sent = key_sent.replace(pp[1], "").strip().rstrip()

        if len(vp_list) > 0:
            for vp in vp_list:
                if (vp[0] == "acl") & (vp[1] in key_sent):
                    key_sent = key_sent.replace(vp[1], "").strip().rstrip()
        cc_list[i] = key_sent
    return cc_list

def extract_pp_by_constituent(tree, sent, hyp_words):
    print("sent:", sent)
    pp_list = []
    sent_words = sent.split(" ")
    tree_positions, position_labels = group_by_level(tree)
    position_labels = sorted(position_labels.items(), key=lambda x: x[0], reverse=False)
    for item in position_labels:
        key = item[0]
        if "PP" in item[1]:
            tree_list = tree_positions[key]
            for i in range(len(item[1])):
                if item[1][i] == "PP":
                    pp_tree = tree[tree_list[i]]
                    pp_words = pp_tree.leaves()
                    last_tree_list = tree_positions[key - 1]
                    for j in range(len(position_labels[key - 1][1])):
                        parent_tree = tree[last_tree_list[j]]
                        if not isinstance(parent_tree, str):
                            parent_tree_positions,  parent_positions_labels = group_by_level(parent_tree)
                            if ("PP" in " ".join(parent_positions_labels[1])) & (
                                    " ".join(pp_words) in " ".join(parent_tree.leaves())):
                                print("position: ", key - 1, j, " type: ", position_labels[key - 1][1][j], ":", pp_words)


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
                    if len(key_words) < 2:
                        continue
                    if len(sent_list) > 0:
                        if key_sent in sent_list[-1]:
                            sent_list.pop()
                            # continue
                    # key_sent = " ".join(key_words)
                    sent_list.append(key_sent)
        if ("NP SBAR" in " ".join(item[1])) | (("NP , SBAR" in " ".join(item[1]))):
            tree_list = tree_positions[key - 1]
            for i in range(len(position_labels[key - 1][1])):
                if position_labels[key - 1][1][i] == "NP":
                    np_tree = tree[tree_list[i]]
                    if not isinstance(np_tree, str):
                        np_tree_positions, np_position_labels = group_by_level(np_tree)
                        if " ".join(np_position_labels[1]) in ["NP SBAR", "NP , SBAR"]:
                            key_sent, key_words, orig_s_idx = format_tree_sent(np_tree.leaves(), hyp_words, sent,
                                                                               sent_words, -1)
                            if len(key_words) < 3:
                                continue
                            if len(np_sbar_list) > 0:
                                if key_sent in np_sbar_list[-1]:
                                    sbar, sbar_words, sbar_s_idx = format_tree_sent(np_tree[np_tree_positions[1][-1]].leaves(), hyp_words, sent,
                                                                                    sent_words, -1)
                                    np_sbar_list[-1] = np_sbar_list[-1].split(sbar)[0].strip()
                                    # continue
                            np_sbar_list.append(key_sent)

                    else:
                        print("Not match condition")
                elif position_labels[key - 1][1][i] == "VP":
                    np_tree = tree[tree_list[i]]
                    if not isinstance(np_tree, str):
                        np_tree_positions, np_position_labels = group_by_level(np_tree)
                        if ("NP SBAR" in " ".join(np_position_labels[1])) | ("NP , SBAR" in " ".join(np_position_labels[1])):
                            key_sent, key_words, orig_s_idx = format_tree_sent(np_tree.leaves(), hyp_words, sent,
                                                                               sent_words, -1)
                            if len(key_words) < 3:
                                continue
                            if len(np_sbar_list) > 0:
                                if key_sent in np_sbar_list[-1]:
                                    continue
                            np_sbar_list.append(key_sent)
                            # break
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
                            key_sent, key_words, orig_s_idx = format_tree_sent(key_words, hyp_words, sent, sent_words,
                                                                               -1)
                            if len(np_pp_list) > 0:
                                if key_sent in np_pp_list[-1]:
                                    continue
                            np_pp_list.append(key_sent)
                    else:
                        print("Not match condition")
    return sent_list, np_sbar_list, np_pp_list


def get_child_tree(tree, sent, hyp_words):
    sent_words = sent.split(" ")
    tree_positions, position_labels = group_by_level(tree)
    child_tree_dict = []
    for key in tree_positions.keys():
        tree_list = tree_positions[key]
        for i in range(len(tree_list)):
            if not isinstance(tree[tree_list[i]], str):
                key_words = tree[tree_list[i]].leaves()
                key_sent, key_words, orig_s_idx = format_tree_sent(key_words, hyp_words, sent, sent_words, -1)
                if orig_s_idx != -1:
                    child_tree_dict.append((str(key) + " " + str(i) + " " + position_labels[key][i], key_sent))

    return child_tree_dict



