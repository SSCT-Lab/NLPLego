from grammar_check import *
import re
import nltk

zero_pattern = re.compile(r'z\d+')
sbar_pattern = re.compile(r's\d+')
pp_pattern = re.compile(r'p\d+|v\d+')
vp_pattern = re.compile(r'v\d+')
sub_pattern = re.compile(r'y\d+')
cc_pattern = re.compile(r'c\d+')
brackets_pattern = re.compile(r'b\d+')
ns_pattern = re.compile(r'ns\d+')
np_pattern = re.compile(r'np\d+')
all_pattern = re.compile(r'y\d+|c\d+|s\d+|p\d+|v\d+|b\d+|z\d+|np\d+|ns\d+')
eng_punctuation = ["–", "—", ";", ",", ".", ":", "--", "--", "-"]

def convert_label(s_words, comp_label):
    temp_words = []
    temp_adjuncts = []
    adjunct = ""
    for i in range(len(comp_label)):
        if comp_label[i] == 1:
            temp_words.append(s_words[i])
            if adjunct != "":
                temp_adjuncts.append(adjunct.strip().rstrip())
                adjunct = ""
        else:
            adjunct += s_words[i] + " "
            if s_words[i] in eng_punctuation:
                temp_words.append(s_words[i])
            else:
                temp_words.append("0")

    return temp_words, temp_adjuncts


def judge_divide(long_words, short_words):
    if " ".join(short_words) in " ".join(long_words):
        s_idx = check_continuity(short_words, long_words, -1)
        e_idx = s_idx + len(short_words)
        if e_idx != len(long_words):
            return False
        else:
            return True


def get_correct_sidx(orig_words, key_words, last_s_idx, cut_idx_list):
    orig_s_idx = check_continuity(key_words, orig_words, last_s_idx)
    if orig_s_idx != -1:
        return orig_s_idx, orig_s_idx + len(key_words) - 1
    else:
        lower_key_words = list(key_words)
        lower_key_words[0] = lower_key_words[0].lower()
        orig_s_idx = check_continuity(lower_key_words, orig_words, last_s_idx)
        if orig_s_idx != -1:
            return orig_s_idx, orig_s_idx + len(key_words) - 1
        start_indexs = list(filter(lambda x: orig_words[x] == key_words[0], list(range(len(orig_words)))))
        for idx in start_indexs:
            search_idx = 1
            for s_idx in range(idx + 1, len(orig_words)):
                skip = False
                for cut_idx in cut_idx_list:
                    if (s_idx >= cut_idx[0]) & (s_idx <= cut_idx[1]):
                        skip = True
                        break
                if skip:
                    continue
                if orig_words[s_idx] != key_words[search_idx]:
                    break
                else:
                    search_idx += 1
                if search_idx == len(key_words):
                    return idx, s_idx

def get_temp_adjunct(temp_words, rep_words, dictionary):
    slot = 0
    adjuncts = []
    adjunct = []
    s_idx = 0
    result = all_pattern.findall(temp_words[s_idx])
    while (len(result) == 0) & (temp_words[s_idx] != "0"):
        s_idx += 1
        if s_idx == len(temp_words) - 1:
            break
        result = all_pattern.findall(temp_words[s_idx])
    last_flag = temp_words[s_idx]
    for j in range(s_idx, len(temp_words)):
        if temp_words[j] not in eng_punctuation:
            result = all_pattern.findall(temp_words[j])
            if (len(result) != 0) | (temp_words[j] == "0"):
                if (temp_words[j] != last_flag) & (len(adjunct) != 0):
                    tag = nltk.pos_tag(rep_words[j - 1].split())
                    if (tag[0][1] not in ["IN", "TO", "DT"]) & (rep_words[j - 1] not in dictionary["end"]):
                        if adjunct[-1] in eng_punctuation:
                            adjunct = adjunct[:-1]
                        adjuncts.append(" ".join(adjunct).replace("\u200b", ""))
                        slot = slot + 1
                        adjunct = []
                last_flag = temp_words[j]
                temp_words[j] = "t" + str(slot)
                adjunct.append(rep_words[j])
            else:
                if len(adjunct) != 0:
                    if adjunct[-1] in eng_punctuation:
                        adjunct = adjunct[:-1]
                    adjuncts.append(" ".join(adjunct).replace("\u200b", ""))
                    slot = slot + 1
                    adjunct = []
        else:
            if rep_words[j] in [";", "–", "—", "--"]:
                temp_words[j] = rep_words[j]
                continue
            if len(adjunct) != 0:
                adjunct.append(rep_words[j])

    if len(adjunct) > 0:
        if adjunct[-1] in eng_punctuation:
            adjunct = adjunct[:-1]
        adjuncts.append(" ".join(adjunct).replace("\u200b", ""))

    return temp_words, adjuncts


def gen_temp_free_order(orig_sents, cut_sents, comp_labels, start_idx, end_idx):
    temp_list = []
    adjunct_list = []
    all_ner = []
    comp_list = []
    dictionary = load_dictionary('./Dictionary.txt')
    for i in range(start_idx, end_idx):
        orig_sent = orig_sents[i]
        cut_sent = cut_sents[i]
        s_words = orig_sent.replace("``", "\"").replace("''", "\"").split(" ")
        #print("original sentences: ", orig_sent)
        res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, sym_list, cc_sent_list, np_sbar_list, np_pp_list, child_tree_dict = grammar_check_one_sent(orig_sent, cut_sent, comp_labels[i], dictionary)
        comp_res = get_res_by_label(s_words, res_label)
        temp_words, temp_adjuncts = convert_label(s_words, res_label)
        cut_idx_list = search_cut_content(s_words)
        new_np_pp_list = list(np_pp_list)
        for np in np_pp_list:
            for j in range(len(pp_list)):
                if np in pp_list[j][1]:
                    new_np_pp_list.remove(np)
        np_pp_list = new_np_pp_list
        for sbar in sbar_list:
            parent_nodes = [node for node in child_tree_dict if sbar[1] in node[1]]
            if len(parent_nodes) != 0:
                print(parent_nodes[-1])
            else:
                print("not match")

        for pp in pp_list:
            parent_nodes = [node for node in child_tree_dict if pp[1] in node[1]]
            if len(parent_nodes) != 0:
                print(parent_nodes[-1])
            else:
                print("not match")
    #     if len(sym_list) > 0:
    #         last_s_idx = -1
    #         for j in range(len(sym_list)):
    #             sym_words = sym_list[j].split(" ")
    #             if "." == sym_words[-1]:
    #                 sym_words.pop()
    #             if len(cut_idx_list) != 0:
    #                 s_idx, e_idx = get_correct_sidx(s_words, sym_words, last_s_idx, cut_idx_list)
    #             else:
    #                 s_idx = check_continuity(sym_words, s_words, -1)
    #                 e_idx = s_idx + len(sym_words) - 1
    #             last_s_idx = s_idx + 1
    #             exist_flag = True
    #             for y in range(s_idx, e_idx + 1):
    #                 if (temp_words[y] != "0") & (temp_words[y] not in eng_punctuation):
    #                     exist_flag = False
    #                     break
    #             if exist_flag:
    #                 if s_words[s_idx - 1] in ["–", "—", ":", ";"]:
    #                     s_idx -= 1
    #                 for s in range(s_idx, e_idx + 1):
    #                     if s_words[s] in eng_punctuation:
    #                         temp_words[s] = s_words[s]
    #                     else:
    #                         temp_words[s] = "y" + str(j)
    #
    #         print("sym:", temp_words)
    #
    #     if len(cc_sent_list) != 0:
    #         last_s_idx = -1
    #         for j in range(len(cc_sent_list)):
    #             if len(cut_idx_list) != 0:
    #                 s_idx, e_idx = get_correct_sidx(s_words, cc_sent_list[j].split(" "), last_s_idx, cut_idx_list)
    #             else:
    #                 s_idx = check_continuity(cc_sent_list[j].split(" "), s_words, -1)
    #                 e_idx = s_idx + len(cc_sent_list[j].split(" ")) - 1
    #             last_s_idx = s_idx + 1
    #             exist_flag = True
    #             for c in range(s_idx, e_idx + 1):
    #                 if (temp_words[c] != "0") & (temp_words[c] not in eng_punctuation):
    #                     exist_flag = False
    #                     break
    #             if exist_flag:
    #                 for s in range(s_idx, e_idx + 1):
    #                     temp_words[s] = "c" + str(j)
    #
    #     if len(np_sbar_list) != 0:
    #         last_s_idx = -1
    #         for j in range(len(np_sbar_list)):
    #             sbar_len = len(np_sbar_list[j].split(" "))
    #             if sbar_len > 2:
    #                 if len(cut_idx_list) != 0:
    #                     s_idx, e_idx = get_correct_sidx(s_words, np_sbar_list[j].split(" "), last_s_idx, cut_idx_list)
    #                 else:
    #                     s_idx = check_continuity(np_sbar_list[j].split(" "), s_words, -1)
    #                     e_idx = s_idx + sbar_len - 1
    #                 last_s_idx = s_idx + 1
    #                 exist_flag = True
    #                 for s in range(s_idx, e_idx + 1):
    #                     result = all_pattern.findall(temp_words[s])
    #                     if (temp_words[s] not in eng_punctuation) & ((len(result) != 0) | (temp_words[s] != "0")):
    #                         exist_flag = False
    #                         break
    #                 if exist_flag:
    #                     for s in range(s_idx, e_idx + 1):
    #                         temp_words[s] = "ns" + str(j)
    #
    #     if len(sbar_list) != 0:
    #         last_s_idx = -1
    #         for j in range(len(sbar_list)):
    #             sbar_len = len(sbar_list[j][1].split(" "))
    #             if sbar_len > 2:
    #                 if len(cut_idx_list) != 0:
    #                     s_idx, e_idx = get_correct_sidx(s_words, sbar_list[j][1].split(" "), last_s_idx, cut_idx_list)
    #                 else:
    #                     s_idx = check_continuity(sbar_list[j][1].split(" "), s_words, -1)
    #                     e_idx = s_idx + sbar_len - 1
    #                 last_s_idx = s_idx + 1
    #                 exist_flag = True
    #                 for s in range(s_idx, e_idx + 1):
    #                     result = all_pattern.findall(temp_words[s])
    #                     if (temp_words[s] not in eng_punctuation) & ((len(result) != 0) | (temp_words[s] != "0")):
    #                         exist_flag = False
    #                         break
    #                 if exist_flag:
    #                     for s in range(s_idx, e_idx + 1):
    #                         temp_words[s] = "s" + str(j)
    #
    #     if len(np_pp_list) != 0:
    #         last_s_idx = -1
    #         for j in range(len(np_pp_list)):
    #             if len(cut_idx_list) != 0:
    #                 s_idx, e_idx = get_correct_sidx(s_words, np_pp_list[j].split(" "), last_s_idx, cut_idx_list)
    #             else:
    #                 s_idx = check_continuity(np_pp_list[j].split(" "), s_words, -1)
    #                 e_idx = s_idx + len(np_pp_list[j].split(" ")) - 1
    #             last_s_idx = s_idx + 1
    #             exist_flag = True
    #             for p in range(s_idx, e_idx + 1):
    #                 result = all_pattern.findall(temp_words[p])
    #                 if (temp_words[p] not in eng_punctuation) & ((len(result) != 0) | (temp_words[p] != "0")):
    #                     exist_flag = False
    #                     break
    #             if exist_flag:
    #                 if s_words[s_idx - 1] in ["and", "or", "but", "—", ":"]:
    #                     s_idx = s_idx - 1
    #                 for p in range(s_idx, e_idx + 1):
    #                     temp_words[p] = "np" + str(j)
    #
    #     if len(pp_list) != 0:
    #         last_s_idx = -1
    #         for j in range(len(pp_list)):
    #             if len(cut_idx_list) != 0:
    #                 s_idx, e_idx = get_correct_sidx(s_words, pp_list[j][1].split(" "), last_s_idx, cut_idx_list)
    #             else:
    #                 s_idx = check_continuity(pp_list[j][1].split(" "), s_words, -1)
    #                 e_idx = s_idx + len(pp_list[j][1].split(" ")) - 1
    #             last_s_idx = s_idx + 1
    #             exist_flag = True
    #             for p in range(s_idx, e_idx + 1):
    #                 result = all_pattern.findall(temp_words[p])
    #                 if (temp_words[p] not in eng_punctuation) & ((len(result) == 0) | (temp_words[p] != "0")):
    #                     exist_flag = False
    #                     break
    #             if exist_flag:
    #                 if s_words[s_idx - 1] in ["and", "or", "but", "—", ":"]:
    #                     s_idx = s_idx - 1
    #                 for p in range(s_idx, e_idx + 1):
    #                     if pp_list[j][0] == "p":
    #                         if j > 0:
    #                             if (pp_list[j - 1][1].split()[0].lower() == "from") & (
    #                                     pp_list[j][1].split()[0].lower() == "to") & (temp_words[s_idx - 1][0] == "p"):
    #                                 temp_words[p] = temp_words[s_idx - 1]
    #                             else:
    #                                 temp_words[p] = "p" + str(j)
    #                         else:
    #                             temp_words[p] = "p" + str(j)
    #                     else:
    #                         if (temp_words[s_idx - 1] != '0') | (s_words[s_idx - 1] in [","]):
    #                             temp_words[p] = "v" + str(j)
    #
    #     # ner need to maintan the same value
    #     if len(ner_list) > 0:
    #         last_s_idx = -1
    #         for j in range(len(ner_list)):
    #             s_idx = check_continuity(ner_list[j].split(" "), s_words, last_s_idx)
    #             last_s_idx = s_idx + 1
    #             e_idx = s_idx + len(ner_list[j].split(" ")) - 1
    #             tmp_flag = " ".join(temp_words[s_idx:e_idx + 1])
    #             result = all_pattern.findall(tmp_flag)
    #             if (temp_words[s_idx] == "0") | (len(result) != 0):
    #                 if len(result) != 0:
    #                     n_label = result[0]
    #                 else:
    #                     n_label = "0"
    #                 if s_idx > 0:
    #                     tag = nltk.pos_tag(s_words[s_idx - 1].split())
    #                     if s_words[s_idx - 1] in ["and", "or", "but", "—", ":", "the", "a", "an"]:
    #                         s_idx = s_idx - 1
    #                     elif tag[0][1] in ["ADJ", "RB"]:
    #                         s_idx = s_idx - 1
    #                 for n in range(s_idx, e_idx + 1):
    #                     if temp_words[n] != n_label:
    #                         temp_words[n] = n_label
    #         print("ner:", temp_words)
    #
    #     if len(for_list) > 0:
    #         print("for_list:", for_list)
    #         for j in range(len(for_list)):
    #             s_idx = check_continuity(for_list[j].split(" "), s_words, -1)
    #             e_idx = s_idx + len(for_list[j].split(" ")) - 1
    #             tmp_flag = " ".join(temp_words[s_idx:e_idx + 1])
    #             result = all_pattern.findall(tmp_flag)
    #             if (temp_words[s_idx] == "0") | (len(result) != 0):
    #                 if len(result) != 0:
    #                     f_label = result[0]
    #                 else:
    #                     f_label = "0"
    #                 for f in range(s_idx, e_idx + 1):
    #                     if temp_words[f] != f_label:
    #                         temp_words[f] = f_label
    #         print("for:", temp_words)
    #
    #     if len(cut_idx_list) != 0:
    #         for j in range(len(cut_idx_list)):
    #             if temp_words[cut_idx_list[j][1]] != temp_words[cut_idx_list[j][1] + 1]:
    #                 for b in range(cut_idx_list[j][0], cut_idx_list[j][1] + 1):
    #                     temp_words[b] = "b" + str(j)
    #
    #     print(temp_words)
    #     temp_words, adjuncts = get_temp_adjunct(temp_words, s_words, dictionary)
    #     print("temp: ", " ".join(temp_words))
    #     print("adjuncts: ", adjuncts)
    #     temp_list.append(" ".join(temp_words))
    #     adjunct_list.append(adjuncts)
    #     all_ner.append(ner_list)
    #     comp_list.append(comp_res)
    # return temp_list, adjunct_list, all_ner, comp_list


def gen_temp_in_order(orig_sents, cut_sents, comp_labels, start_idx, end_idx, dataset):
    temp_list = []
    adjunct_list = []
    all_ner = []
    all_for = []
    all_hyp_words = []
    comp_list = []
    dictionary = load_dictionary('./tools/Dictionary.txt')
    for i in range(start_idx, end_idx):
        print("i = ", i)
        orig_sent = orig_sents[i]
        cut_sent = cut_sents[i]
        rep_words = orig_sent.replace("``", "\"").replace("''", "\"").split(" ")
        #print("original sentences: ", orig_sent)
        res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, sym_list, cc_sent_list, np_sbar_list, np_pp_list, child_tree_dict = grammar_check_one_sent(orig_sent, cut_sent, comp_labels[i], dictionary, dataset)
        #print("original result:", get_res_by_label(rep_words, comp_labels[i]))
        print("modify result:", get_res_by_label(rep_words, res_label))
        hyp_words, spill_words_list = get_hyphen_word(orig_sent.replace("``", "\"").replace("''", "\""))
        comp_res = get_res_by_label(rep_words, res_label)
        temp_words, temp_adjuncts = convert_label(rep_words, res_label)
        cut_idx_list = search_cut_content(rep_words)
        zero_idx = []
        s_idx = -1
        temp_str = ""
        print("orig:", temp_words)
        for j in range(len(temp_words)):
            if (temp_words[j] == "0") | (temp_words[j] in [",", ".", ":", "!", "?"]):
                temp_str += temp_words[j] + " "
                if s_idx == -1:
                    s_idx = j
            else:
                if s_idx != -1:
                    if " 0" in temp_str:
                        while temp_words[s_idx] in [",", ".", ":", "!", "?"]:
                            s_idx += 1
                        zero_idx.append((s_idx, j - 1))
                    s_idx = -1
                    temp_str = ""

        if (s_idx != -1) & (" 0" in temp_str):
            while temp_words[s_idx] in eng_punctuation:
                s_idx += 1
            zero_idx.append((s_idx, len(temp_words) - 2))

        print(zero_idx)

        if len(sym_list) > 0:
            last_s_idx = -1
            for j in range(len(sym_list)):
                sym_words = sym_list[j].split(" ")
                if "." == sym_words[-1]:
                    sym_words.pop()
                if len(cut_idx_list) != 0:
                    s_idx, e_idx = get_correct_sidx(rep_words, sym_words, last_s_idx, cut_idx_list)
                else:
                    s_idx = check_continuity(sym_words, rep_words, -1)
                    e_idx = s_idx + len(sym_words) - 1
                last_s_idx = s_idx + 1
                exist_flag = True
                for y in range(s_idx, e_idx + 1):
                    if (temp_words[y] != "0") & (temp_words[y] not in eng_punctuation):
                        exist_flag = False
                        break
                if exist_flag:
                    if rep_words[s_idx - 1] in ["–", "—", ":", ";"]:
                        s_idx -= 1
                    for s in range(s_idx, e_idx + 1):
                        if rep_words[s] in eng_punctuation:
                            temp_words[s] = rep_words[s]
                        else:
                            temp_words[s] = "y" + str(j)

            print("sym:", temp_words)

        if len(cc_sent_list) != 0:
            last_s_idx = -1
            for j in range(len(cc_sent_list)):
                if len(cut_idx_list) != 0:
                    s_idx, e_idx = get_correct_sidx(rep_words, cc_sent_list[j].split(" "), last_s_idx, cut_idx_list)
                else:
                    s_idx = check_continuity(cc_sent_list[j].split(" "), rep_words, -1)
                    e_idx = s_idx + len(cc_sent_list[j].split(" ")) - 1
                last_s_idx = s_idx + 1
                exist_flag = True
                for c in range(s_idx, e_idx + 1):
                    result = sub_pattern.findall(temp_words[c])
                    if (temp_words[c] != "0") & (temp_words[c] not in eng_punctuation) & (len(result) == 0):
                        exist_flag = False
                        break

                if exist_flag:
                    for c in range(s_idx, e_idx + 1):
                        if rep_words[c] in eng_punctuation:
                            temp_words[c] = rep_words[c]
                        else:
                            temp_words[c] = "c" + str(j)

        if len(sbar_list) > 0:
            last_s_idx = -1
            for j in range(len(sbar_list)):
                if (len(sbar_list[j][1].split(" ")) > 2) & (sbar_list[j][0] == "s"):
                    if len(cut_idx_list) != 0:
                        s_idx, e_idx = get_correct_sidx(rep_words, sbar_list[j][1].split(" "), last_s_idx, cut_idx_list)
                    else:
                        s_idx = check_continuity(sbar_list[j][1].split(" "), rep_words, -1)
                        e_idx = s_idx + len(sbar_list[j][1].split(" ")) - 1
                    last_s_idx = s_idx + 1
                    exist_flag = True
                    for s in range(s_idx, e_idx + 1):
                        result = cc_pattern.findall(temp_words[s]) + sub_pattern.findall(temp_words[s])
                        if (temp_words[s] not in eng_punctuation) & (len(result) == 0) & (temp_words[s] != "0"):
                            exist_flag = False
                            break
                    if exist_flag:
                        last_word = s_idx - 1
                        while rep_words[last_word] == ",":
                            last_word -= 1
                        tag = nltk.pos_tag(rep_words[last_word].split(" "))
                        if tag[0][1] in ["IN", "TO"]:
                            continue
                        if rep_words[s_idx - 1] in ["and", "or", "but", "—", ":", "the", "a", "an"]:
                            s_idx = s_idx - 1
                        for s in range(s_idx, e_idx + 1):
                            if rep_words[s] in eng_punctuation:
                                temp_words[s] = rep_words[s]
                            else:
                                temp_words[s] = "s" + str(j)
            print("sbar:", temp_words)

        if len(vp_list) > 0:
            last_s_idx = -1
            for j in range(len(vp_list)):
                if vp_list[j][0] == "acl":
                    if len(cut_idx_list) != 0:
                        s_idx, e_idx = get_correct_sidx(rep_words, vp_list[j][1].split(" "), last_s_idx, cut_idx_list)
                    else:
                        s_idx = check_continuity(vp_list[j][1].split(" "), rep_words, -1)
                        e_idx = s_idx + len(vp_list[j][1].split(" ")) - 1
                    exist_flag = True
                    for v in range(s_idx, e_idx + 1):
                        result = all_pattern.findall(temp_words[v])
                        if (temp_words[v] not in eng_punctuation) & (temp_words[v] != "0") & (len(result) == 0):
                            exist_flag = False
                            break

                    if exist_flag:
                        for v in range(s_idx, e_idx + 1):
                            if rep_words[v] in eng_punctuation:
                                temp_words[v] = rep_words[v]
                            else:
                                temp_words[v] = "v" + str(j)

        if len(pp_list) > 0:
            last_s_idx = -1
            for j in range(len(pp_list)):
                if len(cut_idx_list) != 0:
                    s_idx, e_idx = get_correct_sidx(rep_words, pp_list[j][1].split(" "), last_s_idx, cut_idx_list)
                else:
                    s_idx = check_continuity(pp_list[j][1].split(" "), rep_words, last_s_idx)
                    e_idx = s_idx + len(pp_list[j][1].split(" ")) - 1
                last_s_idx = s_idx + 1
                exist_flag = True
                for p in range(s_idx, e_idx + 1):
                    result = all_pattern.findall(temp_words[p])
                    if (temp_words[p] != "0") & (temp_words[p] not in eng_punctuation) & (len(result) == 0):
                        exist_flag = False
                        break
                    if len(result) > 0:
                        if result[0][0] == "v":
                            exist_flag = False
                            break
                if exist_flag:
                    if rep_words[s_idx - 1] in ["and", "or", "but", "—", ":", "the", "a", "an"]:
                        s_idx = s_idx - 1
                    if (pp_list[j][0] == "p") & (not ((pp_list[j][1].split(" ")[0] != "of") & (pp_list[j][2] == "of"))):
                        for p in range(s_idx, e_idx + 1):
                            if rep_words[p] in eng_punctuation:
                                temp_words[p] = rep_words[p]
                                continue
                            if j > 0:
                                if (pp_list[j - 1][1].split()[0].lower() == "from") & (pp_list[j][1].split()[0].lower() == "to") & (temp_words[s_idx - 1][0] == "p"):
                                    temp_words[p] = temp_words[s_idx - 1]
                                else:
                                    temp_words[p] = "p" + str(j)
                            else:
                                temp_words[p] = "p" + str(j)
                    else:
                        if (temp_words[s_idx] != temp_words[e_idx]) & (temp_words[s_idx] not in eng_punctuation):
                            for p in range(s_idx, e_idx + 1):
                                if rep_words[p] in eng_punctuation:
                                    temp_words[p] = rep_words[p]
                                else:
                                    temp_words[p] = temp_words[s_idx]
            print("pp:", temp_words)

        if len(ner_list) > 0:
            for j in range(len(ner_list)):
                s_idx = check_continuity(ner_list[j].split(" "), rep_words, -1)
                e_idx = s_idx + len(ner_list[j].split(" ")) - 1
                tmp_flag = " ".join(temp_words[s_idx:e_idx + 1])
                result = all_pattern.findall(tmp_flag)
                if (temp_words[s_idx] == "0") | (len(result) != 0):
                    if len(result) != 0:
                        n_label = result[0]
                    else:
                        n_label = "0"
                    if s_idx > 0:
                        tag = nltk.pos_tag(rep_words[s_idx - 1].split())
                        if rep_words[s_idx - 1] in ["and", "or", "but", "—", ":", "the", "a", "an"]:
                            s_idx = s_idx - 1
                        elif tag[0][1] in ["ADJ", "RB"]:
                            s_idx = s_idx - 1
                    for n in range(s_idx, e_idx + 1):
                        if temp_words[n] != n_label:
                            temp_words[n] = n_label
            print("ner:", temp_words)

        if len(for_list) > 0:
            print("for_list:", for_list)
            for j in range(len(for_list)):
                s_idx = check_continuity(for_list[j].split(" "), rep_words, -1)
                e_idx = s_idx + len(for_list[j].split(" ")) - 1
                tmp_flag = " ".join(temp_words[s_idx:e_idx + 1])
                result = all_pattern.findall(tmp_flag)
                if (temp_words[s_idx] == "0") | (len(result) != 0):
                    if len(result) != 0:
                        f_label = result[0]
                    else:
                        f_label = "0"
                    for f in range(s_idx, e_idx + 1):
                        if temp_words[f] != f_label:
                            temp_words[f] = f_label
            print("for:", temp_words)

        if len(cut_idx_list) > 0:
            for j in range(len(cut_idx_list)):
                if temp_words[cut_idx_list[j][1]] != temp_words[cut_idx_list[j][1] + 1]:
                    for b in range(cut_idx_list[j][0], cut_idx_list[j][1] + 1):
                        temp_words[b] = "b" + str(j)
            print("brackets:", temp_words)

        inpp_list = [w for w in dictionary["integrated"] if w in orig_sent]
        if len(inpp_list) != 0:
            for in_pp in inpp_list:
                s_idx = check_continuity(in_pp.split(" "), rep_words, -1)
                e_idx = s_idx + len(in_pp.split(" ")) - 1
                tmp_flag = " ".join(temp_words[s_idx:e_idx + 1])
                result = all_pattern.findall(tmp_flag)
                if (temp_words[s_idx] == "0") | (len(result) != 0):
                    if len(result) != 0:
                        i_label = result[0]
                    else:
                        i_label = "0"
                    for p in range(s_idx, e_idx + 1):
                        temp_words[p] = i_label

        for tup in zero_idx:
            s_idx = tup[0]
            result = pp_pattern.findall(temp_words[s_idx]) + vp_pattern.findall(temp_words[s_idx]) + sbar_pattern.findall(temp_words[s_idx])
            sbar_last_flag = ''
            pp_last_flag = ''
            while (len(result) == 0) & (s_idx < tup[1]):
                s_idx += 1
                result = pp_pattern.findall(temp_words[s_idx]) + vp_pattern.findall(temp_words[s_idx]) + sbar_pattern.findall(temp_words[s_idx])
                cc_sub_res = sub_pattern.findall(temp_words[s_idx]) + cc_pattern.findall(temp_words[s_idx])
                cc_sbar_sub_res = cc_sub_res + sbar_pattern.findall(temp_words[s_idx])
                if len(cc_sub_res) != 0:
                    sbar_last_flag = cc_sub_res[0]
                if len(cc_sbar_sub_res) != 0:
                    pp_last_flag = cc_sbar_sub_res[0]
                if temp_words[s_idx] == '0':
                    sbar_last_flag = '0'
                    pp_last_flag = '0'
            last_flag = temp_words[s_idx]
            last_s_idx = s_idx
            for j in range(s_idx + 1, tup[1] + 1):
                if temp_words[j] not in eng_punctuation:
                    last_result = pp_pattern.findall(last_flag) + vp_pattern.findall(last_flag) + sbar_pattern.findall(last_flag)
                    if (temp_words[j] != last_flag) & (temp_words[j] not in eng_punctuation) & (len(last_result) != 0):
                        temp_str = " ".join(temp_words[j:tup[1] + 1])
                        if ((last_s_idx == tup[0]) & (temp_words[last_s_idx - 1] not in [";", "–", "—"])):
                            cc_sub_res = sub_pattern.findall(last_flag) + cc_pattern.findall(last_flag)
                            cc_sbar_sub_res = cc_sub_res + sbar_pattern.findall(last_flag)
                            if len(cc_sub_res) != 0:
                                sbar_last_flag = cc_sub_res[0]
                            if len(cc_sbar_sub_res) != 0:
                                pp_last_flag = cc_sbar_sub_res[0]
                            if last_flag == '0':
                                sbar_last_flag = '0'
                                pp_last_flag = '0'
                            last_s_idx = j
                            last_flag = temp_words[j]
                            continue
                        result = sub_pattern.findall(temp_str) + sbar_pattern.findall(temp_str) + cc_pattern.findall(
                            temp_str)
                        if (" 0" in " ".join(temp_words[j:tup[1] + 1])) | (len(result) != 0):
                            if len(result) != 0:
                                if last_result[0][0] == "s":
                                    if sbar_last_flag != '':
                                        if sbar_last_flag in result:
                                            flag = sbar_last_flag
                                        else:
                                            last_s_idx = j
                                            last_flag = temp_words[j]
                                            continue
                                    else:
                                        last_s_idx = j
                                        last_flag = temp_words[j]
                                        continue
                                else:
                                    if pp_last_flag != '':
                                        if pp_last_flag in result:
                                            flag = pp_last_flag
                                        else:
                                            last_s_idx = j
                                            last_flag = temp_words[j]
                                            continue
                                    else:
                                        last_s_idx = j
                                        last_flag = temp_words[j]
                                        continue
                            else:
                                flag = "0"
                            last_s_idx = temp_words.index(last_flag)
                            for z in range(last_s_idx, j):
                                if temp_words[z] not in eng_punctuation:
                                    temp_words[z] = flag

                            last_s_idx = j
                    else:
                        cc_sub_res = sub_pattern.findall(temp_words[j]) + cc_pattern.findall(temp_words[j])
                        cc_sbar_sub_res = cc_sub_res + sbar_pattern.findall(temp_words[j])
                        if len(cc_sub_res) != 0:
                            sbar_last_flag = cc_sub_res[0]
                        if len(cc_sbar_sub_res) != 0:
                            pp_last_flag = cc_sbar_sub_res[0]
                        if temp_words[s_idx] == '0':
                            sbar_last_flag = '0'
                            pp_last_flag = '0'
                    last_flag = temp_words[j]
        print("update: ", temp_words)

        ### check 0
        update_zero_idx = []
        s_idx = -1
        temp_str = ""
        for j in range(len(temp_words)):
            if (temp_words[j] == "0") | (temp_words[j] in eng_punctuation):
                temp_str += temp_words[j] + " "
                if s_idx == -1:
                    s_idx = j
            else:
                if s_idx != -1:
                    if " 0" in temp_str:
                        while temp_words[s_idx] in eng_punctuation:
                            s_idx += 1
                        update_zero_idx.append((s_idx, j - 1))
                    s_idx = -1
                    temp_str = ""

        if (s_idx != -1) & (" 0" in temp_str):
            while temp_words[s_idx] in eng_punctuation:
                s_idx += 1
            update_zero_idx.append((s_idx, len(temp_words) - 2))

        print(update_zero_idx)

        for tup in update_zero_idx:
            result = all_pattern.findall(temp_words[tup[1] + 1])
            if (tup[1] - tup[0] < 2) & (len(result) != 0):
                for z in range(tup[0], tup[1] + 1):
                    temp_words[z] = result[0]

        cc_indexs = list(filter(lambda x: rep_words[x] in ["and", "but", "or"], list(range(len(rep_words)))))
        for idx in cc_indexs:
            if (temp_words[idx] == "0") & (temp_words[idx] != temp_words[idx + 1]):
                if len(all_pattern.findall(temp_words[idx + 1])) != 0:
                    temp_words[idx] = temp_words[idx + 1]

        print(temp_words)
        temp_words, adjuncts = get_temp_adjunct(temp_words, rep_words, dictionary)
        print("temp: ", " ".join(temp_words))
        print("adjuncts: ", adjuncts)
        print(" ")
        temp_list.append(" ".join(temp_words))
        adjunct_list.append(adjuncts)
        all_ner.append(ner_list)
        all_for.append(for_list)
        all_hyp_words.append(hyp_words)
        comp_list.append(comp_res)
    return temp_list, adjunct_list, all_ner, all_for, all_hyp_words, comp_list


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


def gen_sent_temp_main(file_name, label_path, start_idx, end_idx, dataset):
    orig_sent_path = "./comp_input/" + file_name + ".cln.sent"
    orig_sents = load_orig_sent(orig_sent_path)
    cut_sent_path = "./comp_input/n" + file_name + ".cln.sent"
    cut_sents = load_orig_sent(cut_sent_path)
    comp_labels = load_label(label_path)
    temp_list, adjunct_list, ner_list, for_list, hyp_words_list, comp_list = gen_temp_in_order(orig_sents, cut_sents, comp_labels, start_idx, end_idx, dataset)
    return temp_list, adjunct_list, ner_list, for_list, hyp_words_list, comp_list


if __name__ == '__main__':
    #file_name = "sst"
    file_name = "context"
    dataset = "squad"
    #label_path = "./comp_res/w_nsst_result_greedy.sents"
    label_path = "./comp_res/ncontext_result_greedy.sents"
    start_idx = 5182
    end_idx = 6318
    gen_sent_temp_main(file_name, label_path, start_idx, end_idx, dataset)
    # temp_list, adjunct_list, comp_list, ner_list = gen_sent_temp_main(file_name, start_idx, end_idx)
    # sent_path = "./comp_input/" + file_name + ".cln.sent"
    # comp_label = load_label("./comp_label/slahan_w_syn/2_" + file_name + "_result_greedy.sents")
    # orig_sents = load_orig_sent(sent_path)
    # label_list, all_sbar, all_pp, all_conj, comp_list = check_grammar(orig_sents, comp_label)
    # temp_list, adjunct_list = gen_temp(orig_sents, label_list, all_sbar, all_pp)
    # gen_step_sentence(temp_list, adjunct_list, comp_list, file_name)
    # save_temp_adjuncts(temp_list, adjunct_list, file_name)
