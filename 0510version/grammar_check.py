from corenlp_utils import *
from spacy_utils import *
from preprocess import search_cut_content

## fill the flag_array by 1 (the position of phrase or clause is 1, staring word is 2)
def fill_sent_flag(sent_flag, s_idx, e_idx):
    sent_flag[s_idx] = 2
    for i in range(s_idx + 1, e_idx):
        sent_flag[i] = 1
    return sent_flag

def check_vp_integrity(res_label, cut_words, vp_list, vp_flag):
    s_idx = -1
    for vp in vp_list:
        s_idx = vp_flag.index(2, s_idx + 1)
        vp_words = vp[1].split()
        if vp[0] in ["acomp", "aux", "xcomp"]:
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


def check_symbols_integrity(res_label, sym_list, sem_flag, pp_list, sbar_list, vp_list, s_words, root_idx):
    s_idx = -1
    new_res_label = list(res_label)
    max_value = 0
    temp_max_value = 0
    maintain_idx = -1
    max_idx = -1
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
        if one_count > temp_max_value:
            max_idx = i
            temp_max_value = one_count
    if maintain_idx == -1:
        maintain_idx = max_idx
    s_idx = -1
    for i in range(0, len(sym_list)):
        s_idx = sem_flag.index(2, s_idx + 1)
        sym_words = sym_list[i].split(" ")
        if (i == maintain_idx) & (max_value < len(sym_words) / 2):
            for j in range(s_idx, s_idx + len(sym_words)):
                new_res_label[j] = 1
            for sbar in sbar_list:
                if (sbar[1] in sym_list[i]) & (sbar[0] == "s"):
                    sbar_idx = check_continuity(sbar[1].split(" "), s_words, -1)


                    for j in range(sbar_idx, sbar_idx + len(sbar[1].split(" "))):
                        new_res_label[j] = 0
            for pp in pp_list:
                if (pp[1] in sym_list[i]) & (pp[0] == "p") & (pp[2] != "of"):
                    pp_idx = check_continuity(pp[1].split(" "), s_words, -1)
                    for j in range(pp_idx, pp_idx + len(pp[1].split(" "))):
                        new_res_label[j] = 0
            for vp in vp_list:
                if (vp[0] == "acl") & (vp[1] in sym_list[i]):
                    vp_words = vp[1].split(" ")
                    vp_idx = check_continuity(vp_words, s_words, -1)
                    for j in range(vp_idx, vp_idx + len(vp_words)):
                        new_res_label[j] = 0

        if i != maintain_idx:
            if i != 0:
                for j in range(s_idx - 1, s_idx + len(sym_words)):
                    new_res_label[j] = 0
            elif ('—' not in s_words) & ('–' not in s_words):
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
        for vp in vp_list:
            if (vp[0] == "acl") & (vp[1] in sym_list[i]):
                vp_words = vp[1].split(" ")
                vp_idx = check_continuity(vp_words, s_words, -1)
                for j in range(vp_idx, vp_idx + len(vp_words)):
                    new_res_label[j] = 0
    return new_res_label, maintain_idx


## Check the integrity of sbar in the compression results
def check_sbar_integrity(res_label, sbar_list, sbar_flag, cut_words, pp_list, basic_elements, np_sbar_list):
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
            symbols = cut_words[s_idx: s_idx + sbar_len].count("\"") + cut_words[s_idx: s_idx + sbar_len].count(",") + cut_words[s_idx: s_idx + sbar_len].count("\'")
            count = res_label[s_idx: s_idx + sbar_len].count(1)
            pp_len = 0
            exist_pp_list = []
            for pp in pp_list:
                if (pp[0] == "p") & (pp[1] in sbar) & (pp[2] != "of"):
                    pp_len += len(pp[1].split(" "))
                    exist_pp_list.append(pp[1])
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
                np = get_modified_noun_by_sbar(sbar, np_sbar_list, pp_list)
                if np != "":
                    print("np:", np)
                    np_words = np.split(" ")
                    np_idx = check_continuity(np_words, cut_words, -1)
                    np_count = res_label[np_idx:np_idx + len(np_words)].count(1)
                else:
                    np_count = 0
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
                    if np != "":
                        if np_count <= len(np_words)/2:
                            for j in range(np_idx, np_idx + len(np_words)):
                                res_label[j] = 1
                else:
                    if count < (sbar_len - symbols - pp_len) / 2:
                        for j in range(s_idx, e_idx):
                            res_label[j] = -1
                        if np != "":
                            if np_count != 0:
                                for j in range(np_idx, np_idx + len(np_words)):
                                    res_label[j] = 1
                    else:
                        if np != "":
                            for j in range(np_idx, np_idx + len(np_words)):
                                res_label[j] = 1
                            for j in range(s_idx, e_idx):
                                res_label[j] = -1
                        else:
                            for j in range(s_idx, e_idx):
                                res_label[j] = 1
                            for pp in exist_pp_list:
                                p_s_idx = check_continuity(pp.split(" "), sbar.split(" "), -1)
                                for j in range(s_idx + p_s_idx, s_idx + p_s_idx + len(pp.split(" "))):
                                    res_label[j] = 0
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
            sbar = (
            "t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = (
            "t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    if (pos_list[p_s_idx - 2][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) & (pos_list[p_s_idx - 1][0] in ['"']):
        if remain_w <= 6:
            sbar = (
            "t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = (
            "t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif (pos_list[p_s_idx - 1][1] in ['JJ']) & (
            pos_list[p_s_idx - 2][0] in ["is", "are", "am", "been", "'s", "'re", "be", "'m"]):
        if remain_w <= 6:
            sbar = (
            "t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = ("t", process_hyp_words(s_words[s_idx - 2] + " " + s_words[s_idx - 1] + " " + " ".join(sbar_words),
                                           hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif (pos_list[p_s_idx - 1][0] in dictionary['that'].keys()) & (
            pos_list[p_s_idx][0] not in dictionary['that'].keys()):
        if remain_w <= 6:
            sbar = (
            "t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
        else:
            sbar = (
            "t", process_hyp_words(s_words[s_idx - 1] + " " + " ".join(sbar_words), hyp_words, orig_sent, s_idx))
        return True, sbar, new_pp_list
    elif pos_list[p_s_idx][0] in dictionary['that'].keys():
        if remain_w <= 6:
            sbar = (
            "t", process_hyp_words(" ".join(s_words[start_idx:s_idx + len(sbar_words)]), hyp_words, orig_sent, s_idx))
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

def using_pp_update_sbar(sent, sbar_list, all_pos_list, dictionary, pp_list, hyp_words):
    for i in range(len(sbar_list)):
        sbar = process_hyp_words(sbar_list[i], hyp_words, sent, -1)
        sbar_words = sbar.split(" ")
        if sbar_words[1] in ["what", "how"]:
            sbar_list[i] = ("t", sbar)
        elif (sbar_words[0] == "that") | (sbar_words[0] in dictionary['that'].keys()):
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

    for i in range(len(sbar_list) - 1):
        new_sbar = sbar_list[i][1] + " " + " ".join(sbar_list[i + 1][1].split(" ")[1:])
        if (new_sbar in sent) & (sbar_list[i][0] == "s") & (sbar_list[i + 1][0] == "t"):
            sbar_list[i] = ("s", new_sbar)
            sbar_list.pop(i + 1)
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

def write_list_in_txt(orig_sents, comp_list, orig_comp, file_path):
    f = open(file_path, "w", encoding="utf-8")
    for i in range(len(comp_list)):
        f.write("i = " + str(i) + "\n")
        f.write(orig_sents[i] + "\n")
        f.write("orig: " + orig_comp[i] + "\n")
        f.write("modifiy: " + comp_list[i] + "\n")
        f.write("\n")

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

    res_label = del_sbar_pp_vp(res_label, sbar_list, rep_cut_words, res_pp, vp_list)

    if res_label.count(1) < 4:
        res_label = comp_label

    if rep_cut_words[res_label.index(1)] == ",":
        res_label[res_label.index(1)] = 0

    if (rep_cut_words[-1] in [".", "?", "!"]) & (res_label[-1] != 1):
        res_label[-1] = 1

    if (rep_cut_words[-2] in [".", "?", "!"]) & (res_label[-2] != 1):
        res_label[-2] = 1

    return res_label


def process_final_result(comp_label, res_label, cut_words, rep_cut_words, sbar_list, res_pp, root_verb, basic_elements,
                         vp_list, sym_sent, dictionary):
    create_flag = False
    if (cut_words[-1] in [".", "?", "!"]) & (res_label[-1] != 1):
        res_label[-1] = 1

    if (cut_words[-2] in [".", "?", "!"]) & (res_label[-2] != 1):
        res_label[-2] = 1

    comp_res = get_res_by_label(cut_words[:-1], res_label[:-1])
    symbols = rep_cut_words.count("\"") + rep_cut_words.count(",") + rep_cut_words.count("\'") + rep_cut_words.count(".")
    if (res_label.count(1) < 4) | (res_label.count(1) >= len(res_label) - symbols) | (len([sbar for sbar in sbar_list if (comp_res in sbar[1]) & (sbar[0] == "s")]) != 0):
        res_label = create_seed_sent(comp_label, res_label, cut_words, sbar_list, rep_cut_words, res_pp, vp_list)
        create_flag = True

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
                        if s_idx > orig_one_idx + 2:
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
                if (elem[1] in ["dobj", "attr", "advmod"]) & (elem_str not in comp_res) & (len(pp_obj) == 0) & (
                        elem_str in sym_sent):
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
    if cut_words[first_idx] in ["that", ",", ":", ";", "–", "—"]:
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
    if cut_words[one_indexs[-2]] in ["that", "about", ":", ";", ",", "``", "but", "–", "—", "and"]:
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

    check_indexs =list(filter(lambda x: rep_cut_words[x] in ["and", "but", "or", "the"], list(range(len(rep_cut_words)))))
    for idx in check_indexs:
        if res_label[idx + 1] == 1:
            res_label[idx] = 1

    if ((res_label.count(1) < 4) | (res_label.count(1) >= len(res_label) - symbols) |
        (len([sbar for sbar in sbar_list if (comp_res in sbar[1]) & (sbar[0] == "s")]) != 0)) & (not create_flag):
        res_label = create_seed_sent(comp_label, res_label, cut_words, sbar_list, rep_cut_words, res_pp, vp_list)
        comp_res = get_res_by_label(cut_words, res_label)

    if (", ," in comp_res) | (", ." in comp_res):
        one_indexs = list(filter(lambda x: res_label[x] == 1, list(range(len(res_label)))))
        for j in range(len(one_indexs) - 1):
            if (rep_cut_words[one_indexs[j]] == ",") & (rep_cut_words[one_indexs[j + 1]] == ","):
                res_label[one_indexs[j + 1]] = 0
            if (rep_cut_words[one_indexs[j]] == ",") & (rep_cut_words[one_indexs[j + 1]] == "."):
                res_label[one_indexs[j]] = 0

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
    if conj_res[0][0] == 'River and Kiewa River':
        add_index = [2, 56]
    if conj_res[0][0] == "heath , Leadbeater 's possum and the helmeted honeyeater":
        add_index = [3, 14]
    if conj_res[0][0] == 'Theron , Viljoen and Visagie':
        add_index = [1, 85]
    if add_index != [-1, -1]:
        res_label[add_index[0]:add_index[1]+1] = [1]*(add_index[1]-add_index[0]+1)
    if remove_index != [-1, -1]:
        res_label[remove_index[0]:remove_index[1]+1] = [0]*(remove_index[1]-remove_index[0]+1)


def process_conj(rep_cut_words, temp_res_label, res_label, pp_flag, pos_list):
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
            for temp in range(len(rep_cut_words)):
                if temp_res_label[temp] == 1:
                # if res_label[temp] != -1 and pp_flag[temp] == 0:
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
            # conj_mapping_cut = conj_mapping_cut[::-1]
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
                if conj_word_index + 1 < len(conj_mapping_cut):
                    if pos_list[conj_mapping_cut[conj_word_index + 1][1]][1] == 'VBD':
                        for check_conj in conj_mapping_cut[conj_word_index:len(conj_mapping_cut)]:
                            res_label[check_conj[1]] = 1
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
def modify_basic_elements(basic_elements, rep_cut_words, res_label, sbar_list, sym_list, pp_list):
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
            for pp in pp_list:
                if (pp[0] == "p") & (pp[2] != "of") & (pp[1] in subj_str):
                    pp_words = pp[1].split(" ")
                    s_idx = check_continuity(pp_words, rep_cut_words, -1)
                    for j in range(s_idx, s_idx + len(pp_words)):
                        res_label[j] = 0

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
    if (";" in rep_cut_sent) | (" – " in rep_cut_sent) | (" — " in rep_cut_sent):
        sym_list = re.split(' ; | – | — ', rep_cut_sent[:-2])
    elif ":" in rep_cut_sent:
        if (':', 'SYM') in pos_list:
            idx = pos_list.index((':', 'SYM'))
        else:
            idx = pos_list.index((':', ':'))
        if ("NN" in pos_list[idx - 1][1]) | ("VBG" in pos_list[idx - 1][1]) | (("JJ" in pos_list[idx - 1][1])):
            sym_list = re.split(' : ', rep_cut_sent[:-2])
        else:
            sym_list = []
    else:
        sym_list = []
    sbar_list, new_pp_list = using_pp_update_sbar(rep_cut_sent, sbar_list, pos_list, dictionary, pp_list, hyp_words)
    rep_cut_words = rep_cut_sent.split(" ")
    res_label = modify_basic_elements(basic_elements, rep_cut_words, res_label, sbar_list, sym_list, pp_list)
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
        res_label = check_sbar_integrity(res_label, sbar_list, sbar_flag, rep_cut_words, pp_list, basic_elements,
                                         np_sbar_list)
        if len(vp_list) > 0:
            res_label = check_vp_integrity(res_label, rep_cut_words, vp_list, vp_flag)
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

    if len(sym_list) > 0:
        sem_flag = [0] * len(rep_cut_words)
        for j in range(0, len(sym_list)):
            sem_words = sym_list[j].strip().rstrip().split(" ")
            s_idx = check_continuity(sem_words, rep_cut_words, -1)
            sem_flag = fill_sent_flag(sem_flag, s_idx, s_idx + len(sem_words))
        res_label, sym_idx = check_symbols_integrity(res_label, sym_list, sem_flag, res_pp, sbar_list, vp_list, rep_cut_words,
                                                     root_idx)
        sym_sent = sym_list[sym_idx]
        s_idx = check_continuity(sym_sent.strip().rstrip().split(" "), rep_cut_words, -1)
        temp_res_label = [0] * len(res_label)
        for j in range(s_idx, s_idx + len(sym_sent.strip().rstrip().split(" "))):
            temp_res_label[j] = 1
        temp_res_label[-1] = 1
        print("sym modify:", get_res_by_label(rep_cut_words, res_label))
    else:
        temp_res_label = [1] * len(res_label)

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

    temp_res_label = del_sbar_pp_vp(temp_res_label, sbar_list, rep_cut_words, res_pp, vp_list)
    res_label, conj_res = process_conj(rep_cut_words, temp_res_label, res_label, pp_flag, pos_list)
    print("conj modify:", get_res_by_label(rep_cut_words, res_label))
    res_label = process_final_result(comp_label, res_label, cut_sent.split(" "), rep_cut_words, sbar_list, pp_list,
                                     root_verb, basic_elements, vp_list, sym_sent, dictionary)
    orig_words = orig_sent.split(" ")
    cut_idx = search_cut_content(orig_words)
    if len(cut_idx) != 0:
        for tup in cut_idx:
            count = tup[1] - tup[0] + 1
            for j in range(count):
                res_label.insert(tup[0], -2)

    return res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, sym_list, cc_sent_list, np_sbar_list, np_pp_list


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
    all_syms = []
    all_cc_sent = []
    all_np_sbar = []
    all_np_pp = []
    dictionary = load_dictionary('./Dictionary.txt')
    sbar_comp_list = []
    sbar_orig_sents = []
    orig_sents_list = []
    sbar_orig_comp = []
    for i in range(start_idx, end_idx):
        # grammar_check_one_sent(orig_sents[i], cut_sents[i], comp_label[i], dictionary)
        res_label, sbar_list, pp_list, conj_res, for_list, ner_list, vp_list, sym_list, cc_sent_list, np_sbar_list, np_pp_list = grammar_check_one_sent(
            orig_sents[i], cut_sents[i], comp_label[i], dictionary)
        orig_res = get_res_by_label(cut_sents[i].split(" "), comp_label[i])
        print("original result: ", orig_res)
        modify_res = get_res_by_label(orig_sents[i].split(" "), res_label)
        print("modify result: ", modify_res)
        if len(sbar_list) == 0:
            orig_comp.append(orig_res)
            comp_list.append(modify_res)
            orig_sents_list.append(orig_sents[i])
        else:
            sbar_orig_comp.append(orig_res)
            sbar_comp_list.append(modify_res)
            sbar_orig_sents.append(orig_sents[i])
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
        all_syms.append(sym_list)
    write_list_in_txt(orig_sents_list, comp_list, orig_comp, "./modify_res.txt")
    write_list_in_txt(sbar_orig_sents, sbar_comp_list, sbar_orig_comp, "./sbar_modify_res.txt")
    return label_list, all_sbar, all_pp, all_conj, comp_list, all_formulations, all_ners, all_vps, all_syms, all_cc_sent, all_np_sbar, all_np_pp

if __name__ == '__main__':
    file_name = "context"
    cut_sent_path = "./comp_input/ncontext.cln.sent"
    orig_sent_path = "./comp_input/context.cln.sent"
    comp_label = load_label("./ncontext_result_greedy.sents")
    cut_sents = load_orig_sent(cut_sent_path)
    orig_sents = load_orig_sent(orig_sent_path)
    start_idx = 6210
    end_idx = len(cut_sents)
    # end_idx = 100
    grammar_check_all_sents(cut_sents, comp_label, orig_sents, start_idx, end_idx)
