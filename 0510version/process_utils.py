import re

from preprocess import load_formulation, format_formulation

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
            if key not in ["comp", "start", "end", "verb", "integrated"]:
                dictionary[key] = {}
            else:
                dictionary[key] = []
        elif len(line) > 1:
            if key not in ["comp", "start", "end", "verb", "integrated"]:
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


## Complement prep phrase
def get_the_complete_phrase(p_word, h_word, s_word, pp, pos_list, all_pos_list, pp_list, hyp_words, orig_sent,
                            last_s_idx):
    new_pp = list(pp)
    new_pos_list = list(pos_list)
    if " ".join(pp) not in " ".join(s_word):
        p_idx = s_word.index(p_word)
    else:
        p_idx = check_continuity(pp, s_word, last_s_idx)
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
    else:
        n_p_idx = s_word.index(p_word)
        if n_p_idx < p_idx:
            idx = p_idx - 1
            while idx != n_p_idx - 1:
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

    return hyphen_words, word_list


## St.
def get_abbr_word(sent):
    abbr_words = []
    s_word = sent.split(" ")
    for w in s_word:
        if ("." in w) & (len(w) != 1):
            abbr_words.append(w)
    return abbr_words


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


def del_sbar_in_phrase(words, hyp_words, sent):
    str = process_hyp_words(" ".join(words), hyp_words, sent, -1).replace(
        "et al .", "et al.")
    str = re.split(' ; | – | — | , | who | which | that | where | when | why ', str)[0]
    if len(str) > 2:
        if str[-2:] == " ,":
            str = str[:-2]
    return str


def get_res_by_label(words, comp_label):
    res_words = []
    for i in range(len(words)):
        if comp_label[i] == 1:
            res_words.append(words[i])
    comp_res = " ".join(res_words)
    # print("final result: ", comp_res)
    return comp_res


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

    sort_alpha_ner_list = sorted(alpha_ner_list, key=lambda i: len(i.split(" ")), reverse=True)
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
    elif (" + " not in sent) & ("+" in ner) & ("+ " in sent):
        ner = ner.replace(" + ", "+ ")
    puncts = ["/", "-", "–", "−"]
    for p in puncts:
        if (" " + p + " " not in sent) & (p in ner) & (p in sent):
            ner = ner.replace(" " + p + " ", p)
    ner = ner.replace("St .", "St.")
    return ner


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
    if "of" in words:
        of_indexs = list(filter(lambda x: words[x] == "of", list(range(len(words)))))
        for idx in of_indexs:
            if (tmp_words[idx - 1] != "#") & (tmp_words[idx + 1] != "#"):
                tmp_words[idx] = words[idx]
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


def del_sbar_pp_vp(res_label, sbar_list, rep_cut_words, res_pp, vp_list):
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
            if (pp[0] == "p") & (pp[2] not in ["of", "than"]):
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


def get_modified_noun_by_sbar(sbar, np_sbar_list, pp_list):
    np = ""
    match_flag = False
    for np_sbar in np_sbar_list:
        if sbar in np_sbar:
            np = np_sbar.split(sbar)[0].strip()
            if len(np) == 0:
                np = ""
                break
            if np[-1] == ",":
                np = np[:-2]
            print("match")
            print(sbar)
            print(np_sbar)
            match_flag = True
            break

    if match_flag:
        for pp in pp_list:
            if (pp[1] in np) & (pp[0] == "p"):
                np = np.split(pp[1])[0].strip()

    return np
