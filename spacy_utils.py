import spacy
from process_utils import *

spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.add_pipe("merge_entities")
punctions = ["–", "—", ";", ",", ".", ":", "\""]
def get_verb_phrases(sent, hyp_words, spill_words_list, sbar_list):
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
            basic_elements.append((i, token.dep_, token.text, token.pos_, token.text))
            root_verb = token.text
            root_idx = i
            break
        i += 1

    if (root_idx == -1) & (("VERB" in pos_list) | ("AUX" in pos_list)):
        root_idx = 0
        while pos_list[root_idx] not in ["VERB", "AUX"]:
            root_idx += 1
        root_verb = doc[root_idx].text
        basic_elements.append((root_idx, "ROOT", root_verb, doc[root_idx].pos_, token.head.text))

    i = 0
    for token in doc:
        if ("subj" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (
                "ROOT" in [token.head.dep_, token.head.head.dep_]):
            subj_words = [tok.orth_ for tok in token.subtree]
            subj_str = del_sbar_in_phrase(subj_words, hyp_words, sent)
            basic_elements.append((i, token.dep_, token.text, subj_str, token.head.text))
        if ("expl" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (
                "ROOT" in [token.head.dep_, token.head.head.dep_]):
            basic_elements.append((i, token.dep_, token.text, token.pos_, token.head.text))
        if ("advmod" in token.dep_) & (token.head.dep_ == "ROOT") & (token.text in ["here", "there"]):
            if root_idx < i:
                adv_words = s_word[root_idx:i + 1]
            else:
                adv_words = s_word[i: root_idx + 1]
            adv_str = del_sbar_in_phrase(adv_words, hyp_words, sent)
            basic_elements.append((i, token.dep_, adv_str, token.pos_, token.head.text))
        if ("obj" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                obj_words = s_word[root_idx:i + 1]
            else:
                obj_words = s_word[i: root_idx + 1]
            obj_str = del_sbar_in_phrase(obj_words, hyp_words, sent)
            basic_elements.append((i, token.dep_, obj_str, token.pos_, token.head.text))

        if ("dative" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                dat_words = s_word[root_idx:i + 1]
            else:
                dat_words = s_word[i: root_idx + 1]
            dat_str = del_sbar_in_phrase(dat_words, hyp_words, sent)
            basic_elements.append((i, token.dep_, dat_str, token.pos_, token.head.text))

        if ("comp" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
                comp_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
                if (basic_elements[-1][2] not in comp_str) | (basic_elements[-1][1] == "ROOT"):
                    basic_elements.append((i, token.dep_, comp_str, token.pos_, token.head.text))
                    # print("comp_str:", comp_str)
        if ("attr" in token.dep_) & (token.head.pos_ in ["VERB", "AUX"]) & (token.head.dep_ == "ROOT"):
            if root_idx < i:
                vp_words = s_word[root_idx:i + 1]
            else:
                vp_words = s_word[i: root_idx + 1]
            attr_str = process_hyp_words(" ".join(vp_words), hyp_words, sent, -1).replace("et al .", "et al.")
            if basic_elements[-1][2] not in attr_str:
                basic_elements.append((i, token.dep_, attr_str, token.pos_, token.head.text))
            # print("attr_str:", attr_str)
        key = token.text + "-" + str(i)
        if key not in dep_map.keys():
            dep_map[key] = []
        dep_map[key].append((token.head.text, token.head.pos_, token.dep_, token.text))
        i += 1

    subj_list = [elem for elem in basic_elements if ((("subj" in elem[1]) | ("expl" in elem[1])) & (elem[0] < root_idx))]

    if len(subj_list) == 2:
        new_subj = subj_list[0][3] + " " + subj_list[1][3]
        if check_continuity(new_subj.split(), sent.split(), -1) != -1:
            basic_elements.remove(subj_list[0])
            basic_elements.remove(subj_list[1])
            basic_elements.append((subj_list[0][0], subj_list[0][1], subj_list[0][2], new_subj, subj_list[0][4]))


    noun_subj_elem = get_subj_by_noun(root_idx, pos_list, sbar_list, doc, hyp_words, sent)

    if len([elem for elem in basic_elements if ((("subj" in elem[1]) | ("expl" in elem[1])) & (elem[0] < root_idx))]) == 0:
        basic_elements.append(noun_subj_elem)

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
                acl_str = del_sbar_in_phrase(acl_word, hyp_words, sent)
                save_flag = True
                if acl_word[0] == w.text:
                    for j in range(len(vp_list)):
                        vp = vp_list[j]
                        if vp[1] in acl_str:
                            save_flag = False
                            break

                        elif acl_str in vp[1]:
                            new_acl = vp[1].split(acl_str)[0].strip()
                            if new_acl != '':
                                vp_list[j] = (vp[0], new_acl)
                            else:
                                save_flag = False
                            break

                    if save_flag & (acl_str != ''):
                        vp_list.append(("acl", acl_str))

        if (w.dep_ == "xcomp") & (w.pos_ == "VERB"):
            network = [t.text for t in list(w.children)]
            if len(network) != 0:
                xcomp_word = [tok.orth_ for tok in w.subtree]
                for f in formulations:
                    nf = f.replace(" ", "")
                    if xcomp_word[-1] == nf[:len(xcomp_word[-1])]:
                        vp_for = xcomp_word[-1]
                        for idx in range(i + len(xcomp_word), len(s_word)):
                            vp_for = vp_for + s_word[idx]
                            if vp_for == nf:
                                comp_f = f
                                xcomp_word[-1] = comp_f
                                break
                        break
                xcomp_str = del_sbar_in_phrase(xcomp_word, hyp_words, sent)
                save_flag = True
                if xcomp_word[0] == w.text:
                    for vp in vp_list:
                        if vp[1] in xcomp_str:
                            save_flag = False
                            break
                    if save_flag & (xcomp_str != ''):
                        vp_list.append(("xcomp", xcomp_str))

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
                                                                                     sent, -1).replace("et al .",
                                                                                                       "et al."))
                                    if temp[1] in sent:
                                        vp_list.pop()
                                        vp_list.append(temp)
                                        continue
                            save_flag = True
                            for vp in vp_list:
                                if vp[1] in aco_str:
                                    save_flag = False
                                    break
                            if save_flag & (aco_str != ''):
                                vp_list.append(("acomp", aco_str))

                        if (dep_re[2] in ["auxpass", "aux"]) & (dep_re[0] in s_word[i:]):
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
                            if save_flag & (pass_str != ''):
                                vp_list.append(("aux", pass_str))

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
                                            vp_list[-1][0],
                                            last_vp + " " + process_hyp_words(" ".join(vp_word), hyp_words,
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
                            if save_flag & (oprd_str != ''):
                                vp_list.append(("oprd", oprd_str))
        i += 1

    return vp_list, basic_elements, root_verb, root_idx, noun_subj_elem

## Determine whether the prepositional phrase contains a prepositional phrase
def exist_pp(pos_list, pp_word, dictionary, key_pp, to_flag, spill_words_list):
    count = 0
    key = -1
    sent = " ".join(pp_word).replace(" - ", "-").replace(" – ", "–")
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    i = 0
    p_idx = pp_word.index(key_pp)
    if p_idx + 1 < len(pp_word):
        if pp_word[p_idx + 1] in ["-", "–"]:
            p_idx = pp_word.index(key_pp, p_idx + 1)
    if p_idx - 1 >= 0:
        if pp_word[p_idx - 1] in ["-", "–"]:
            p_idx = pp_word.index(key_pp, p_idx + 1)
    comp_flag = False
    for pp in dictionary["comp"]:
        if pp in " ".join(pp_word).lower():
            if key_pp.lower() == pp.split(" ")[0]:
                p_idx = check_continuity(pp.split(" "), pp_word, -1)
                if p_idx == -1:
                    pp = pp[0].upper() + pp[1:]
                    p_idx = check_continuity(pp.split(" "), pp_word, -1)
                p_idx = p_idx + len(pp.split(" ")) - 1
                break
            else:
                if key_pp in pp.split(" ")[1:]:
                    comp_flag = True

    first_to = -1
    if key_pp == "from":
        if "to" in pp_word[p_idx:]:
            first_to = pp_word[p_idx:].index("to")
    for w in doc:
        if i > p_idx:
            if (pos_list[i] == "ADP") & (w.text not in ["of", "v", "than"]):
                if (i - 1 >= 0) & (i + 1 < len(pp_word)) & (w.text in spill_words_list):
                    if (pp_word[i - 1] in ["-", "–", "−"]) | (pp_word[i + 1] in ["-", "–", "−"]):
                        i += 1
                        continue
                network = [t.text for t in list(w.children)]
                if len(network) != 0:
                    if w.head.pos_ in ["VERB", "AUX"]:
                        ## ing modify noun
                        if w.head.dep_ == "acl":
                            count += 1
                            key = pp_word.index(w.head.text)
                            if (pos_list[key - 1] == "PRON") | (key - 1 == p_idx):
                                count -= 1
                                i += 1
                                continue
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
                            i += 1
                            continue
                    elif w.text == "as":
                        if "as" in pp_word[:i]:
                            as_idx = i - 1
                            while pp_word[as_idx] != "as":
                                as_idx -= 1
                            if pos_list[as_idx] == "ADV":
                                i += 1
                                continue
                        elif pp_word[i - 1] == "such":
                            count += 1
                            i -= 1
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
                if i - 1 != p_idx:
                    count += 1
            elif (w.text == "to") & (not to_flag):
                if ((key_pp == "from") & ((first_to + p_idx) == i)) | (pos_list[i - 1] == "VERB"):
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
    if count == 1:
        if key - 1 != p_idx:
            return True, key, comp_flag

    return False, key, comp_flag

## obtain the prefix of "of" phrase
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
                            while (all_pos_list[search_idx:i].count("NUM") == 0) & (search_idx > 0):
                                search_idx = search_idx - 1
                            if all_pos_list[search_idx:i].count("NUM") != 0:
                                prep_of.append(" ".join(s_word[search_idx:i]))
                            else:
                                while (all_pos_list[search_idx:i].count("DET") == 0) & (search_idx > 0):
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
    if e_idx + 1 < len(s_word):
        if (s_word[e_idx + 1] == "\"") & (pp_word.count("\"") % 2 != 0):
            pp_word.append(s_word[e_idx + 1])
            pos_list.append(all_pos_list[e_idx + 1])


    return pp_word, pos_list, comp_f, comp_abbr


## obtain all prepositional phrases in one sentence by dependency relation
def get_prep_list_by_dependency(sent, hyp_words, spill_words_list, abbr_words, basic_elems):
    # print(sent)
    pp_list = []
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    dictionary = load_dictionary("./tools/Dictionary.txt")
    # noun_chunks = []
    all_pos_list = [tok.pos_ for tok in doc]
    s_word = [tok.text for tok in doc]
    pp_flag = [0] * len(sent.split(" "))
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
        if (w.pos_ == "ADP") | (w.text in ["to"]) & (" ".join(s_word[i - 2:i + 1]) not in dictionary["comp"]):
            network = [t.text for t in list(w.children)]
            if len(network) != 0:
                pp_word = [tok.orth_ for tok in w.subtree]
                pos_list = [tok.pos_ for tok in w.subtree]
                if pp_word[-1] in ["and", "or"]:
                    pp_word.pop(len(pp_word) - 1)
                    pos_list.pop(len(pos_list) - 1)

                if (w.text == "between") & ("," in pp_word):
                    c_idx = pp_word.index(",")
                    pp_word = pp_word[:c_idx]
                    pos_list = pos_list[:c_idx]

                pp_str = process_hyp_words(" ".join(pp_word), hyp_words, sent, last_s_idx)
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
                    i += 1
                    continue
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
                                #s_idx = check_continuity(last_pp_word, s_word, last_s_idx - 1)
                                s_idx = check_continuity(last_pp_word, sent.split(" "), last_s_idx - 1)
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
                        if h_idx != -1:
                            vp_flag = True
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                            pos_list = old_pos_list
                            vp_flag = False
                else:
                    ## save verb phrase
                    if (w.head.pos_ in ["VERB", "AUX"]) & (w.dep_ in ["prep", "agent"]):
                        if (pp_word[0] not in ["during", "after", "before", "via", "due", "among"]) & (
                                "in order " not in " ".join(pp_word)):
                            pp_word, pos_list, h_idx = get_the_complete_phrase(w.text, w.head.text, s_word, pp_word,
                                                                               pos_list, all_pos_list, pp_list,
                                                                               hyp_words,
                                                                               sent, last_s_idx)
                            if h_idx != -1:
                                vp_flag = True
                            if pp_flag[h_idx] == 1:
                                pp_word = alternative
                                pos_list = old_pos_list
                                vp_flag = False
                            if len([prep for prep in dictionary["verb"] if prep == " ".join(pp_word[:2])]) != 0:
                                vp_flag = False
                        else:
                            if w.head.dep_ == "acl":
                                tmp_str = w.head.text + " " + w.text
                                if tmp_str in sent:
                                    v_idx = check_continuity(tmp_str.split(" "), s_word, -1)
                                    pp_word.insert(0, w.head.text)
                                    pos_list.insert(0, all_pos_list[v_idx])
                                    pp_flag[v_idx] = 0
                                    vp_flag = False
                    elif (w.head.pos_ in ["ADV", "ADJ", "NOUN"]) & (w.head.dep_ in ["advmod", "attr", "acomp", "conj", "npadvmod"]) \
                            & (w.head.head.pos_ in ["VERB", "AUX"]) & (" ".join(pp_word) != "as well as"):
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word,
                                                                           pos_list, all_pos_list, pp_list, hyp_words,
                                                                           sent, last_s_idx)
                        if h_idx != -1:
                            vp_flag = True
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                            pos_list = old_pos_list
                            vp_flag = False
                    elif (w.head.pos_ in ["ADJ", "ADV"]) & (w.text == "than"):
                        s_idx = check_continuity(pp_word, s_word, last_s_idx)
                        h_idx = s_idx
                        while s_word[h_idx] not in [w.head.text, ","]:
                            h_idx -= 1
                            pp_word.insert(0, s_word[h_idx])
                            pos_list.insert(0, all_pos_list[h_idx])
                        if s_word[h_idx - 1] in ["more", "less"]:
                            pp_word.insert(0, s_word[h_idx - 1])
                            pos_list.insert(0, all_pos_list[h_idx - 1])
                        vp_flag = True

                    ## process on Monday...in March
                    if (alternative[0] in ["on", "in"]) & (len(alternative) == 2):
                        if (alternative[1] in dictionary["on"].keys()) | (alternative[1] in dictionary["in"].keys()):
                            pp_str = " ".join(alternative)
                            pp_list.append(('p', pp_str, w.text))
                            #s_idx = check_continuity(alternative, s_word, last_s_idx)
                            s_idx = check_continuity(alternative, sent.split(" "), last_s_idx)
                            #pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
                            pp_flag = fill_pp_flag(pp_str, sent.split(" "), pp_flag, s_idx)
                            i += 1
                            continue
                    if (w.text == "as") & (alternative[0] in dictionary["as"].keys()):
                        pp_word = alternative
                        pos_list = old_pos_list
                        if h_idx != -1:
                            vp_flag = False

                if (w.text == "as") & (pp_word[0] == "as") & (pp_word.count("as") == 1):
                    if "as" in s_word[:i]:
                        as_idx = i - 1
                        while s_word[as_idx] != "as":
                            as_idx -= 1
                        if (all_pos_list[as_idx] == "ADV") & (as_idx > last_s_idx):
                            pp_word = s_word[as_idx:i] + pp_word
                            pos_list = all_pos_list[as_idx:i] + pos_list
                        if "ADJ" in all_pos_list[as_idx + 1 : i]:
                            vp_flag = True

                if (pp_word[0] == "to") & (s_word[i - 1] in dictionary["to"].keys()):
                    pp_word.insert(0, s_word[i - 1])
                    pos_list.insert(0, all_pos_list[i - 1])
                    vp_flag = True

                if (pp_word[0] == "from") & (s_word[i - 1] in dictionary["from"].keys()):
                    pp_word.insert(0, s_word[i - 1])
                    pos_list.insert(0, all_pos_list[i - 1])

                ## cut long prep
                flag, key, comp_flag = exist_pp(pos_list, pp_word, dictionary, w.text, False, spill_words_list)
                if flag & (key > 1):
                    if pp_word[key - 1] in ["and", "or", "but"]:
                        pp_word = pp_word[:key - 1]
                    else:
                        pp_word = pp_word[:key]
                if comp_flag:
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

                if len(pp_word) == 1:
                    i += 1
                    continue

                if (pp_word[-1] in ["which", "what", "that"]) | (pp_word[0] == "v") | ((pp_word[-2] == w.text) & (pp_word[-1] in punctions)):
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
                pp_str = cut_sub_sent_in_pp_sbar(pp_str, pp_str.split(" "), w.text)

                if len(pp_list) > 0:
                    if pp_str in pp_list[-1][1]:
                        i += 1
                        continue
                    if pp_list[-1][1] in pp_str:
                        for j in range(last_s_idx, last_s_idx + len(pp_list[-1][1].split(" "))):
                            pp_flag[j] = 0
                        pp_list.pop()
                        last_s_idx = -1

                if pp_str in sent:
                    #s_idx = check_continuity(pp_word, s_word, last_s_idx)
                    s_idx = check_continuity(pp_str.split(" "), sent.split(" "), last_s_idx)
                    if pp_flag[s_idx] != 1:
                        if len(pp_list) > 0:
                            if pp_list[-1][1] in pp_str:
                                pp_list.pop()
                        if vp_flag:
                            pp_list.append(('v', pp_str, w.text))
                        else:
                            pp_list.append(('p', pp_str, w.text))
                        #pp_flag = fill_pp_flag(pp_str, s_word, pp_flag, s_idx)
                        pp_flag = fill_pp_flag(pp_str, sent.split(" "), pp_flag, s_idx)
                    elif pp_flag[s_idx + 1] != 1:
                        if pp_list[-1][1].split(" ")[-1] == pp_word[0]:
                            new_pp_str = pp_list[-1][1] + " " + " ".join(pp_str.split(" ")[1:])
                            pp_list[-1] = (pp_list[-1][0], new_pp_str, pp_list[-1][2])
                            pp_flag = fill_pp_flag(" ".join(pp_str.split(" ")[1:]), sent.split(" "), pp_flag, s_idx + 1)
                last_s_idx = s_idx

            elif (w.text == "to") & (" ".join(s_word[i - 2:i + 1]) not in dictionary["comp"]):
                if (w.head.pos_ in ["VERB", "AUX"]) & (w.dep_ == "aux"):
                    if w.head.dep_ != "ROOT":
                        pp_word = [tok.orth_ for tok in w.head.subtree]
                        pos_list = [tok.pos_ for tok in w.head.subtree]
                    elif len([ele for ele in basic_elems if (ele[1] == "dobj") & (w.head.text in ele[2])]) != 0:
                        dobj_elem = [ele for ele in basic_elems if (ele[1] == "dobj") & (w.head.text in ele[2])][0]
                        temp_tree = spacy_nlp(w.text + " " + dobj_elem[2])
                        pp_word = [tok.orth_ for tok in temp_tree]
                        pos_list = [tok.pos_ for tok in temp_tree]
                    else:
                        i += 1
                        continue

                    if (check_continuity(pp_word, s_word, -1) == 0) & (len(pp_word) > len(s_word)/3 * 2) & (w.head.text in s_word[i:]):
                        p_count = pp_word.count(w.text)
                        j = 0
                        p_idx = -1
                        while j < p_count:
                            p_idx = pp_word.index(w.text, p_idx + 1)
                            h_idx = pp_word.index(w.head.text, p_idx + 1)
                            if pp_word[p_idx:h_idx].count(w.text) == 1:
                                pp_word = pp_word[p_idx:]
                                pos_list = pos_list[p_idx:]
                                break
                            j += 1

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
                        i += 1
                        continue
                    alternative = list(pp_word)
                    old_pos_list = list(pos_list)
                    ## 状语从句修饰词 从句补充
                    if w.head.dep_ in ["ccomp", "xcomp", "advcl"]:
                        pp_word, pos_list, h_idx = get_the_complete_phrase(w.head.text, w.head.head.text, s_word,
                                                                           pp_word, pos_list, all_pos_list, pp_list,
                                                                           hyp_words, sent, last_s_idx)
                        if h_idx != -1:
                            vp_flag = True
                        if pp_flag[h_idx] == 1:
                            pp_word = alternative
                            pos_list = old_pos_list
                            vp_flag = False
                        if len([prep for prep in dictionary["verb"] if prep == " ".join(pp_word[:2])]) != 0:
                            vp_flag = False

                    if (pp_word[0] == "to") & (s_word[i - 1] + " to" in dictionary["to"]):
                        pp_word.insert(0, s_word[i - 1])
                        pos_list.insert(0, all_pos_list[i - 1])
                    ## cut long prep
                    flag, key, comp_flag = exist_pp(pos_list, pp_word, dictionary, w.text, True, spill_words_list)
                    if comp_flag:
                        i += 1
                        continue
                    if flag & (key > 1):
                        pp_word = pp_word[:key]
                        if w.text in " ".join(pp_word).split(" , ")[0].split():
                            pp_str = " ".join(pp_word).split(" , ")[0]
                        else:
                            pp_str = " ".join(pp_word)
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
                    pp_str = cut_sub_sent_in_pp_sbar(pp_str, pp_str.split(" "), w.text)
                    if len(pp_list) > 0:
                        if pp_str in pp_list[-1][1]:
                            i += 1
                            continue
                        if pp_list[-1][1] in pp_str:
                            for j in range(last_s_idx, last_s_idx + len(pp_list[-1][1].split(" "))):
                                pp_flag[j] = 0
                            pp_list.pop()
                            last_s_idx = -1

                    if pp_str in sent:
                        #s_idx = check_continuity(pp_word, s_word, last_s_idx)
                        s_idx = check_continuity(pp_str.split(" "), sent.split(" "), last_s_idx)
                        if sent.split(" ")[s_idx - 1] in dictionary["to"].keys():
                            pp_str = " ".join(sent.split(" ")[s_idx - 2:s_idx]) + " " + pp_str
                            for j in range(s_idx - 2, s_idx):
                                pp_flag[j] = 0
                            s_idx = s_idx - 2

                        if pp_flag[s_idx] != 1:
                            if vp_flag:
                                pp_list.append(('v', pp_str, w.text))
                            else:
                                pp_list.append(('p', pp_str, w.text))
                            pp_flag = fill_pp_flag(pp_str, sent.split(" "), pp_flag, s_idx)
                        elif pp_flag[s_idx + 1] != 1:
                            if pp_list[-1][1].split(" ")[-1] == pp_word[0]:
                                new_pp_str = pp_list[-1][1] + " " + " ".join(pp_str.split(" ")[1:])
                                pp_list[-1] = (pp_list[-1][0], new_pp_str, pp_list[-1][2])
                                pp_flag = fill_pp_flag(" ".join(pp_str.split(" ")[1:]), sent.split(" "), pp_flag, s_idx + 1)
                last_s_idx = s_idx
        i += 1

    return pp_list


def extract_conj(text):
    # print("conj_str:", text)
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
            if " , " in new_ner:
                ner_list.extend(new_ner.split(" , "))
            else:
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
            # if ("G" in n_w) | ("K" in n_w):
            #     i += 1
            #     continue
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


def extra_adj_adv(sent, hyp_words):
    spacy_nlp.disable_pipe("merge_entities")
    doc = spacy_nlp(sent)
    source_words = [tok.text for tok in doc]
    hyp_words_list = [hw[1] for hw in hyp_words]
    adj_adv_list = []
    i = 0
    for token in doc:
        if (("".join(source_words[i:i + 3]) in hyp_words_list)|("".join(source_words[i - 2:i + 1]) in hyp_words_list)):
            mod_str = ""
            if "".join(source_words[i:i + 3]) in hyp_words_list:
                mod_str = process_hyp_words("".join(source_words[i:i + 3]), hyp_words, sent, -1)
            if "".join(source_words[i - 2:i + 1]) in hyp_words_list:
                mod_str = process_hyp_words("".join(source_words[i - 2:i + 1]), hyp_words, sent, -1)
            if (mod_str == "") | len(list(set(["of", "too", "as", "how"]) & set(mod_str.lower().split(" ")))) != 0:
                continue
            if token.head.text not in mod_str:
                if token.head.pos == "ADJ":
                    adj_adv_list.append(("ADV", adv_str, token.head.text, "ADJ"))
                elif token.head.pos == "NOUN":
                    adj_adv_list.append(("ADJ", adv_str, token.head.text, "NOUN"))
                elif token.head.head.pos == "NOUN":
                    adj_adv_list.append(("ADJ", adv_str, token.head.head.text, "NOUN"))
        else:
            if (token.dep_ == "amod") & (token.pos_ == "ADJ") & (token.head.pos_ not in ["ADP"]):
                words = [tok.orth_ for tok in token.subtree]
                adj_str = process_hyp_words(" ".join(words), hyp_words, sent, -1)
                if len(list(set(["of", "too", "as", "how"])&set(adj_str.lower().split(" ")))) == 0:
                    adj_adv_list.append(("ADJ", adj_str, token.head.text, token.head.pos_))
            if (token.dep_ == "advmod") & (token.pos_ == "ADV"):
                words = [tok.orth_ for tok in token.subtree]
                adv_str = process_hyp_words(" ".join(words), hyp_words, sent, -1)
                if len(list(set(["of", "too", "as", "how"])&set(adv_str.lower().split(" ")))) == 0:
                    adj_adv_list.append(("ADV", adv_str, token.head.text, token.head.pos_))
                #print("ADV ", adv_str, " ", token.head.text, token.head.pos_)
        i += 1

    return adj_adv_list



