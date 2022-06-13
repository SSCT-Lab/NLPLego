import re

import pandas as pd
import os



def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def splice_words(org_words, comp_flag):
    comp = ""
    i = 0
    while i < len(org_words):
        if comp_flag[i] == "1":
            comp += org_words[i] + " "
        i += 1
    return comp[:-1]


def read_result(org_path, comp_path):
    orgs = open(org_path, mode="r")
    comps = open(comp_path, mode="r")
    comp_list = []
    source_list = []
    org_line = orgs.readline()
    comp_line = comps.readline()
    while org_line:
        org_line = org_line.replace("''", "\"")
        org_line = org_line.replace("``", "\"")
        org_line = org_line[4:-6].split(" ")
        comp_line = comp_line[4:-6].split(" ")
        comp = splice_words(org_line, comp_line)
        comp_list.append(comp)
        source_list.append(" ".join(org_line))
        print(comp)
        print(" ".join(org_line))
        org_line = orgs.readline()
        comp_line = comps.readline()
    return comp_list, source_list


def save_to_csv(name, list, file_path):
    dict_temp = {name: list}
    csv_file = pd.DataFrame(dict_temp)
    csv_file.to_csv(file_path, encoding="utf_8")


# def format_input(file_path, sent_path, strip_path):
#     f = open(file_path, mode="r")
#     w1 = open(sent_path, mode="w")
#     w2 = open(strip_path, mode="w")
#     line = f.readline()
#     while line:
#         for a in abbr:
#             if a in line:
#                 idx = line.find(a)
#                 if line[idx - 1] != " ":
#                     line = line[0:idx] + " " + line[idx:]
#         for punc in english_punctuations:
#             if punc in line:
#                 idx = line.find(punc)
#                 while idx != -1:
#                     if line[idx - 1] != " ":
#                         line = line[0:idx] + " " + line[idx:]
#                         idx = idx + 1
#                     if (line[idx + 1] != " ") & (punc not in [".", "!", "?"]):
#                         line = line[0:idx + 1] + " " + line[idx + 1:]
#                     if (line[idx + 1] != " ") & (punc == "."):
#                         if (line[idx + 1] != "\n") & (not line[idx + 1].isdigit()):
#                             line = line[0:idx + 1] + " " + line[idx + 1:]
#                     idx = line.find(punc, idx + 1)
#         if "\"" in line:
#             count = 1
#             idx = line.find("\"")
#             while idx != -1:
#                 if count % 2 != 0:
#                     if line[idx + 1] != " ":
#                         line = line[0:idx] + "``" + " " + line[idx + 1:]
#                 if count % 2 == 0:
#                     if line[idx - 1] != " ":
#                         line = line[0:idx] + " " + "''" + line[idx + 1:]
#                 idx = line.find("\"", idx + 1)
#                 count += 1
#         print(line)
#         if line[-1] == "\n":
#             line = line[:-1]
#         if "cannot" in line:
#             line = line.replace("cannot", "can not")
#         line = line[0].upper() + line[1:]
#         w1.write("<s> " + line + " <\s>\n")
#         w2.write(line.strip().rstrip() + "\n")
#         line = f.readline()


def gen_dep_file(sent_path, dep_path):
    f = open(sent_path, mode="r")
    w = open(dep_path, mode="w")
    line = f.readline()
    while line:
        words = line[:-1].split(" ")
        w_l = ""
        for wo in words:
            w_l += "0-0" + " "
        w_l = w_l.strip().rstrip()
        w.write(w_l + "\n")
        line = f.readline()


def convert_label(sent_path, label_path, res_path):
    s_p = open(sent_path, mode="r")
    l_p = open(label_path, mode="r")
    s_line = s_p.readline()
    l_line = l_p.readline()
    res_file = open(res_path, mode="w")
    while s_line:
        s_line = s_line[:-1]
        s_words = s_line.split(" ")[1:-1]
        l_words = l_line.split(" ")[1:-1]
        res_words = []
        for i in range(len(l_words)):
            if l_words[i] == "1":
                res_words.append(s_words[i])
        res_line = " ".join(res_words)
        print(res_line)
        s_line = s_p.readline()
        l_line = l_p.readline()
        res_file.write(res_line + "\n")
    s_p.close()
    l_p.close()
    res_file.close()


if __name__ == '__main__':
    file_name = "context"
    orig_txt = "./orig_sent/" + file_name + "_orig.txt"
    cln_path = "./comp_input/" + file_name + ".cln.sent"
    strip_path = "./comp_input/" + file_name + ".cln.strip.sent"
    format_input(orig_txt, cln_path, strip_path)
    dep_path = "./comp_input/" + file_name + ".cln.dep"
    gen_dep_file(cln_path, dep_path)
    # model_name = "slahan_w_syn"
    # num = "2"
    # sent_path = "./comp_input/" + file_name + ".cln.sent"
    # res_dir = "./comp_res/" + model_name + "/"
    # mkdir(res_dir)
    # res_path = "./comp_res/" + model_name + "/" + num + "_" + file_name + "_res.txt"
    # label_path = "./comp_label/" + model_name + "/" + num + "_" + file_name + "_result_greedy.sents"
    # convert_label(sent_path, label_path, res_path)