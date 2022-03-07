import json
from comp_sent import *
from nltk.corpus import stopwords

pattern = re.compile(r'[A-Z]+\.\s[A-Z]\.\s+')
match = pattern.findall("S  ")

english_punctuations = [',', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '—', '\'', '{', '}']
abbr = ["n't", "'s", "'re", "'ll", "'m"]
wrong_abbr = [" n ' t ", " ' s ", " ' re ", " ' ll ", " ' m "]


def extract_ques_ans():
    with open("/Users/pinji/Desktop/日常工作/MRC数据集/SQuAD/dev-v2.0.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(data.keys())
    data_list = data['data']
    print(len(data))
    context_list = []
    qas_list = []
    context_ans = []
    for data in data_list:
        paras = data['paragraphs']
        for para in paras:
            context = para['context']
            qas = para['qas']
            ques_list = []
            ans_list = []
            context_list.append(context)
            for ques in qas:
                question = ques['question']
                answer = ques['answers']
                ques_list.append(question)
                ans_list.append(answer)
            qas_list.append(ques_list)
            context_ans.append(ans_list)
    print(len(context_list))
    print(len(qas_list))

    length = len(context_list)
    x = 0
    while x < length:
        match = pattern.findall(context_list[x])
        if len(match) > 0:
            del context_list[x]
            del qas_list[x]
            del context_ans[x]
            x -= 1
            length -= 1
        elif ". . ." in context_list[x]:
            del context_list[x]
            del qas_list[x]
            del context_ans[x]
            x -= 1
            length -= 1
        x += 1

    f1 = open("./context.txt", "w")
    for i in range(len(context_list)):
        f1.write("context_id=" + str(i) + "\n")
        context_list[i] = context_list[i].replace("\n", "")
        sent_list = context_list[i].split(". ")
        sent_list = [x for x in sent_list if x != '']
        f1.write(sent_list[0])
        for j in range(1, len(sent_list)):
            end = sent_list[j - 1][-3:]
            if (len(sent_list[j].split(" ")) > 3) & (sent_list[j][0].isupper()) & (end not in ["e.g", "U.S"]):
                f1.write(".\n")
                f1.write(sent_list[j])
            else:
                f1.write(". " + sent_list[j])
        f1.write("\n")
        f1.write("\n")
    f1.close()

    f2 = open("./questions.txt", "w")
    for i in range(len(qas_list)):
        f2.write("context_id=" + str(i) + "\n")
        ques_list = qas_list[i]
        for que in ques_list:
            f2.write(que + "\n")
        f2.write("\n")

    f2.close()

    f3 = open("./answers.txt", "w")
    for i in range(len(context_ans)):
        f3.write("context_id=" + str(i) + "\n")
        ans_list = context_ans[i]
        for ans in ans_list:
            line = ""
            if len(ans) != 0:
                for a in ans:
                    line += str(a) + "|"

                line = line[:-1] + "\n"
                f3.write(line)
            else:
                f3.write("None" + "\n")
        f3.write("\n")
    f3.close()


def read_txt(file_path, type):
    f = open(file_path, "r")
    res_list = []
    if type == "context":
        line = f.readline()
        contexts = []
        while line:
            if "context_id" in line:
                if len(contexts) != 0:
                    res_list.append(contexts)
                contexts = []
            elif len(line) > 1:
                contexts.append(line[:-1])
            line = f.readline()
        res_list.append(contexts)

    if type == "question":
        line = f.readline()
        questions = []
        while line:
            if "context_id" in line:
                if len(questions) != 0:
                    res_list.append(questions)
                questions = []
            elif len(line) > 1:
                questions.append(line[:-1])
            line = f.readline()
        res_list.append(questions)

    if type == "answer":
        line = f.readline()
        answers = []
        while line:
            if "context_id" in line:
                if len(answers) != 0:
                    res_list.append(answers)
                answers = []
            elif len(line) > 1:
                line = line[:-1].split("|")
                answers.append(line)
            line = f.readline()
        res_list.append(answers)
    f.close()
    return res_list


def word_extraction(sentence):
    # 提取句子中的词们
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    sentence = re.sub(r, '', sentence)
    words = sentence.split()
    stop_words = set(stopwords.words('english'))
    cleaned_text = [w.lower() for w in words if not w in stop_words]
    return cleaned_text


# def tokenize(sentences):
#     #对所有句子做 step1,生成词表
#     r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
#     words = []
#     for sentence in sentences:
#         sentence = re.sub(r, '', sentence)
#         w = word_extraction(sentence)
#         words.extend(w)
#     words = sorted(list(set(words)))
#     return words
#
#
# def build_word_bag(allsentences, vocab):
#     bow_list = []
#     for sentence in allsentences:
#         words = word_extraction(sentence)
#         bag_vector = np.zeros(len(vocab))
#         for w in words:
#             for i, word in enumerate(vocab):
#                 if word == w:
#                     bag_vector[i] += 1
#         print("{0}\n{1}\n".format(sentence, np.array(bag_vector)))
#         bow_list.append(bag_vector)


def match_ques_context():
    context_list = read_txt("./context.txt", "context")
    answer_list = read_txt("./answers.txt", "answer")
    ques_list = read_txt("./questions.txt", "question")
    match_res = []
    for i in range(len(context_list)):
        contexts = context_list[i]
        questions = ques_list[i]
        answers = answer_list[i]
        match_list = []
        context_range = []
        start = 0
        for j in range(len(contexts)):
            c_range = []
            context = contexts[j]
            end = start + len(context) - 1
            c_range.append(start)
            c_range.append(end)
            start = start + len(context) + 1
            context_range.append(c_range)
        for a_idx in range(len(answers)):
            match = {}
            match["question"] = questions[a_idx]
            ans_list = answers[a_idx]
            match["answer"] = set()
            if "None" not in ans_list:
                for a in range(len(ans_list)):
                    ans = ans_list[a]
                    ans = json.loads(json.dumps(eval(ans)))
                    ans_text = ans['text']
                    match["answer"].add(ans_text)
                answer_start = ans['answer_start']
                for r in range(len(context_range)):
                    c_range = context_range[r]
                    if (answer_start >= c_range[0]) & (answer_start <= c_range[1]):
                        match["c_idx"] = r
                        match["context"] = contexts[r]
                        if ans_text not in match["context"]:
                            r = r - 1
                            match["context"] = contexts[r]
                            if ans_text not in contexts[r]:
                                print(match["question"])
                                print(ans_text)
                                print(contexts[r])
                                print("wrong")
                        match_list.append(match)
                        break
            else:
                match["answer"].add("None")
                match["c_idx"] = -1
                match["context"] = "None"
                match_list.append(match)

        match_res.append(match_list)
    return match_res


def write_in_txt(match_res):
    f = open("./ans_context.txt", "w")
    for i in range(len(match_res)):
        f.write("context_id = " + str(i) + "\n")
        match_list = match_res[i]
        for match in match_list:
            f.write("question: " + match["question"] + "\n")
            f.write("answer: " + "|".join(match["answer"]) + "\n")
            f.write("context_idx: " + str(match["c_idx"]) + "\n")
            f.write("context: " + match["context"] + "\n")
            f.write("\n")

        f.write("==========FIN==========" + "\n")
        f.write("\n")

    f.close()


def del_brackets(sent):
    while "(" in sent:
        s_idx = sent.index("(")
        e_idx = sent.index(")")
        sent = sent[0:s_idx] + sent[e_idx + 2:]
    return sent


def load_formulation(file_path):
    f = open(file_path, 'r')
    formulations = []
    line = f.readline()
    while line:
        line = line[:-1]
        formulations.append(line)
        line = f.readline()
    return formulations


def write_original_text(context_list):
    f = open("./orig_sent/context_orig.txt", "w")
    for i in range(0, len(context_list)):
        contexts = context_list[i]
        for con in contexts:
            f.write(con + "\n")

    f.close()


def format_input(file_path, sent_path, strip_path):
    f = open(file_path, mode="r")
    w1 = open(sent_path, mode="w")
    w2 = open(strip_path, mode="w")
    line = f.readline()
    formulations = load_formulation("./formulation.txt")
    while line:
        for a in abbr:
            if a in line:
                idx = line.find(a)
                if line[idx - 1] != " ":
                    line = line[0:idx] + " " + line[idx:]
        for punc in english_punctuations:
            if punc in line:
                idx = line.find(punc)
                while idx != -1:
                    if (line[idx - 1].isdigit()) & ((line[idx + 1].isdigit())) & (punc in [".", ","]):
                        idx = line.find(punc, idx + 1)
                        continue
                    else:
                        break_flag = False
                        if punc in ["(", "["]:
                            temp = line[idx:idx + 4].strip().rstrip()
                            if temp[-1] == ".":
                                temp = temp[:-1]
                            for fm in formulations:
                                if temp in fm:
                                    break_flag = True
                                    break
                        if punc in [")", "]", "!"]:
                            temp = line[idx - 3:idx + 1].strip().rstrip()
                            for fm in formulations:
                                if temp in fm:
                                    break_flag = True
                                    break
                        if break_flag:
                            idx = line.find(punc, idx + 1)
                            continue
                        if line[idx - 1] != " ":
                            line = line[0:idx] + " " + line[idx:]
                            idx = idx + 1
                        if (line[idx + 1] != " ") & (punc not in [".", "!", "?"]):
                            line = line[0:idx + 1] + " " + line[idx + 1:]
                        if (line[idx + 1] != " ") & (punc == "."):
                            if (line[idx + 1] != "\n") & (not line[idx + 1].isdigit()):
                                line = line[0:idx + 1] + " " + line[idx + 1:]
                        idx = line.find(punc, idx + 1)
        for i in range(len(wrong_abbr)):
            if wrong_abbr[i] in line:
                line = line.replace(wrong_abbr[i], " " + abbr[i] + " ")
        if "\"" in line:
            count = 1
            idx = line.find("\"")
            while idx != -1:
                if count % 2 != 0:
                    if line[idx + 1] != " ":
                        line = line[0:idx] + "``" + " " + line[idx + 1:]
                if count % 2 == 0:
                    if line[idx - 1] != " ":
                        line = line[0:idx] + " " + "''" + line[idx + 1:]
                idx = line.find("\"", idx + 1)
                count += 1
        if line[-1] == "\n":
            line = line[:-1]
        if "cannot" in line:
            line = line.replace("cannot", "can not")
        if "d ' état" in line:
            line = line.replace("d ' état", "d'état")
        line = line.replace(", \"", ", \'\'")
        line = line[0].upper() + line[1:]
        if (line[-1] == ".") & (line[-2] != " "):
            line = line[:-1] + " ."
        if line[-4:] == ". \'\'":
            line = line[:-4] + " . ''"
        line = " ".join(line.split())
        print(line)
        w1.write("<s> " + line + " <\s>\n")
        w2.write(line.strip().rstrip() + "\n")
        line = f.readline()


def preprocess_compress():
    # context_list = read_txt("./context.txt", "context")
    # write_original_text(context_list)
    # file_name = "context"
    # orig_txt = "./orig_sent/" + file_name + "_orig.txt"
    # cln_path = "./comp_input/" + file_name + ".cln.sent"
    # strip_path = "./comp_input/" + file_name + ".cln.strip.sent"
    # format_input(orig_txt, cln_path, strip_path)
    # dep_path = "./comp_input/" + file_name + ".cln.dep"
    # gen_dep_file(cln_path, dep_path)
    file_name = "ncontext"
    cln_path = "./comp_input/" + file_name + ".cln.sent"
    strip_path = "./comp_input/" + file_name + ".cln.strip.sent"
    dep_path = "./comp_input/" + file_name + ".cln.dep"
    f1 = open(cln_path, "r")
    f2 = open(strip_path, "w")
    line = f1.readline()
    while line:
        f2.write(line.replace("<s> ", "").replace(" <\s>", ""))
        line = f1.readline()
    f2.close()
    gen_dep_file(cln_path, dep_path)

def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def del_pare():
    context_list = read_txt("./context.txt", "context")
    formulations = load_formulation("./formulation.txt")
    del_context_list = []
    for i in range(len(context_list)):
        contexts = context_list[i]
        del_contests = []
        for j in range(len(contexts)):
            if ("(" in contexts[j]) | ("{" in contexts[j]) | ("[" in contexts[j]):
                cut_contents = extra_cut_content(contexts[j])
                del_res = contexts[j]
                for c in cut_contents:
                    del_flag = True
                    for f in formulations:
                        if c in f:
                            del_flag = False
                            break
                    if del_flag:
                        del_res = del_res.replace(c, "")
                del_res = " ".join(del_res.split())
                del_contests.append(del_res)
            else:
                contexts[j] = " ".join(contexts[j].split())
                del_contests.append(contexts[j])
        del_context_list.append(del_contests)
    return del_context_list


def write_del_context(del_context_list):
    f = open("./cut_context.txt","w")
    for i in range(len(del_context_list)):
        f.write("context_id=" + str(i) + "\n")
        for j in range(len(del_context_list[i])):
            f.write(del_context_list[i][j] + "\n")
        f.write("\n")
    f.close()


def update_comp_label(cut_comp_lable_path, cln_sent_path, cut_cln_sent_path):
    f1 = open(cut_comp_lable_path, 'r')
    cut_comp_lables = []
    line = f1.readline()
    while line:
        line = line[:-1]
        cut_comp_lables.append(line)
        line = f1.readline()

    f2 = open(cln_sent_path, 'r')
    orig_contexts = []
    line = f2.readline()
    while line:
        line = line[:-1]
        orig_contexts.append(line)
        line = f2.readline()

    f3 = open(cut_cln_sent_path, 'r')
    cut_contexts = []
    line = f3.readline()
    while line:
        line = line[:-1]
        cut_contexts.append(line)
        line = f3.readline()

    f = open("./context_result_greedy.sents", 'w')
    for i in range(len(cut_comp_lables)):
        comp_lable = cut_comp_lables[i].split(" ")
        orig_words = orig_contexts[i].split(" ")
        cut_idx = search_cut_content(orig_words)
        if len(cut_idx) != 0:
            for tup in cut_idx:
                count = tup[1] - tup[0] + 1
                for j in range(count):
                    comp_lable.insert(tup[0], '0')
        if len(comp_lable) != len(orig_words):
            print("orig context: ", orig_words)
            print("comp label: ", comp_lable)
        f.write(" ".join(comp_lable) + "\n")

def extra_cut_content(s):
    i = 0
    # 栈
    t = []
    # 字典 存储括号
    the_dic = {'(': ')', "{": "}", "[": "]"}
    # 开始遍历字符串
    idx_list = []
    content_list = []
    while i < len(s):
    # 左括号则入栈，继续遍历
        if s[i] in the_dic.keys():
            t.append(s[i])
            idx_list.append(i)
            i = i + 1
            continue
        # 右括号进行判断
        if s[i] in the_dic.values():
        # 右括号需要入栈，但当前栈为空，输出False
            if len(t) == 0:
                return False
            else:
            # 判断栈顶括号是否和当前字符表示的括号相匹配
                temp = t[-1]
                # 匹配则出栈顶元素，继续遍历
                if s[i] == the_dic.get(temp):
                    temp_str = s[idx_list[-1]:i+1]
                    content_list.append(temp_str)
                    t.pop()
                    idx_list.pop()
                    i = i + 1
                # 否则结束 False
                else:
                    break
        else:
            i = i + 1
    if len(t) != 0:
        print("Wrong")
    content_list.sort(key=lambda i:len(i), reverse=True)
    print(content_list)
    return content_list

def search_cut_content(orig_words):
    i = 0
    t = []
    the_dic = {'(': ')', '[': ']', '{': '}'}
    idx_list = []
    cut_idx = []
    while i < len(orig_words):
        if orig_words[i] in the_dic.keys():
            t.append(orig_words[i])
            idx_list.append(i)
            i = i + 1
            continue
        if orig_words[i] in the_dic.values():
            if len(t) == 0:
                return cut_idx
            else:
                temp = t[-1]
                # 匹配则出栈顶元素，继续遍历
                if orig_words[i] == the_dic.get(temp):
                    idx_tup = (idx_list[-1], i)
                    while len(cut_idx) > 0:
                        if (cut_idx[-1][0] > idx_tup[0]) & (cut_idx[-1][1] < idx_tup[1]):
                            cut_idx.pop()
                        else:
                            break

                    cut_idx.append(idx_tup)
                    t.pop()
                    idx_list.pop()
                    i = i + 1
                # 否则结束 False
                else:
                    break
        else:
            i = i + 1
    if len(t) != 0:
        print("Wrong")
    return cut_idx

def format_formulation(formu):
    for punc in ['(', ')', '[', ']', "!", ",", "≡"]:
        idx = formu.find(punc)
        while idx != -1:
            if formu[idx - 1] != " ":
                formu = formu[0:idx] + " " + formu[idx:]
                idx = idx + 1
            if idx + 1 < len(formu):
                if formu[idx + 1] != " ":
                    formu = formu[0:idx + 1] + " " + formu[idx + 1:]
            idx = formu.find(punc, idx + 1)
    return formu

if __name__ == '__main__':
    # del_context_list = del_pare()
    # write_del_context(del_context_list)
    preprocess_compress()
    # match_res = match_ques_context()
    # write_in_txt(match_res)
    # cut_comp_lable_path = "./comp_label/slahan_w_syn/ncontext_result_greedy.sents"
    # cln_sent_path = "./comp_input/context.cln.sent"
    # update_comp_label(cut_comp_lable_path, cln_sent_path, "./comp_input/ncontext.cln.sent")
    # strs = "For instance, the language {xx | x is any binary string} can be solved in linear time on a multi-tape Turing machine, but necessarily requires quadratic time in the model of single-tape Turing machines."
    # extra_cut_content(strs)

