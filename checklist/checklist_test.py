import spacy
import re
from checklist.editor import Editor
from checklist.perturb import Perturb
import itertools
import numpy as np
from nltk.corpus import stopwords

from preprocess import read_txt
from process_utils import load_orig_sent, get_sentence_len, format_question

test_str = ["thank you we got on a different flight to Chicago",
            "I am from China",
            "John wants to go to China"]
# test_str = "European authorities fined Google a great record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
# test_str = "thank you we got on a different flight to Chicago"
nlp = spacy.load("en_core_web_sm")
doc = list(nlp.pipe(test_str))


def multiple_replace(text, adict):  # 利用re替换多个子串
    if not adict:
        return

    rx = re.compile('|'.join(map(re.escape, adict)))

    def replace_one(match):
        return adict[match.group(0)]

    return rx.sub(replace_one, text)


def generate_replace_dict(doc, unrep):  # 生成需要替换的子串字典
    ner_replace_dict = {}
    auto_replace_dict = {}
    count = 0
    for x in doc:
        if (x.pos_ == 'ADJ' or x.pos_ == 'ADV') & (x.text[0][0].islower()) & (x.text not in unrep):
            #         print(x.text)
            auto_replace_dict[x.text] = "{mask}"
            count += 1
            if count >= 2:
                break

    return ner_replace_dict, auto_replace_dict


def replace_permutation(replace_dict):  # 生成替换的排列组合，弃用！
    editor = Editor()
    samples = []
    repeat_num = len(replace_dict)
    permu = list(itertools.product([False, True], repeat=repeat_num))
    for cur_permu in permu[1:]:
        replace_keys = np.array(list(replace_dict.keys()))[np.array(cur_permu)]
        replace_values = np.array(list(replace_dict.values()))[np.array(cur_permu)]
        cur_replace_dict = dict(zip(replace_keys, replace_values))
        # print(cur_replace_dict)
        template = multiple_replace(test_str, cur_replace_dict)
        print("current template: ", template)
        samples.extend(editor.template(template).data)
    return samples


def result_to_a_list(res):  # 将checklist奇怪的构成转换为str数组
    if not res:
        return []
    return np.array(res)[:, 1:].flatten().tolist()


def replace_location(data: list, n):
    # n : 对于每一个句子生成的样本数
    ret = Perturb.perturb(data, Perturb.change_location, n=n)  # 替换city and country names，spacy没有识别到则返回 []
    return result_to_a_list(ret.data)


def replace_name(data: list, n):
    ret = Perturb.perturb(data, Perturb.change_names, n=n)  # 替换city and country names，spacy没有识别到则返回 []
    return result_to_a_list(ret.data)


def replace_number(data: list, n):
    '''
        Does not change '2' or '4' to avoid abbreviations (this is 4 you, etc)
        20% 数值范围内修改
    '''
    ret = Perturb.perturb(data, Perturb.change_number, n=n)
    return result_to_a_list(ret.data)


def contractions(strs: list):
    ret = []
    for str in strs:
        tmp_ret = Perturb.perturb(str, Perturb.contractions).data
        if tmp_ret:
            ret.append(tmp_ret[0][1])
    return ret


def add_negation(data: list):
    try:
        ret = Perturb.perturb(data, Perturb.add_negation)
    except StopIteration:
        print("exception")
    ret = Perturb.perturb(data, Perturb.add_negation)
    return [ret.data[0][1]]


def add_typos(data: list, n):
    ret = []
    for i in range(n):
        tmp_ret = Perturb.perturb(test_str, Perturb.add_typos)  # Typos，每次调用随机拼错 -_-
        ret.extend(result_to_a_list(tmp_ret.data))
    return ret

def aug_squad(orig_sent_list, sent_context_map, ques_list):
    f = open("./checklist_squad.txt", "a")
    all_tests = []
    roberta_n = 8
    editor = Editor()
    for i in range(2219, 2220):
        f.write("sent_id = " + str(i) + "\n")
        print(orig_sent_list[i])
        context_idx = sent_context_map[i]
        unrep_words = []
        for q in ques_list[context_idx]:
            q_w = [w for w in format_question(q).split() if w.lower() not in set(stopwords.words("english"))]
            unrep_words.extend(q_w)
        sub_doc = nlp(orig_sent_list[i])
        tests = []
        if "{" not in orig_sent_list[i]:
            _, auto_replace_dict = generate_replace_dict(sub_doc, unrep_words)
            template = multiple_replace(str(sub_doc.doc), auto_replace_dict)
            if template is not None:
                tests.extend(editor.template(template, remove_duplicates=True).data[:roberta_n])
            for t in tests:
                print(t)
                f.write(t + "\n")
        f.write("\n")
        all_tests.append(tests)
    f.close()


def aug_sst(sst_sents):
    print(sst_sents)
    for sent in sst_sents:
        doc = nlp(sent)
        samples = []
        samples.extend(replace_location(doc, 2))
        samples.extend(replace_name(doc, 2))
        samples.extend(replace_number(doc, 2))
        #samples.extend(contractions(test_str))
        samples.extend(add_negation(doc))
        print(samples)

if __name__ == "__main__":
    sst_file = open("./comp_input/sst.cln.sent", "r")
    sst_sents = []
    line = sst_file.readline()
    while line:
        line = " ".join(line.split()[1:-2])
        sst_sents.append(line)
        line = sst_file.readline()

    aug_sst(sst_sents)
