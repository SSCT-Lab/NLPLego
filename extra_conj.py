import spacy
from nltk import CoreNLPParser
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP

## Stanford Corenlp constituency parser
eng_parser = CoreNLPParser('http://127.0.0.1:9000')
## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")


## load dictionary.txt (Saved some fixed collocations)
def load_dictionary(d_path):
    d = open(d_path, "r")
    line = d.readline()
    dictionary = {}
    while line:
        if "key " in line:
            key = line[:-1].split(" ")[-1]
            dictionary[key] = []
        elif len(line) > 1:
            l_words = line[:-1].split(" ")
            index = l_words.index(key) - 1
            dictionary[key].append(l_words[index])
        line = d.readline()
    return dictionary

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

## load the sentences especiailly for conj
def load_conj_sent(path):
    conj_sents = open(path, mode="r")
    sent = conj_sents.readline()
    sent_list = []
    while sent:
        if 'but also' in sent:
            sent = sent.replace('but also', 'but')
        sent_list.append(sent.strip())
        sent = conj_sents.readline()
    return sent_list

# 并列连词：
#     表转折：but，yet，while
#     表选择：or，either...or...，neither...or...，otherwise
#     表因果：for，so
#     ！！表并列：and，or，either…or , neither…nor , not only…but (also) , both…and , as well as
#     最简单思路：向前找preconj，向后找conj
#     特殊情况：and/or时多个并列项
#     提高正确率：判断两端是否为同一tag/pos_，合并两端动短名短
#     token.text, token.pos_, token.dep_, token.head.text, token.head.pos_
def extract_conj(orig_sents):
    res = []
    # as well as单独处理
    for i in range(len(orig_sents)):
        text = orig_sents[i]
        doc = nlp(text)
        ans = []
        min = 0
        j = 0
        while j < len(doc):
            if doc[j].dep_=='preconj':
#                 two words conj
                j = two_conj(j,doc,ans)
                min = j+1
                continue
            elif doc[j].dep_ == 'conj':
                j = single_conj(min,j,doc,ans)
                min = j
            else:
                pass
            j+=1
        print(text,ans)
        res.append(ans)
    print(res)


def single_conj(min,j,doc,ans):
    flag = 0
    str = ''
    if min != 0 and doc[min] == doc[j].head:
        for i in doc[min+1:j+1]:
            str += ' '+i.text
        if str.isspace() == False:
            ans[-1] = (ans[-1]+str).strip()
        return j+1;
    for i in range(min, j+1):
        if doc[i] == doc[j].head:
            flag = 1
        if flag == 1:
            str += ' ' + doc[i].text
    if str.isspace() == False:
        ans.append(str.strip())
    return j;

def two_conj(j,doc,ans):
    str = ''
    for i in range(j, len(doc)):
        str += ' '+doc[i].text
        if doc[i].dep_ == 'conj' and doc[i].head == doc[j].head:
            break
    if str.isspace() == False:
        ans.append(str.strip())
    return i+1;

def test(orig_sents):
    for i in range(5):
        text = orig_sents[i]
        doc = list(nlp(text))
        for token in doc:
            print("{0}/{1} <--{2}-- {3}/{4}".format(
                token.text, token.pos_, token.dep_, token.head.text, token.head.pos_))
            print("subtree: ",end='  ')
            for k in token.subtree:
                print("{0} ".format(k.text),end=' ')
            print()
    text = "Not only he but also his son joined the Party two years ago."
    doc = list(nlp(text))
    for token in doc:
        print("{0}/{1} <--{2}-- {3}/{4}".format(
            token.text, token.pos_, token.dep_, token.head.text, token.head.pos_))
        print("subtree: ", end='  ')
        for k in token.subtree:
            print("{0} ".format(k.text), end=' ')
        print()

def MultiTreePaths(root):
    def helper(root,path,res):
        if type(root)==str:
            res.append(path+str(root))
            return
        l = len(root)
        for i in range(l):
            if len(root)>=i:
                if root[i]:
                    helper(root[i],path+str(root.label())+'->',res)
    if root is None:
        return []
    l = []
    helper(root,'',l)
    return l



if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    comp_label = load_label("./comp_res/2_" + file_name + "_result_greedy.sents")
    orig_sents = load_orig_sent(sent_path)

    # check_grammar(orig_sents, comp_label)

    # txt = "The pandemic has severely disrupted the education and childcare systems that millions of parents rely on while they 're at work."
    # txt = "I love bananas,apples and pears."
    # doc = nlp(txt)
    # for token in doc:
    #     print("{0}/{1} <--{2}-- {3}/{4}".format(
    #         token.text, token.pos_, token.dep_, token.head.text, token.head.pos_))

    path = "./comp_input/conj.txt"
    conj_sents = load_conj_sent(path)
    extract_conj(conj_sents)
    extract_conj(orig_sents)

    # corenlp = StanfordCoreNLP(r"D:/PycharmProjects/stanfordnlp", lang='en')
    # tree = Tree.fromstring(corenlp.parse(txt))
    # r = MultiTreePaths(tree)
    # print(r)
    # test(conj_sents)




