import spacy

## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")

## load the original sentences from file
def load_orig_sent(orig_path):
    orig_sents = open(orig_path, mode="r",encoding='utf-8')
    sent = orig_sents.readline()
    sent_list = []
    while sent:
        sent = sent[:-1]
        s_words = sent.split(" ")[1:-1]
        sent = " ".join(s_words)
        sent_list.append(sent)
        sent = orig_sents.readline()
    return sent_list

def output_conj_sent(path,res):
    f = open(path,'w')
    for i in range(len(res)):
        f.write("i = " + str(i) + "\n")
        for j in res[i]:
            f.write(j + "\n")
        f.write("\n")
    f.close()

# 命名实体：
def extract_ner(orig_sents):
    res = []
    # as well as单独处理
    for i in range(len(orig_sents)):
        text = orig_sents[i]
        doc = nlp(text)
        li = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE', 'ORG', 'NORP', 'PRODUCT', 'EVENT', 'LOC'] and len(ent.text.split()) >= 2:
                li.append(ent.text)
            # print(ent.text, ent.label_)
        extract_ner_byAlpha(text,li)
        res.append(li)
    return res;

def extract_ner_byAlpha(text,li):
    arr = text.split()
    arr[0] = arr[0][0].lower()+arr[0][1:-1]
    for i in range(len(arr)):
        if not arr[i][0].isupper():
            arr[i] = '#'
    s = ''
    # print(arr)
    for word in arr:
        if word == '#':
            s = s.strip()
            if len(s.split()) > 1 and str_in_list(s, li):
                list_in_str(s, li)
                li.append(s)
            s = ''
        else:
            s += word + ' '
    return li

def str_in_list(s,li):
    for i in li:
        if s in i:
            return False
    return True

def list_in_str(s,li):
    for item in li:
        if item in s:
            li.remove(item)


if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    orig_sents = load_orig_sent(sent_path)
    res = extract_ner(orig_sents)
    output_path = './comp_res/ner.txt'
    output_conj_sent(output_path, res)


