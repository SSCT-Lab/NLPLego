import spacy

## SpaCy dependency parser
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")
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

def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_root_phrase(doc):
    for token in doc:
        if 'ROOT' in token.dep_:
            return token.text


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

def write_list_in_txt(subj, predicate, dobj, file_path):
    f = open(file_path, "w")
    for i in range(len(subj)):
        f.write("i = " + str(i) + "\n")
        f.write("subj: " + subj[i] + "\n")
        f.write("predicate: " + predicate[i] + "\n")
        f.write("dobj: " + dobj[i] + "\n")
        f.write("\n")


if __name__ == '__main__':
    file_name = "business"
    sent_path = "./comp_input/" + file_name + ".cln.sent"
    comp_label = load_label("./comp_res/slahan_w_syn/2_" + file_name + "_result_greedy.sents")
    orig_sents = load_orig_sent(sent_path)
    subj = []
    dobj = []
    predicate = []
    for sentence in orig_sents:
        doc = nlp(sentence)
        root = get_root_phrase(doc)
        subject_phrase = get_subject_phrase(doc)
        object_phrase = get_object_phrase(doc)
        subj.append(str(subject_phrase))
        predicate.append(root)
        dobj.append(str(object_phrase))
    write_list_in_txt(subj, predicate, dobj, "./comp_res/subj_dobj.txt")
