import nltk
from nltk.corpus import wordnet
from gen_temp import *
import re
from nltk import CoreNLPParser
from nltk.corpus import stopwords
import spacy
import transformers

nlp = spacy.load("en_core_web_sm")
sbar_pattern = re.compile(r't\d+')
unmasker = transformers.pipeline('fill-mask', model='bert-base-uncased')
BERT_SCORE = 0.1

## Remove punctuation and named entities
def filer_word(pos_list, adjunct, ner_list):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    stops = set(stopwords.words("english"))
    #tag_list = nltk.pos_tag(adjunct_word)
    doc = nlp(adjunct)
    word_pos = [tok.pos_ for tok in doc]
    masked_word = []
    masked_adjunct = []
    adjunct_word = [tok.text for tok in doc]
    if "-" in adjunct_word:
        index = adjunct_word.index("-")
    for i in range(len(adjunct_word)):
        word = adjunct_word[i]
        if "-" in adjunct_word:
            if i not in [index - 1, index, index + 1]:
                for ner in ner_list:
                    if word in ner:
                        continue
                if (word not in stops) & (word not in english_punctuations) & (word_pos[i] in pos_list):
                    masked_word.append(word)
                    temp_phrase = list(adjunct_word)
                    temp_phrase[i] = "[MASK]"
                    masked_adjunct.append(" ".join(temp_phrase).replace(" - ", "-"))
        else:
            for ner in ner_list:
                if word in ner:
                    continue
            if (word not in stops) & (word not in english_punctuations) & (word_pos[i] in pos_list):
                masked_word.append(word)
                temp_phrase = list(adjunct_word)
                temp_phrase[i] = "[MASK]"
                masked_adjunct.append(" ".join(temp_phrase).replace(" - ", "-"))

    if len(masked_adjunct) == 0:
        masked_adjunct.append(" ".join(adjunct_word).replace(" - ", "-"))
        masked_word.append("X")

    return masked_word, masked_adjunct


def gen_mask_phrase(adjunct_list, pos_list, all_ner):
    all_masked_adjunct = []
    all_masked_word = []
    for i in range(len(adjunct_list)):
        adjuncts = adjunct_list[i]
        ner_list = all_ner[i]
        masked_adjunct_list = []
        masked_word_list = []
        for adjunct in adjuncts:
            masked_word, masked_adjunct = filer_word(pos_list, adjunct, ner_list)
            masked_word_list.append(masked_word)
            masked_adjunct_list.append(masked_adjunct)
        all_masked_adjunct.append(masked_adjunct_list)
        all_masked_word.append(masked_word_list)
    return all_masked_word, all_masked_adjunct


def gen_masked_sent(j, temp, masked_adjuncts):
    pred_list = []
    new_temp = []
    slot = ["t" + str(j)] * temp.count("t" + str(j))
    for i in range(len(masked_adjuncts)):
        new_sent = temp.replace(" ".join(slot), masked_adjuncts[i])
        result = set(sbar_pattern.findall(new_sent))
        sent_word = new_sent.split(" ")
        if len(result) != 0:
            new_temp.append(new_sent)
            for r in result:
                rep_slot = [r] * sent_word.count(r)
                new_sent = new_sent.replace(" ".join(rep_slot), "")
                new_sent = format_sent(new_sent)
            pred_list.append(new_sent)
        else:
            new_temp.append(new_sent)
            pred_list.append(new_sent)
    return pred_list, new_temp

def format_abbr(sent):
    abbr = ["n't", "'s", "'re", "'ll", "'m"]
    words = sent.split(" ")
    for w in words:
        if w in abbr:
            idx = sent.find(w)
            sent = sent[:idx - 1] + sent[idx:]
    return sent


def pred_sent_by_bert(step_list, masked_temp, words):
    tests_set = set()
    new_temps = set()
    for i in range(len(step_list)):
        mask_sent = step_list[i]
        word = words[i]
        if "[MASK]" in mask_sent:
            print(mask_sent)
            pred_res = unmasker(mask_sent)
            for r in pred_res:
                if (r['score'] > BERT_SCORE) & ("##" not in r['token_str']):
                    token_str = r['token_str']
                    new_sent = mask_sent.replace("[MASK]", token_str)
                    new_temp = masked_temp[i].replace("[MASK]", token_str)
                    new_sent = format_abbr(new_sent)
                    new_temp = format_abbr(new_temp)
                    tests_set.add(new_sent)
                    new_temps.add(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_sent = format_abbr(new_sent)
            new_temp = format_abbr(new_temp)
            new_temps.add(new_temp)
            tests_set.add(new_sent)
        else:
            new_sent = format_abbr(mask_sent)
            tests_set.add(new_sent)
            new_temps.add(format_abbr(masked_temp[i]))
    return tests_set, new_temps


def search_syn(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return set(synonyms)


def pred_sent_by_syn(pred_list, masked_temp, words):
    tests_set = set()
    new_temps = set()
    for i in range(len(pred_list)):
        mask_sent = pred_list[i]
        word = words[i]
        if "[MASK]" in mask_sent:
            print(mask_sent)
            syns = search_syn(word)
            for s in syns:
                new_sent = mask_sent.replace("[MASK]", s)
                new_temp = masked_temp[i].replace("[MASK]", s)
                new_sent = format_abbr(new_sent)
                new_temp = format_abbr(new_temp)
                tests_set.add(new_sent)
                new_temps.add(new_temp)
            new_sent = mask_sent.replace("[MASK]", word)
            new_temp = masked_temp[i].replace("[MASK]", word)
            new_sent = format_abbr(new_sent)
            new_temp = format_abbr(new_temp)
            new_temps.add(new_temp)
            tests_set.add(new_sent)
        else:
            new_sent = format_abbr(mask_sent)
            tests_set.add(new_sent)
            new_temps.add(format_abbr(masked_temp[i]))
    return tests_set, new_temps


def gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open(file_path, mode="a")
    all_tests = []
    for i in range(0, 10):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        w.write(format_abbr(comp) + "\n")
        temp = temp_list[i]
        tests_list = []
        tests_list.append([format_abbr(comp)])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        next_temp_list = []
        for j in range(len(masked_adjunct_list)):
            w.write("insert t" + str(j) + "\n")
            if j == 0:
                pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
                next_temp_list.extend(new_temps)
                for test in new_tests:
                    w.write(test + "\n")
                w.write("\n")
                tests_list.append(new_tests)
            else:
                new_temp_list = []
                tests_list.append([])
                for t in range(len(next_temp_list)):
                    pred_list, masked_temp = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
                    words = masked_word_list[j]
                    new_tests, new_temps = pred_sent_by_syn(pred_list, masked_temp, words)
                    new_temp_list.extend(new_temps)
                    tests_list[-1].extend(new_tests)
                    for test in new_tests:
                        w.write(test + "\n")
                    w.write("\n")
                next_temp_list = new_temp_list
        w.write("FIN\n")
        all_tests.append(tests_list)
    w.close()
    return all_tests


def gen_sent_by_bert(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct):
    w = open(file_path, mode="a")
    all_tests = []
    for i in range(0, 10):
        w.write("sent_id = " + str(i) + "\n")
        comp = comp_list[i]
        w.write(format_abbr(comp) + "\n")
        temp = temp_list[i]
        tests_list = []
        tests_list.append([format_abbr(comp)])
        masked_adjunct_list = all_masked_adjunct[i]
        masked_word_list = all_masked_word[i]
        next_temp_list = []
        for j in range(len(masked_adjunct_list)):
            w.write("insert t" + str(j) + "\n")
            if j == 0:
                pred_list, masked_temp = gen_masked_sent(j, temp, masked_adjunct_list[j])
                words = masked_word_list[j]
                new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
                next_temp_list.extend(new_temps)
                for test in new_tests:
                    w.write(test + "\n")
                w.write("\n")
                tests_list.append(new_tests)
            else:
                new_temp_list = []
                tests_list.append([])
                for t in range(len(next_temp_list)):
                    pred_list, masked_temp = gen_masked_sent(j, next_temp_list[t], masked_adjunct_list[j])
                    words = masked_word_list[j]
                    new_tests, new_temps = pred_sent_by_bert(pred_list, masked_temp, words)
                    new_temp_list.extend(new_temps)
                    tests_list[-1].extend(new_tests)
                    for test in new_tests:
                        w.write(test + "\n")
                    w.write("\n")
                next_temp_list = new_temp_list
        w.write("FIN\n")
        all_tests.append(tests_list)
    w.close()
    return all_tests


if __name__ == '__main__':
    file_name = "business"
    temp_list, adjunct_list, comp_list, ner_list = gen_sent_temp_main(file_name)
    pos_list = ['NOUN', 'VERB', 'ADJ', 'ADV']
    all_masked_word, all_masked_adjunct = gen_mask_phrase(adjunct_list, pos_list, ner_list)
    file_path = "./" + file_name + "_syn_test.txt"
    all_tests = gen_sent_by_syn(file_path, comp_list, temp_list, all_masked_word, all_masked_adjunct)

