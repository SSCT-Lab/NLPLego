## NLPLego: Assembling Test Generation for Natural Language Processing applications

### Environment

    python 3.6 (The compression model requires python 3.6, if you change to another model, you can switch to a higher version of python.)

    CUDA 10.0 (The compression model requires this version, if you change to another model, you can switch to a higher version)

    cuDNN 7.6 (The compression model requires this version, if you change to another model, you can switch to a higher version)

    nltk with the resource 'stopwords', 'wordnet', 'omw-1.4' and 'averaged_perceptron_tagger'

    spacy3.2 with trained pipelines：en_core_web_lg-3.2.0 (You can use a higher version of SpaCy. the parsing results of different versions, may be slightly different, but it does not matter.)


### Step

    1.   Download stanfordnlp, link is https://stanfordnlp.github.io/CoreNLP

        Make sure you have the Jdk8 environment installed before running, you can run this script as follows:

```
dir: In the JAR package directory
command: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

    2.Select the specified task and specify the start point and end point

    3.You can run this script as follows:

```
python gen_tests.py -T xx -S xx -E xx -CS xx (MRC required) -CE xx (MRC required)

MRC: python gen_tests.py -T MRC -S 0 -E 871 -CS 0 -CE 200 

SA: python gen_tests.py -T SA --S 0 -E 100

SSM: py -T SSM -S 0 -E 100
```

     task can be "MRC","SA","SSM"; S(start_idx) and E(end_idx) should be integer and be within the scope of the data source.

    4.You can find the output in the specified directory

```
dir: ./new_tests/
filename: 
    mrc: dev_mrc_lego_test.json
    sa:  sa_lego_test.tsv
    ssm: qqp_lego_test.tsv
```

### Experiment

##### Dataset
- SQuAD2.0：https://rajpurkar.github.io/SQuAD-explorer/
- GLUE：https://gluebenchmark.com/

##### Model
- ALBERT：https://github.com/google-research/albert
- XLNet：https://github.com/zihangdai/xlnet
- ERNIE2.0：https://github.com/PaddlePaddle/ERNIE
- RoBERTa：https://github.com/pytorch/fairseq
- DeBERTa: https://github.com/microsoft/DeBERTa


### Multidimensional capability assessment

***result_analysis*** holds assessment results of DeBERTa and ChatGPT
Included:

- DeBERTa_Res
	- wmc_res (Word Meaning Comprehension)
	- ner_res (Named Entity Recognition)
	- lr_res (Logical Reasioning)
	- sr_res (Sentiment Recognitn)

- ChatGPT_Res
	- wmc_res
	- ner_res
	- lr_res
	- sr_res
