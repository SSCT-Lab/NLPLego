## NLPLego: Assembling Test Generation for Natural Language Processing applications

### Environment

    python3.6

    CUDA10.0

    cuDNN7.6

    nltk with the resource 'stopwords', 'wordnet', 'omw-1.4' and 'averaged_perceptron_tagger'

    spacy3.2 with trained pipelines：en_core_web_lg-3.2.0

    transformers

    pytorch

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
python gen_tests.py --task=xx --start_idx=xx --end_idx=xx

eg: python gen_tests.py --task=sa --start_idx=0 --end_idx=100
```

        and task can be "MRC","SA","SSM"; start_idx and end_idx should be integer and be within the scope of the data source.

    4.You can find the output in the specified directory

```
dir: ./new_tests/
filename: 
    mrc: dev_mrc_lego_test.json
    sa:  sa_lego_test.tsv
    ssm: qqp_lego_test.tsv
```

