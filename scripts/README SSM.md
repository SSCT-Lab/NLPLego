# README SSM

## Required data

1. Sentence 1 (original sentence)
2. Sentence 2 (the sentence after the original sentence is inserted)
3. whether the label is the same or not

> According to the results of the model run out to determine the model run out of the results are correct, the last thing we need is the model of the error data.

## Process

### 1. Entity Recognition

#### 1.1. Run the model

[qqp.py](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b09e95db-846a-49c9-a788-c7a5fad9f1b4/qqp.py)

Input: test data

Output: error judgment data from the model.

#### 1.2. Determining bias in named entities

[likelihood_entity.py](. /entity/likelihood_entity.py)

Input: the file generated in part 1.1.

Output: first use the final result to contain all problematic results, each line containing the text and number of each type of named entity.

Logic: first judge the insertion words according to the original sentence, then judge whether the insertion words contain entities, if so, save the data of the line.

### 2. Lexical understanding

#### 2.1. Run the model

[qqp.py](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b09e95db-846a-49c9-a788-c7a5fad9f1b4/qqp.py)

Input: test data

Output: error judgment data in the model.

#### 2.2. Determine if the lexical understanding is correct or not

[SSM_meaning.py](. /meaning/SSM_meaning.py)

Input: data generated in 2.1

Output: the final result file

Logic: for statements with the same seed clause (same original clause, different insertions), group them together. On this basis, two by two comparison is made, if the model judges that the two insertions are semantically the same, but the 2 complete sentences are not semantically the same, then it is judged to be the wrong data and saved. If the model judges that 2 complete sentences have the same semantics, but 2 insertions do not have the same semantics, then it is also wrong data to be saved.

Translated with www.DeepL.com/Translator (free version)