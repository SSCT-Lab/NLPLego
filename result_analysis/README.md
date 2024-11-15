# README

## CSV Structure

### Files in xxx_Res/wmc_res folder

#### mrc_wmc_res.csv

- context

  The context of the question

- origin

  The origin question in the data set

- origin_ans

  The answer of origin question in the data set

- new

  The new question generated by Lego

- new_ans

  The answer of new question generated by NLP model

- answers

  The standard answer set in the data set

- origin_result

  Whether the answer of origin question is correct

- result

  Whether the answer of new question is correct under metamorphic relation

- origin_token

  The token on the position that replaced token takes in the origin question

- replaced_token

  The replaced token

#### sa_wmc_res_xxx.csv

##### DeBERTa

- original_text

  original text in the dataset

- original_sentiment

  original sentiment (1 stands for positive, 0 stands for negative)

- original_neg_score

  the possibility of negative sentiment of original text

- original_pos_score

  the possibility of positive sentiment of original text

- insert_text

  the text inserted in the dataset

- insert_sentiment

  insert text's sentiment (1 stands for positive, 0 stands for negative)

- insert_neg_score

  the possibility of negative sentiment of insert text

- insert_pos_score

  the possibility of positive sentiment of insert text

- res_text

  original text combined with the insert text

- res_sentiment

  result sentiment (1 stands for positive, 0 stands for negative)

- res_neg_score

  the possibility of negative sentiment of result text

- res_pos_score

  the possibility of positive sentiment of result text

- index

  the index in the original dataset

- group_no

  the test cases with the same group_no come from the same seed sentence

##### ChatGPT

> The corresponding relationship with the DeBERTa columns above

- s1 : original_text

- s2 : insert_text

- s3 : res_text

- r1 : original_sentiment

- r2 : insert_sentiment

- r3 : res_sentiment

- v1

  the possibility of sentiment of original text (negative number stands for the negative sentiment possibility)

- v2

  the possibility of sentiment of insert text (negative number stands for the negative sentiment possibility)

- v3

  the possibility of sentiment of result text (negative number stands for the negative sentiment possibility)

#### ssm_wmc_res.csv

- group_no

  the sentences with the same group_no means that they come from the same seed sentence

- original_text

  the seed sentence

- text_a

  One of the sentence in SSM task

- insert_text_a

  the inserted text of text_a compared to the seed sentence

- text_b

  The other sentence in SSM task

- insert_text_b

  the inserted text of text_b compared to the seed sentence

- wrong reason

  why the model result is regarded as wrong

### Files in xxx_Res/ner_res folder

#### mrc_ner_res.csv

- context

  The context of the question

- origin

  The origin question in the data set

- origin_ans

  The answer of origin question in the data set

- new

  The new question generated by Lego

- new_ans

  The answer of new question generated by NLP model

- answers

  The standard answer set in the data set

- origin_result

  Whether the answer of origin question is correct

- result

  Whether the answer of new question is correct under metamorphic relation

- XXX (e.g. CARDINAL, EVENT ...)

  The number of a certain category of the entity

- XXX_text (e.g. CARDINAL_text, EVENT_text ...)

  What the entity is in the sentence (blank or 'blank' stands for null)

- XXX_start_and_end_error(e.g. CARDINAL_start_end_error, EVENT_start_end_error ...)

  Whether the situation the model did not recognize the position of the entity occurred 

#### sa_ner_res_xxx.csv

##### DeBERTa

- original_text

  original text in the dataset

- original_sentiment

  original sentiment (1 stands for positive, 0 stands for negative)

- original_neg_score

  the possibility of negative sentiment of original text

- original_pos_score

  the possibility of positive sentiment of original text

- insert_text

  the text inserted in the dataset

- insert_sentiment

  insert text's sentiment (1 stands for positive, 0 stands for negative)

- insert_neg_score

  the possibility of negative sentiment of insert text

- insert_pos_score

  the possibility of positive sentiment of insert text

- res_text

  original text combined with the insert text

- res_sentiment

  result sentiment (1 stands for positive, 0 stands for negative)

- res_neg_score

  the possibility of negative sentiment of result text

- res_pos_score

  the possibility of positive sentiment of result text

- XXX (e.g. CARDINAL, EVENT ...)

  the number of a certain category of the entity

- XXX_text (e.g. CARDINAL_text, EVENT_text ...)

  what the entity is in the sentence (blank or 'blank' stands for null)

##### ChatGPT

> The corresponding relationship with the DeBERTa columns above

- s1 : original_text

- s2 : insert_text

- s3 : res_text

- r1 : original_sentiment

- r2 : insert_sentiment

- r3 : res_sentiment

- v1

  the possibility of sentiment of original text (negative number stands for the negative sentiment possibility)

- v2

  the possibility of sentiment of insert text (negative number stands for the negative sentiment possibility)

- v3

  the possibility of sentiment of result text (negative number stands for the negative sentiment possibility)

- XXX (e.g. CARDINAL, EVENT ...)

  the number of a certain category of the entity

- XXX_text (e.g. CARDINAL_text, EVENT_text ...)

  what the entity is in the sentence (blank or 'blank' stands for null)

#### sum_ner_res.csv

##### DeBERTa

- id

- text_a

  One of the sentence in SSM task

- text_b

  The other sentence in SSM task

- label

  1 stands for that 2 sentences are same meaning (according to the model)

- XXX (e.g. CARDINAL, EVENT ...)

  the number of a certain category of the entity

- XXX_text (e.g. CARDINAL_text, EVENT_text ...)

  what the entity is in the sentence (blank or 'blank' stands for null)

##### ChatGPT

- q1

  One of the sentence in SSM task

- q2

  The other sentence in SSM task

- gpt

  The answer according to the ChatGPT

- XXX (e.g. CARDINAL, EVENT ...)

  the number of a certain category of the entity

- XXX_text (e.g. CARDINAL_text, EVENT_text ...)

  what the entity is in the sentence (blank or 'blank' stands for null)

### Files in xxx_Res/lr_res folder

#### mrc_lr_res.csv

- context

  The context of the question

- origin

  The origin question in the data set

- origin_ans

  The answer of origin question in the data set

- new

  The new question generated by Lego

- new_ans

  The answer of new question generated by NLP model

- answers

  The standard answer set in the data set

- origin_result

  Whether the answer of origin question is correct

- result

  Whether the answer of new question is correct under metamorphic relation

- coordination

  Whether the new question has coordination dependency

- causality

  Whether the new question has causality dependency

- hypothesis

  Whether the new question has hypothesis dependency

### Files in xxx_Res/sr_res folder

##### DeBERTa

- original_text

  original text in the dataset

- original_sentiment

  original sentiment (1 stands for positive, 0 stands for negative)

- original_neg_score

  the possibility of negative sentiment of original text

- original_pos_score

  the possibility of positive sentiment of original text

- insert_text

  the text inserted in the dataset

- insert_sentiment

  insert text's sentiment (1 stands for positive, 0 stands for negative)

- insert_neg_score

  the possibility of negative sentiment of insert text

- insert_pos_score

  the possibility of positive sentiment of insert text

- res_text

  original text combined with the insert text

- res_sentiment

  result sentiment (1 stands for positive, 0 stands for negative)

- res_neg_score

  the possibility of negative sentiment of result text

- res_pos_score

  the possibility of positive sentiment of result text

- index

  the index in the original dataset

- group_no

  the test cases with the same group_no come from the same seed sentence

##### ChatGPT

> The corresponding relationship with the DeBERTa columns above

- s1 : original_text

- s2 : insert_text

- s3 : res_text

- r1 : original_sentiment

- r2 : insert_sentiment

- r3 : res_sentiment

- v1

  the possibility of sentiment of original text (negative number stands for the negative sentiment possibility)

- v2

  the possibility of sentiment of insert text (negative number stands for the negative sentiment possibility)

- v3

  the possibility of sentiment of result text (negative number stands for the negative sentiment possibility)

### Files in xxx_Res/sm_res folder

#### deberta_ssm_sm.csv

- derivation_sentence_list

  sentences from the same derivation tree which have same meanings
