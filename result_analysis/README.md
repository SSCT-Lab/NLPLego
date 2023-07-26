# README

## CSV Structure

### Files in entity folder

#### SSM

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

#### SA

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

### Files in meaning folder

#### SSM

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

#### SA

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

### Files in checklist folder

- original_text

  original text in the dataset

- original_sentiment

  original sentiment (1 stands for positive, 0 stands for negative)

- original_neg_score

  the possibility of negative sentiment of original text

- original_pos_score

  the possibility of positive sentiment of original text

- res_text

  original text combined with the insert text

- res_sentiment

  result sentiment (1 stands for positive, 0 stands for negative)

- res_neg_score

  the possibility of negative sentiment of result text

- res_pos_score

  the possibility of positive sentiment of result text

- sent_id

  the test cases with the same sent_id come from the same seed sentence
