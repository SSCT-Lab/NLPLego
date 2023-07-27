# README MRC

## Required data

1. origin_question (the origin question)
2. context (the corresponding context of the question in squad-v2)
3. question (the new question after mutation)
4. answers (the corresponding answer set in squad-v2)
5. is_leaf_node (whether the new question is a leaf node or not)


## Process

### 1. Entity Recognition

#### 1.1. Run the model

[run_deberta_MRC.py](./run/run_deberta_MRC.py) or [run_t5_MRC.py](./run/run_t5_MRC.py)

Input: Test data

Output: Judgment data from the model, including the origin_answer(result of origin_question), origin_result(whether the answer is correct or not), answer(result of question), result(whether the answer is correct or not according to the metamorphic relation), wrong_reason(reason for why result is not correct)

#### 1.2. Determining bias in named entities

[MRC_entity.py](./entity/MRC_entity.py)

Input: The file generated in part 1.1.

Output: First use the final result to contain all problematic results, each line containing the text and number of each type of named entity, and whether the wrong reason is related to location error

Logic: first judge the entity contained in the answer set, and get the correspoding text of the entity. Then judge whether the wrong reason is related to location error.

### 2. Lexical understanding

#### 2.1. Run the model

[run_deberta_MRC.py](./run/run_deberta_MRC.py) or [run_t5_MRC.py](./run/run_t5_MRC.py)

Input: The file generated in part 2.1.

Output: Judgment data from the model, including the origin_answer(result of origin_question), origin_result(whether the answer is correct or not), answer(result of question), result(whether the answer is correct or not according to the metamorphic relation), wrong_reason(reason for why result is not correct)

#### 2.2. Determine if the lexical understanding is correct or not

[MRC_meaning.py](./meaning/MRC_meaning.py)

Input: 

Output: Wrong answers of model data and corresponding tokens, where the original question's answer is correct, but new question has one token replaced.

### 3. Logic inference

#### 3.1. Run the model

[run_deberta_MRC.py](./run/run_deberta_MRC.py) or [run_t5_MRC.py](./run/run_t5_MRC.py)

Input: Test data

Output: Judgment data from the model, including the origin_answer(result of origin_question), origin_result(whether the answer is correct or not), answer(result of question), result(whether the answer is correct or not according to the metamorphic relation), wrong_reason(reason for why result is not correct)

#### 3.2. Find logic dependencies in wrong answers' questions

[MRC_logic.py](./logic/MRC_logic.py)

Input: The file generated in part 3.1.

Output: The logic dependencies contained in the question, including coordination, causality and hypothesis

Logic: The coordination is judged with stanfordcorenlp, the causality and hypothesis are judged with specific tokens 
