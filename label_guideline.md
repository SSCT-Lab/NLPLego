## Introduction
In this task, you need to determine whether the suspected bug output by NLPLego is a real bug or not. When testing an NLP model, a real bug is a wrong output from the NLP model.
Therefore, the contents of the table we will give are predicted to be cases where the output of the model is erroneous. **You need to determine whether the output of the model actually contains errors based on the input. If in fact the output of the model is correct, then it is NLPLego that is generating false positives.**

Specifically, we will provide the outputs of the model for three types of NLP tasks: 
- machine reading comprehension
- sentiment analysis
- semantic similarity measurement

Next, we will detail how to annotate the outputs of NLPLego based on the type of NLP task.

## False Positive (FP) in Software Testing

**A false positive in software testing occurs when a test case incorrectly indicates the presence of a defect or issue that does not actually exist.**
Here are some examples to illustrate this:
1. **Automated Tests:**
   - An automated test script flags a feature as broken because it misinterprets the expected behavior, but the feature is working correctly.

2. **Static Code Analysis:**
   - A static analysis tool reports a security vulnerability in the code, but upon manual inspection, the code is secure and the warning is incorrect.

In software testing, a false positive means the tests incorrectly report a problem where there is none. Reducing false positives is essential to ensure the efficiency and reliability of the testing process, avoiding unnecessary debugging and investigation.
False positives can happen in various contexts, and here are a few examples to illustrate to help you understand their meanings:
1. **Medical Testing:**
   - A pregnancy test shows a positive result, indicating pregnancy, but the person is not actually pregnant. This is a false positive.

2. **Smoke Detectors:**
   - A smoke detector goes off while you're cooking, even though there is no fire. The detector falsely detects a fire situation.


## Machine Reading Comprehension (MRC)
### What is MRC
```
Machine Reading Comprehension (MRC) is a crucial task in the field of NLP that enables computers to understand and interpret textual information. 
In this task, models are typically given a passage of text along with a related question. 
The goal of the model is to find or generate the correct answer based on the provided text.
```
### Input of MRC
The input to MRC consists of a context and a question. 
The following example shows a context and multiple related questions. 

> Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

> Question: 
> In what country is Normandy located?
> When were the Normans in Normandy?
> What is France a region of?

### Output of MRC
There are two cases here, one is that the question can be answered according to the context, and the other is that the question cannot be answered according to the context.
**In the first case, the output of model is not empty, and the output satisfies the following conditions, then the output is considered to be correct.**
```
1. The answer directly addresses the question
2. The answer is explicitly supported by the context
3. The answer is complete and accurate
```
Here are the correct answers we give to the above questions:
> France
> 10th and 11th centuries 
> *null (The output is empty, indicating that the question cannot be answered according to the context, and the answer is empty.)*

**If the output of the model is null, or if any of the above conditions are not met, the output is considered to be incorrect.** Often, when the model doesn't really understand the question and context, it will give a text that is similar to the correct answer in terms of part of speech, type, etc. This is shown below:
> Roman (Any country name in the context)
> 11th centuries (Incomplete and accurate answer)
> Denmark (The place name closest to the keyword "a region in France")

**In the second case, when given questions that should not be answered, the model gives answers that are not null, that should all be judged as incorrect output.** 

### Description of the table
The first column is the **original context**, the second is **the context after synonym replacement**, the third is the **question**, the fourth is the **correct answer**, and the fifth is **the answer output by the model**.

The sentences marked blue in the first and second columns are the ones that are relevant to the question, that is, the ones that contain the correct answer. The words highlighted in red in the second column are the replaced words.
In the process of labeling, you need to pay attention to whether the replaced words will have an impact on the answer to the question, resulting in the output of the model inconsistent with the correct answer.
**If the output of model is actually correct based on the context after the replacement, it is considered a false positive and you need to fill in FP in the sixth column**.

## Sentiment Analysis (SA)
### What is SA
```
SA interprets sentiments and determines the appropriate sentiment polarity of a given text (positive or negative). 
Depending on the type of text being processed, the core SA tasks are categorized into document-level sentiment classification, sentence-level
sentiment classification, and aspect-level sentiment classification.
```
### Input of SA
The input for SA can be words, phrases, sentences, and document. In this task, the inputs we give are single phrases and single sentences. It is shown below:
```
Input 1:
a beautifully
Input 2:
contrived , well-worn situations
Input 3:
it 's robert duvall !
Input 4:
it does n't follow the stale , standard , connect-the-dots storyline which has become commonplace in movies that explore the seamy underbelly of the criminal world
Input 5:
that 's so sloppily written and cast that you can not believe anyone more central to the creation of bugsy than the caterer
```

### Output of SA
The output is the predefined sentiment label. **A label of 1 indicates that the model considers the sentiment polarity of a given text is positive, and a label of 0 indicates that the model considers sentiment polarity of a given text is negative.**

```
Input 1:
1 (positive)
Input 2:
0 (negative)
Input 3:
1 (positive)
Input 4:
1 (positive) 
Input 5:
0 (negative)
```
The label is inferred from the polarity probability given by the model, which polarity is considered by the model if the probability is greater than 0.5.

### Description of the table
NLPLego is to determine whether the output of the model is wrong by comparing the sentiment change of the sentence before and after inserting the adjunct.
Thus, the first column is the sentence $S_1$ before inserting the adjunct, the second column is the sentiment label of $S_1$ output by the model, the third column is the inserted adjunct, the fourth column is the sentiment label of the adjunct output by the model, the fifth column is the sentence $S_2$ after inserting the adjunct, the sixth colum is the the sentiment label of $S_2$ output by the model, and **the seventh column is the change in the probability corresponding to a certain label before and after the insertion of the adjunct**.

| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 | Column 7 |
|----------|----------|----------|----------|----------|----------|----------|
| It's a attempt .    | 0     | brave     | 0 | It's a brave attempt .| 1 | inc of pos: 0.9904825640842319 |

As the data in the table shows, if a negative adjective is inserted in front of "attemp", the sentence should become more negative, but the model thinks the positivity of the newly generated sentence is increased by 0.99. Through inference, it can be found that the reason is that the model mistakenly thinks "brave" is negative, but its sentiment in the complete sentence is recognized correctly, resulting in a large change in sentiment reverse.
Thus, this line contains a real bug and is not a false positive.
**In this task, a false positive means that the SA results correspond to the contents in the first, third, and fifth columns are correct , and the trend of sentiment change is also correct.**
**It is worth noting that false positives are usually caused by negative words, fixed collocations, and metaphors.**
For example, inseting positive words after negative words does not lead to a rise in the positivity of the newly generated sentences.

## Semantic Similarity Measurement (SSM)
### What is SSM
```
Semantic similarity measurement is a critical aspect of natural language processing (NLP) that focuses on determining how similar two pieces of text are in terms of their underlying meanings. 
This concept is essential for a wide range of applications, including information retrieval, text summarization, question answering, and machine translation.
In our task, the input of SSM is a pair of sentences. 
The output of SSM can be a predefined label that represents the same or different semantics. 
```
### Input of SSM
In the table we have given, the inputs are two interrogatives. 
It is shown below:
```
Input 1:
Q1: What can make Physics easy to learn? Q2: How can you make physics easy to learn?
Input 2:
Q1: How can I be a good geologist? Q2: What should I do to be a great geologist?
Input 3:
Q1: When do you use シ instead of し? Q2: When do you use "&" instead of "and"?
Input 4:
Q1: What is the step by step guide to invest in share market in india? Q2: What is the step by step guide to invest in share market?
```
### Output of SSM
**A label of 1 indicates that the model considers the two questions semantically similar, and a label of 0 indicates that the model considers the two questions semantically dissimilar.**
The output corresponding to the above inputs is shown below:
```
Output1:
1 (Very similar sentence meaning, different expressions)
Output2:
1 (Very similar sentence meaning, different expressions)
Output3:
0 (Although the sentence structure is the same, the key words are completely different and the meaning of the sentence is completely different)
Output4:
0 (The sentence pattern is the same, but Q1 adds a limited range (in india), Q2 does not, resulting in different meanings.)
```
Here, the trick to judge whether the output of the model is correct is by inferring whether two questions would obtain highly similar answers. If the semantics of the answers are close, then the semantics of the questions are also close.

### Description of the table
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
|What is the name for a female ? | What is the most common name for a female ? | 1| 

The first column is **the first question in the input**, the second column is **the second question in the input**, and the third column is **the output of the model**, with **0 indicating different** and **1 indicating the same**.
All of our given inputs are pairs of interrogatives that are presupposed to be semantically inconsistent, and the table gives the cases in which the NLPLego considers the output of model to be incorrect. 
**Since the goal of our task is to find the wrong output of NLPLego, when two questions actually express semantic proximity, they can be considered false positives and require you to mark them as FP in the fourth column of the table.**
