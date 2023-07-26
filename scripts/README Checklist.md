# README Checklist

## 需要的数据

1. 句子1（原句）
2. 句子2（原句插入之后的句子）
3. 是否相同的label

> 根据模型跑出来的结果进行判断模型运行出来的结果是否正确，最后我们需要的是模型的错误数据。

## 流程

### 1. run数据

./run/run_deberta_SA.py

输入：checklist的测试数据

输出：模型运行结果

### 2. 预处理

./checklist/pre_deal.py

输入：上一步得到的数据

输出：根据sent_id进行分组之后的数据

### 3. 将数据进行连接

./checklist/join_sentid.py

输入：上一步得到的结果数据，和checklist的测试数据

输出：根据原句和生成数据进行连接，生成新的连接之后的数据。

### 4. 捕获错误

./checklist/catch_failure.py

输入：上一步得到的连接之后的数据

输出：checklist生成的能够捕获错误的数据