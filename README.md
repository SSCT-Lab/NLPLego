### Code refactoring and decoupling

#### gen_tests运行新数据文件

1. comp_input新建对应的两个cln.sent文件
  
2. comp_res新建对应的ncontext_result_greedy.sents
  
3. txt_files新建对应context的txt文件
  
4. gen_tests入口方法中修改file_name和label_path的赋值