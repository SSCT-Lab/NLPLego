#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os


for file_path in os.listdir("/Users/seedoilz/Codes/research/sampling"):
    if os.path.isdir(file_path) or file_path.endswith("ipynb") or file_path == '.DS_Store':
        continue
    if file_path.endswith("tsv"):
        df = pd.read_csv(file_path, sep='\t')
        file_path = file_path.replace("tsv", "csv")
    else:
        df = pd.read_csv(file_path)
        continue
    drop_index_list = []
    for index, row in df.iterrows():
        if row['text_a'] == row['text_b']:
            drop_index_list.append(index)
    df = df.drop(drop_index_list)
    print(len(drop_index_list))
    sample_df_list = []
    for i in range(4):
        sample_df = df.sample(n=100)
        sample_df_list.append(sample_df)
        df.drop(sample_df.index, inplace=True)
    dir_path = "./samples_" + file_path.replace(".csv", "")
    os.makedirs(dir_path)
    for i in range(4):
        sample_df_list[0].to_csv(dir_path + "/sample_" + str(i+1) + "_" +file_path, index=False)





