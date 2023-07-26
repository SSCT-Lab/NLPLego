#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


df = pd.read_csv("/Users/seedoilz/Desktop/gpt_SA_result_syn.csv")
df['index'] = -1
for index, row in df.iterrows():
    df.loc[index, 'index'] = index


# In[9]:


df.to_csv("/Users/seedoilz/Desktop/gpt_SA_result_syn.csv")


# In[ ]:




