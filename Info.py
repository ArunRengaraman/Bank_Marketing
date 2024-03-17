
# coding: utf-8

# In[6]:


import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[7]:


def app():
    st.title("Bank Marketing")
    st.write("Source : https://archive.ics.uci.edu/dataset/222/bank+marketing")
    st.write("""
# Bank Marketing Dataset

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required to assess if the product (bank term deposit) would be subscribed ('yes') or not ('no').

There are four datasets available:

 **bank-full.csv**: Contains all examples and 17 inputs, ordered by date (an older version of the dataset with fewer inputs).


The smaller datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
""")

  
