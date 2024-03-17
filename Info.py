
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

1. **bank-additional-full.csv**: Contains all examples (41,188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014].
2. **bank-additional.csv**: Contains 10% of the examples (4,119), randomly selected from the first dataset, with 20 inputs.
3. **bank-full.csv**: Contains all examples and 17 inputs, ordered by date (an older version of the dataset with fewer inputs).
4. **bank.csv**: Contains 10% of the examples and 17 inputs, randomly selected from the third dataset (an older version of the dataset with fewer inputs).

The smaller datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
""")

  