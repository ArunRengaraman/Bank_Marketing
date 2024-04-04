
# coding: utf-8

# In[47]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Info
import Bank
import Visualization
# In[48]:


PAGES={"Details":Info,"Model":Bank,"Visual":Visualization}


# In[49]:
st.sidebar.title("Choose your option to navigate")
selection=st.sidebar.radio("Go to",list(PAGES.keys()))
page=PAGES[selection]
page.app()
