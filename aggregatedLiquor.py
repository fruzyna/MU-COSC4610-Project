
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns

sns.set(style="ticks", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Read in aggregated data
liquor = pd.read_csv('data/aggregatedData.csv', index_col='County')
liquor.head()

