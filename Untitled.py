#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
eq = pd.read_csv('C:/Users/98743/Desktop/ECON7950 Business and Economic Forecasting with Big Data/A1_eq.csv')


# In[16]:


import matplotlib.pyplot as plt
ax = eq.plot()
plt.show()


# In[21]:


detrended = signal.detrend(Date)
detrended_df = pd.DataFrame(detrended)
detrended_df.plot()


# In[ ]:


# Student Name: CUI Jiaqi
# Student ID: 22447377


# In[ ]:


# Q1 (a)


# In[23]:


df = pd.read_csv('C:/Users/98743/Desktop/ECON7950 Business and Economic Forecasting with Big Data/A1_eq.csv')
df.Date = pd.to_datetime(df.Date)


# In[24]:


from scipy import signal
detrended = signal.detrend(df.Date)

detrended_df = pd.DataFrame(detrended)
detrended_df.plot()


# In[ ]:


# Q1 (b)


# In[26]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(df.Date)
print(result[1])


# In[ ]:


#The detrended data is stationary, since the p-value of the test is far below 0.05, the null-hypothesis is rejected at 5% significant level.


# In[ ]:


# Q1 (c)


# In[53]:


detrended_df.rename(columns={'0':"y"}, inplace=True)
detrended_df


# In[54]:


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(detrended_df, test_size=0.20, shuffle=False)


# In[56]:


from statsmodels.tsa.ar_model import AutoReg
lag=2
ar_model = AutoReg(data_train['Date'], lags=lag).fit()


# In[34]:


detrended_df['yhat']   = ar_model.predict(1,48, dynamic=False)
detrended_df


# In[ ]:




