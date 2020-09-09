#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


u = "http://bit.ly/w-data"
s_data = pd.read_csv(u)
print("Data imported successfully")

s_data.head(10)


# In[4]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[5]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  
print(X)
print(y)


# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[7]:


from sklearn.linear_model import LinearRegression  
regression = LinearRegression()  
regression.fit(X_train, y_train) 

print("Training complete.")


# In[8]:


line = regression.coef_*X+regression.intercept_

plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[10]:


print(X_test) 
y_pred = regression.predict(X_test)
print(y_pred)


# In[11]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[12]:


hours = 9.25
own_pred = regression.predict(hours.reshape(-1,1))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[13]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




