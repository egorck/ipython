#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


income_data = pd.read_csv("C:\\Users\\79771\\Downloads\\222_472_compressed_adult.csv.zip", compression='zip')
income_data


# In[7]:


income_data.info()


# In[35]:


income_data.shape

Сколько мужчин и женщин (признак sex) представлено в этом наборе данных?
# In[8]:


income_data['gender'].value_counts()

Каков средний возраст (признак age) женщин?
# In[14]:


income_data[income_data['gender']== 'Female']['age'].mean()

 Какова доля граждан Германии (признак native-country)?
# In[29]:


float( (income_data['native-country'] == 'Germany').sum() ) / income_data.shape[0]

 Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает более 50K в год (признак salary) и тех, кто получает менее 50K в год?
# In[55]:


age1 = income_data[income_data['income']=='<=50K']['age']
age2 = income_data[income_data['income']=='>50K']['age']


# In[56]:


print(f'The average age of the first group is {round(age1.mean())} +- {round(age1.std())} years')


# In[57]:


print(f'The average age of the second group is {round(age2.mean())} +- {round(age2.std())} years')

7. Выведите статистику возраста для каждой расы (признак race) и каждого пола. Используйте groupby и describe. Найдите таким образом максимальный возраст мужчин расы Amer-Indian-Eskimo.
# In[62]:


income_data.groupby(['gender','race'])['age'].max()

 Какое максимальное число часов человек работает в неделю (признак hours-per-week)? Сколько людей работают такое количество часов и каков среди них процент зарабатывающих много?
# In[67]:


income_data[income_data['hours-per-week']==99].shape[0]

Посчитайте среднее время работы (hours-per-week) зарабатывающих мало и много (salary) для каждой страны (native-country).
# In[81]:


df1 = income_data[income_data['income']=='<=50K']
df2 = income_data[income_data['income']=='>50K']


# In[80]:


df1.pivot_table(['hours-per-week', 'income'], 
['native-country'], aggfunc='mean')


# In[82]:


df2.pivot_table(['hours-per-week', 'income'], 
['native-country'], aggfunc='mean')


# In[ ]:




