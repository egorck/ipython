#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[63]:


data_test = pd.read_csv("C:\\Users\\Egor\\Downloads\\adult_test.csv.zip",compression='zip')
data_test


# In[64]:


data_train= pd.read_csv("C:\\Users\\Egor\\Downloads\\adult_train.csv.zip",compression='zip')


# In[65]:


data_test['Target'].value_counts()


# In[66]:


data_train.at[data_train['Target'] == ' <=50K', 'Target'] = 0
data_train.at[data_train['Target'] == ' >50K', 'Target'] = 1


# In[67]:


data_test.at[data_test['Target'] == ' <=50K.', 'Target'] = 0
data_test.at[data_test['Target'] == ' >50K.', 'Target'] = 1


# # Первичный анализ данных.

# In[68]:


data_test.describe(include='all').T


# Проверяем типы данных

# In[69]:


data_train.dtypes


# In[70]:


data_test.dtypes


# In[73]:


#получение уникальных значений столбца
data_test.Age.unique()


# In[72]:


data_test.drop(0, inplace=True)


# In[74]:


data_test['Age'] = data_test['Age'].astype(int)


# In[75]:


data_test.dtypes


# In[76]:


#Также приведем показатели типа float в int для соответствия train и test выборок.
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)


# Заполним пропуски в количественных полях медианными значениями, а в категориальных – наиболее часто встречающимся значением

# In[77]:


categorical_columns_train = [c for c in data_train.columns 
                             if data_train[c].dtype.name == 'object']
numerical_columns_train = [c for c in data_train.columns 
                           if data_train[c].dtype.name != 'object']

categorical_columns_test = [c for c in data_test.columns 
                            if data_test[c].dtype.name == 'object']
numerical_columns_test = [c for c in data_test.columns 
                          if data_test[c].dtype.name != 'object']


# In[78]:


for c in categorical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].mode())
for c in categorical_columns_test:
    data_test[c] = data_test[c].fillna(data_train[c].mode())
    
for c in numerical_columns_train:
    data_train[c] = data_train[c].fillna(data_train[c].median())
for c in numerical_columns_test:
    data_test[c] = data_test[c].fillna(data_train[c].median())    


# Кодируем категориальные признаки 'Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'. Это можно сделать с помощью метода pandas get_dummies.

# In[79]:


data_train = pd.concat([data_train, pd.get_dummies(data_train['Workclass'], 
                                                   prefix="Workclass"),
                      pd.get_dummies(data_train['Education'], prefix="Education"),
                      pd.get_dummies(data_train['Martial_Status'], prefix="Martial_Status"),
                      pd.get_dummies(data_train['Occupation'], prefix="Occupation"),
                      pd.get_dummies(data_train['Relationship'], prefix="Relationship"),
                      pd.get_dummies(data_train['Race'], prefix="Race"),
                      pd.get_dummies(data_train['Sex'], prefix="Sex"),
                      pd.get_dummies(data_train['Country'], prefix="Country")],
                     axis=1)


# In[80]:


data_test = pd.concat([data_test, pd.get_dummies(data_test['Workclass'], prefix="Workclass"),
                      pd.get_dummies(data_test['Education'], prefix="Education"),
                      pd.get_dummies(data_test['Martial_Status'], prefix="Martial_Status"),
                      pd.get_dummies(data_test['Occupation'], prefix="Occupation"),
                      pd.get_dummies(data_test['Relationship'], prefix="Relationship"),
                      pd.get_dummies(data_test['Race'], prefix="Race"),
                      pd.get_dummies(data_test['Sex'], prefix="Sex"),
                      pd.get_dummies(data_test['Country'], prefix="Country")],
                     axis=1)


# In[83]:


data_train.drop(['Workclass', 'Education', 'Martial_Status',
                 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'],
                axis=1, inplace=True)
data_test.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation', 
                'Relationship', 'Race', 'Sex', 'Country'],
               axis=1, inplace=True)


# In[84]:


data_test.describe(include='all').T


# In[85]:


set(data_train.columns) - set(data_test.columns)


# In[86]:


data_train.shape, data_test.shape


# В тестовой выборке не оказалось Голландии. Заведем необходимый признак из нулей.

# In[87]:


data_test['Country_ Holand-Netherlands'] = np.zeros([data_test.shape[0], 1])


# In[91]:


X_train = data_train.drop(['Target'], axis=1)
y_train = data_train['Target']

X_test = data_test.drop(['Target'], axis=1)
y_test = data_test['Target']


# # Дерево решений без настройки параметров

# In[92]:


tree = DecisionTreeClassifier(max_depth=3,random_state=17)
tree.fit(X_train,y_train)


# In[93]:


tree_predictions = tree.predict(X_test)


# In[94]:


accuracy_score(y_test, tree_predictions)


# # Настроим параметры через кросс-валиацию

# In[95]:


tree_params = {'max_depth': range(2, 11)}
locally_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17),
                                 tree_params, cv=5)                  
locally_best_tree.fit(X_train, y_train)


# In[96]:


locally_best_tree.best_params_


# In[97]:


locally_best_tree.best_score_


# In[98]:


tuned_tree = DecisionTreeClassifier(max_depth=9, random_state=17)
tuned_tree.fit(X_train, y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
accuracy_score(y_test, tuned_tree_predictions)


# # Случайные леса без настройки параметров

# In[99]:


rf = RandomForestClassifier(n_estimators=100, random_state=17)
rf.fit(X_train, y_train)


# In[102]:


from sklearn.model_selection import cross_val_score


# In[103]:


cv_scores = cross_val_score(rf, X_train, y_train, cv=3)


# In[104]:


cv_scores, cv_scores.mean()


# In[105]:


forest_predictions = rf.predict(X_test)


# In[106]:


accuracy_score(y_test,forest_predictions)


# # Настроим параметры

# In[108]:


forest_params = {'max_depth': range(10, 16),
                 'max_features': range(5, 105, 20)}

locally_best_forest = GridSearchCV(
    RandomForestClassifier(n_estimators=10, random_state=17,
                           n_jobs=-1),
    forest_params, cv=3, verbose=1)

locally_best_forest.fit(X_train, y_train)


# In[109]:


locally_best_forest.best_params_


# In[110]:


locally_best_forest.best_score_


# In[111]:


tuned_forest_predictions = locally_best_forest.predict(X_test) 
accuracy_score(y_test,tuned_forest_predictions)

