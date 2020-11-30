#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[142]:


df_train = pd.read_csv('House_sales_train.csv')
df_train.head(3)


# In[143]:


df_test = pd.read_csv('House_prices_test.csv')
df_test.head(3)


# In[144]:


test_id = df_test.Id


# In[140]:


df_train.shape


# In[5]:


df_test.shape


# In[96]:


y = df_train.SalePrice
y


# In[6]:


df_all = pd.concat((df_test, df_train)).reset_index(drop=True)


# In[7]:


df_all.drop('SalePrice', axis=1, inplace=True)
df_all.head(3)


# # Категориальные фичи

# In[8]:


categorical_features = [feature for feature in df_all.columns 
                        if df_all[feature].dtype == 'O']

print(f'Number of categorical variables: {len(categorical_features)}')

df_train[categorical_features].head(3)


# In[9]:


categorical_with_nan = [feature for feature in df_all[categorical_features]
                       if df_all[feature].isnull().any()]


# In[10]:


len(categorical_with_nan)


# In[11]:


for feature in categorical_with_nan:
    print(f'{feature}: \t {np.around(df_all[feature].isnull().mean()*100, 2)}%  missing value')


# # Числовые фичи

# In[12]:


numerical_features = [feature for feature in df_all.columns 
                      if df_all[feature].dtypes != 'O']

print('Number of numerical features: ', len(numerical_features))

df_all[numerical_features].head(3)


# In[13]:


numerical_with_nan = [feature for feature in df_all[numerical_features] 
                      if df_all[feature].isnull().sum()
                      and df_all[feature].dtypes != 'O']

numerical_with_nan


# In[14]:


for feature in numerical_with_nan:
    print(f'{feature}: \t {np.around(df_all[feature].isnull().mean()*100, 2)}% missing value')


# ## Дискретные значения:

# In[15]:


discrete_features = [feature for feature in numerical_features 
                    if len(df_all[feature].unique())<25 
                    and feature not in ['Id']]

print('Discrete Feature Count: ', len(discrete_features))

df_all[discrete_features].head(3)


# ## Непрерывные значения:

# In[16]:


continuous_features = [feature for feature in numerical_features 
                      if feature not in discrete_features + ['Id']]

print(f'Continuous Features Count:  {len(continuous_features)}')

df_all[continuous_features].head(3)


# ## Посмотрим на распределения непрерывных величин

# In[17]:


for feature in continuous_features:
    df_all = df_all.copy()
    df_all[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title('Count vs ' + feature)
    plt.show()


# # Отбор фич

# In[18]:


def display_only_missing(df):
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print(missing_data)


# In[19]:


display_only_missing(df_train)


# In[20]:


df_train.PoolQC.value_counts()


# In[21]:


df_train.drop("PoolQC", axis=1, inplace=True)
df_test.drop("PoolQC", axis=1, inplace=True)


# In[22]:


df_train.MiscFeature.value_counts()


# In[23]:


df_train.drop("MiscFeature", axis=1, inplace=True)
df_test.drop("MiscFeature", axis=1, inplace=True)


# In[24]:


df_train.Alley.value_counts()


# In[25]:


df_train.drop("Alley", axis=1, inplace=True)
df_test.drop("Alley", axis=1, inplace=True)


# In[26]:


df_train.Fence.value_counts()


# In[27]:


df_train.drop("Fence", axis=1, inplace=True)
df_test.drop("Fence", axis=1, inplace=True)


# In[28]:


df_train.FireplaceQu.value_counts()


# In[29]:


import seaborn as sns

sns.countplot(df_train["FireplaceQu"])


# In[30]:


sns.boxplot(data=df_train, x="SalePrice", y="FireplaceQu")


# In[31]:


df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna(0)
df_test["FireplaceQu"] = df_test["FireplaceQu"].fillna(0)


# In[32]:


sns.boxplot(data=df_train, x="SalePrice", y="FireplaceQu")


# In[33]:


sns.regplot(data=df_train, x="SalePrice",y="LotFrontage")


# In[34]:


df_train["LotFrontage"] = df_train["LotFrontage"].fillna( df_train["LotFrontage"].median())
df_test["LotFrontage"] = df_test["LotFrontage"].fillna( df_train["LotFrontage"].median())


# In[35]:


df_train.GarageQual.value_counts()


# In[36]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageQual")


# In[37]:


df_train.drop("GarageQual", axis=1, inplace=True)
df_test.drop("GarageQual", axis=1, inplace=True)


# In[38]:


df_train.GarageFinish.value_counts()


# In[39]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageFinish")


# In[40]:


df_train["GarageFinish"] = df_train["GarageFinish"].fillna("NoGarage")
df_test["GarageFinish"] = df_test["GarageFinish"].fillna("NoGarage")


# In[41]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageFinish")


# In[42]:


df_train.GarageCond.value_counts()


# In[43]:


df_train["GarageCond"] = df_train["GarageCond"].fillna("NoGarage")


# In[44]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageCond")


# In[45]:


df_train.drop("GarageCond", axis=1, inplace=True)
df_test.drop("GarageCond", axis=1, inplace=True)


# In[46]:


sns.distplot(df_train.GarageYrBlt)


# In[47]:


sns.regplot(data=df_train,x="SalePrice",y="GarageYrBlt")


# In[48]:


#Заполним минимумом
df_train["GarageYrBlt"] = df_train["GarageYrBlt"].fillna(df_train.GarageYrBlt.min())
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].fillna(df_train.GarageYrBlt.min())


# In[49]:


df_train.GarageType.value_counts()


# In[50]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageType")


# In[51]:


df_train["GarageType"] = df_train["GarageType"].fillna("NoGarage")
df_test["GarageType"] = df_test["GarageType"].fillna("NoGarage")


# In[52]:


sns.boxplot(data=df_train, x="SalePrice", y="GarageType")


# In[53]:


df_train.BsmtQual.value_counts()


# In[54]:


sns.boxplot(data=df_train, x="SalePrice", y="BsmtQual")


# In[55]:


df_train["BsmtQual"] = df_train["BsmtQual"].fillna("NoBsmt")
df_test["BsmtQual"] =df_test["BsmtQual"].fillna("NoBsmt")


# In[56]:


df_train.drop("BsmtCond", axis=1, inplace=True)
df_test.drop("BsmtCond", axis=1, inplace=True)


# In[57]:


df_train["BsmtExposure"] = df_train["BsmtExposure"].fillna("NoBsmt")
df_test["BsmtExposure"] =df_test["BsmtExposure"].fillna("NoBsmt")


# In[58]:


df_train.BsmtFinType1.value_counts()


# In[59]:


sns.boxplot(data=df_train, x="SalePrice", y="BsmtFinType1")


# In[60]:


df_train["BsmtFinType1"] = df_train["BsmtFinType1"].fillna("NoBsmt")
df_test["BsmtFinType1"] =df_test["BsmtFinType1"].fillna("NoBsmt")


# In[61]:


df_train.drop("BsmtFinType2", axis=1, inplace=True)
df_test.drop("BsmtFinType2", axis=1, inplace=True)


# In[62]:


df_train.MasVnrType.value_counts()


# In[63]:


sns.boxplot(data=df_train, x="SalePrice", y="MasVnrType")


# In[64]:


df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_test["MasVnrType"] =df_test["MasVnrType"].fillna("None")


# In[65]:


df_train["Electrical"] = df_train["Electrical"].fillna("SBrkr")
df_test["Electrical"] =df_test["Electrical"].fillna("SBrkr")


# In[66]:


df_train.MSZoning.value_counts()


# In[67]:


sns.boxplot(data=df_train, x="SalePrice", y="MSZoning")


# In[68]:


df_train["MSZoning"] = df_train["MSZoning"].fillna("RL")
df_test["MSZoning"] = df_test["MSZoning"].fillna("RL")


# In[69]:


df_test.Functional.value_counts()


# In[70]:


df_train.Functional.value_counts()


# In[71]:


sns.boxplot(data=df_train, x="SalePrice", y="Functional")


# In[72]:


df_test["Functional"] =df_test["Functional"].fillna("Typ")


# In[73]:


df_train["Functional"].isnull().any()


# In[74]:


df_test["BsmtFullBath"] =df_test["BsmtFullBath"].fillna(0)


# In[75]:


df_test["BsmtHalfBath"] =df_test["BsmtHalfBath"].fillna(0)


# In[76]:


df_test.Utilities.value_counts()


# In[77]:


df_train.Utilities.value_counts()


# In[78]:


df_test.SaleType.value_counts()


# In[79]:


df_train.SaleType.value_counts()


# In[80]:


sns.boxplot(data=df_train, x="SalePrice", y="SaleType")


# In[81]:


df_test["SaleType"] =df_test["SaleType"].fillna("WD")


# In[82]:


sns.distplot(df_test.GarageArea)


# In[83]:


#Заполним минимумом.

df_test["GarageArea"] =df_test["GarageArea"].fillna(df_test.GarageArea.min())


# In[84]:


df_test["GarageCars"] =df_test["GarageCars"].fillna(df_test.GarageCars.min())


# In[85]:


df_test.KitchenQual.value_counts()


# In[86]:


df_test["KitchenQual"] =df_test["KitchenQual"].fillna("TA")


# In[87]:


sns.distplot(df_test.TotalBsmtSF)


# In[88]:


df_test["TotalBsmtSF"] =df_test["TotalBsmtSF"].fillna(df_test.TotalBsmtSF.min())


# In[89]:


df_test["BsmtUnfSF"] =df_test["BsmtUnfSF"].fillna(df_test.BsmtUnfSF.min())


# In[90]:


df_test["BsmtFinSF2"] =df_test["BsmtFinSF2"].fillna(df_test.BsmtFinSF2.min())


# In[91]:


df_test["BsmtFinSF1"] =df_test["BsmtFinSF1"].fillna(df_test.BsmtFinSF1.min())


# In[92]:


df_test["Exterior1st"] =df_test["Exterior1st"].fillna("VinlSd")


# In[93]:


df_test["Exterior2nd"] =df_test["Exterior2nd"].fillna("VinlSd")


# # Моделирование

# In[111]:


df_train.columns[df_train.dtypes == "object"]


# In[112]:


from sklearn.preprocessing import LabelEncoder
for col in df_train.columns[df_train.dtypes == "object"]:
    df_train[col] = df_train[col].factorize()[0]
df_train = df_train.drop(["Id"], axis=1)


# In[128]:


for col in df_test.columns[df_test.dtypes == "object"]:
    df_test[col] = df_test[col].factorize()[0]


# In[115]:


X = df_train.drop('SalePrice', axis=1)
X.head(3)


# In[116]:


y = df_train.SalePrice


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33 ,random_state = 42)


# In[119]:


from xgboost.sklearn import XGBRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
random_grid={'learning_rate':[0.001,0.01],
            'max_depth':[10,30],
            'n_estimators':[200,300],
            'subsample':[0.5,0.7]
}
xgb = XGBRegressor(objective='reg:linear')
grid_search=GridSearchCV(estimator=xgb,param_grid = random_grid,cv = 3, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error')


# In[120]:


grid_search.fit(X_train,y_train)
print("\nGrid Search Best parameters set :")
print(grid_search.best_params_)


# In[133]:


sns.scatterplot(y_test,pred)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.show()


# In[134]:


Test_predict=grid_search.predict(df_test)


# In[135]:


prediction = pd.DataFrame(Test_predict, columns=['SalePrice'])

