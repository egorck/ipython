#!/usr/bin/env python
# coding: utf-8

# Задание 2. 
# Яндекс.Еда осуществляет доставку еды из ресторанов. При этом у каждого ресторана есть зона, в рамках которой осуществляется доставка. Зона представляет собой полигон (заданы координаты его вершин). Пользователь в зависимости от своего местоположения (координат точки) видит разное количество доступных ресторанов. Нам важно, чтобы у каждого пользователя было достаточное количество ресторанов для выбора. Задача заключается в том, чтобы для каждого пользователя посчитать доступное ему количество ресторанов.
# 
# Использовать Python (результат .py или .ipynb файл).
# Данные, которые есть (для простоты в формате .csv, несколько первых строк): user_coordinates.csv (примерно 300 тыс. строк, user_id – идентификатор пользователя)
# user_id,loc_lat,loc_lon
# 1,55.737564,37.345186
# 2,56.234564,37.234590
# 3,55.234578,36.295745
# 
# place_zone_coordinates.csv (примерно 500 тыс. строк,
# place_id – идентификатор ресторана, point_number – порядковый номер вершины полигона)
# place_id,loc_lat,loc_lon,point_number
# 1,55.747022,37.787073,0
# 1,55.751713,37.784328,1
# 1,55.753878,37.777638,2
# 1,55.751031,37.779351,3
# 2,55.803885,37.458311,0
# 2,55.808677,37.464054,1
# 2,55.809763,37.461314,2
# 2,55.810840,37.458654,3
# 
# Формат результата:
# id,number_of_places_available
# 1,2
# 2,19
# 3,0
# 

# In[1]:


import pandas as pd
import numpy as np


# In[13]:


user_df = pd.read_csv('user_coordinates.txt')


# In[14]:


user_df


# In[7]:


zone_df = pd.read_csv('zone_coordinates.txt')


# In[8]:


zone_df


# Добавим функцию для определения принадлежности точки с координатами x(loc_lat) и y(loc_lon) к многоугольнику 

# In[9]:


def inPolygon(x, y, xp, yp):
    c=0
    for i in range(len(xp)):
        if (((yp[i]<=y and y<yp[i-1]) or (yp[i-1]<=y and y<yp[i])) and 
            (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])): c = 1 - c    
    return c


# Ниже функции для получения x и y координат конкретного юзера

# In[10]:


def get_x(user):
    x = user_df[(user_df['user_id']==user)].loc_lat
    return float(x)
def get_y(user):
    y = user_df[(user_df['user_id']==user)].loc_lon
    return float(y)


# Ниже функции для получения массивов координат точек многоугольника

# In[11]:


def get_xp(place):
    xp = zone_df[(zone_df['place_id']==place)].loc_lat
    return xp.to_list()
def get_yp(place):
    yp = zone_df[(zone_df['place_id']==place)].loc_lon
    return yp.to_list()


# In[15]:


#Создадим перечень юзеров
user_list = user_df['user_id'].to_list()


# In[16]:


#Теперь перечень ресторанов
place_list = zone_df['place_id'].unique().tolist()


# In[17]:


# dataframe для записи ответа
answer = pd.DataFrame({'user_id':[],'number_of_places':[]})


# Функция для проверки доступных для юзера ресторанов и запись в ответ

# In[18]:


def mainfunc(user_list,place_list):
    for i in user_list:
        number = 0
        for j in place_list:
            if inPolygon(x=get_x(i),y=get_y(i),xp=get_xp(j),yp=get_yp(j)) == 1:
                number+=1
        answer.loc[i] = {'user_id': i, 'number_of_places': number}
            


# In[19]:


mainfunc(user_list=user_list,place_list=place_list)


# In[20]:


answer

