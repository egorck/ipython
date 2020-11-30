#!/usr/bin/env python
# coding: utf-8

# Задача 1. Минимизация гладкой функции
# 
# Рассмотрим все ту же функцию из задания по линейной алгебре: f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2), но теперь уже на промежутке [1, 30]
# 

# In[1]:


import numpy as np
from scipy.optimize import minimize
from matplotlib import pylab as plt

def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)

x1 = 1
x2 = 30

x = np.linspace(1.0, 30.0)
y = f(x)

print(x,'\n',y)
plt.plot (x,y)
plt.show


# In[2]:


f1 = open('submission-1.txt' , 'w')
res11 = minimize(f,np.array(2.0), method='BFGS')
res12 = minimize(f,np.array(30.0), method='BFGS')
print(res11)
print(res12)
f1.write(str(np.round(res11.fun, decimals=2)) + ' ' +
         str(np.round(res12.fun, decimals=2)))


# Задача 2. Глобальная оптимизация
# Теперь попробуем применить к той же функции f(x) метод глобальной оптимизации — дифференциальную эволюцию.
# 

# In[3]:


from scipy.optimize import differential_evolution

bounds = [(x1,x2)]

res2 = differential_evolution(f,bounds)

f2 = open('submission-2.txt' , 'w')
print(res2)
f2.write(str(np.round(res2.fun[0], decimals=2)))

print(res2)


# Задача 3. Минимизация негладкой функции
# 
# Теперь рассмотрим функцию h(x) = int(f(x)) на том же отрезке [1, 30], т.е. теперь каждое значение f(x) приводится к типу int и функция принимает только целые значения.
# 

# In[4]:


def h(x):
    return np.int(f(x))
y = np.vectorize(h)
plt.plot(x,y(x))
plt.show()


# In[7]:



f3 = open('submission-3.txt' , 'w')
min_BFGS = minimize(h,30.0,method='BFGS')
print(min_BFGS)

min_dif = differential_evolution(h,bounds)
print(min_dif)
f3.write('str(min_BFGS.fun) + str(min_dif.fun[0])')


# In[ ]:




