#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# ## 病人信息数据读取

# In[57]:


# 将病人的信息以及对应的疾病分别读取到两个列表中
info = []
label = []
with open('./hf_round1_label.txt', 'r', encoding = 'utf-8') as f:
    for l in f.readlines():
        row = l.rstrip().split('\t')
        info.append(row[:3])
        label.append(row[3:])


# ### 读取病人信息

# In[58]:


df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
df_info.head(5)


# ### 读取疾病列表
# 
# 疾病列表将作为疾病标签的列

# In[59]:


df_arrythmia = pd.read_csv('./hf_round1_arrythmia.txt', header=None, sep='\t')
col = pd.Index(df_arrythmia[0])
col


# ### 读取one-hot疾病标签

# In[60]:


# 创建一个map函数将label内每个病人的疾病标签转变为Series的index，value为1
s = lambda x: pd.Series(1, index=x)
s_mapping = map(lambda x: pd.Series(1, index=x), label)

# 将列表中的Series合并转为DataFrame
df_label = pd.DataFrame(list(s_mapping), columns=col).fillna(0)
df_label.head()


# In[61]:


# 核对两个表格的尺寸
df_info.shape, df_label.shape


# ### 合并表格

# In[62]:


df_train = pd.concat([df_info, df_label], axis=1)
df_train.head()


# ## ECG信息读取

# ### 多进程读取ECG数据

# In[75]:


import multiprocessing
from multiprocessing import Process, Queue, Manager


# In[76]:


import pdb
pdb.set_trace()

def load_data(queue, mdict, dtype='train'):
    '''
    采用numpy直接把ECG的txt文件读取为array格式
    queue -- 为多线程队列，这里存放所有ECG文件名
    mdict -- 为多线程库中Manager的字典格式，用于多个线程内的数据共享
    dtype -- 选择train或者test，默认为train
    '''
    while not queue.empty():
        # 去除队列中的序号以及文件名
        index, file = queue.get()
        # 读取文件为numpy array
        data = np.loadtxt('./{}/{}'.format(dtype, file), skiprows=1)
        # 把数据保存到Manager字典中
        mdict[index] = data
        
def file_queue(flist):
    '''
    按传入的列表顺序生成队列，保存序号以及文件名
    '''
    queue = Queue()
    for i, f in enumerate(flist):
        queue.put((i, f))
    return queue

def gen_arr_dict(queue, n_pro=4, dtype='train'):
    arr_dict = {}                        # 先声明一个字典作为最后的输出
    with Manager() as m:                 # 调用multiprocessing库中的Manager进行多进程数据共享
        mdict = m.dict()                 # 使用字典格式（因为要用key记录文件的顺序）
        ps = []
        for i in range(n_pro):           # 写个循环生成子进程，以下为使用python使用多进程的标准格式
            p = Process(target=load_data, args=(queue, mdict, dtype)) 
            ps.append(p)
            p.start()
            
        for p in ps:
            p.join()
            
        for i, k in mdict.items():        # 由于Manager字典不能直接作为输出，需要另外复制一份字典
            arr_dict[i] = k  
    return arr_dict


# In[77]:


f_list = df_info['file'].tolist()[:400]    # 生成文件名列表，这里演示取400条数据进行读取
q_4 = file_queue(f_list)
q_1 = file_queue(f_list)
q_4.qsize(), q_1.qsize()


# In[78]:


multiprocessing.cpu_count()                # 查看机器的CPU数，用于确定进程数


# In[79]:


get_ipython().run_cell_magic('time', '', 'arr_dict_4 = gen_arr_dict(q_4, n_pro=4)    # 4个进程同时读取\nlen(arr_dict_4)')


# In[68]:


get_ipython().run_cell_magic('time', '', 'arr_dict_1 = gen_arr_dict(q_1, n_pro=1)    # 1个进程读取，时间多了一倍')


# ### 合并及数据处理

# 以上步骤生成的是一个存放所有numpy array的字典，接下来我们需要将字典转化成一个单独的numpy array，把比赛介绍中的另外四项数据加入进去，结果会是一个(n, 5000, 12)的numpy array

# In[69]:


def ecg_aggr(arr_dict):
    n = len(arr_dict)                                # 字典中存放的array数
    data = np.zeros((n, 5000, 12))                   # 先生成一个0数列
    for i in range(n):
        arr = arr_dict[i]
        data[i,:,:arr.shape[1]] = arr                # 按顺序将前8列的数据填入
    data[:,:,-4] = data[:,:,1] - data[:,:,0]         # 依次计算9~12列的数据
    data[:,:,-3] = -(data[:,:,0] + data[:,:,1])/2
    data[:,:,-2] = data[:,:,0] - data[:,:,1]/2
    data[:,:,-1] = data[:,:,1] - data[:,:,0]/2
    return data 


# In[70]:


data = ecg_aggr(arr_dict_4)


# In[71]:


data.shape


# ### ECG图像生成

# In[72]:


fig = plt.figure(figsize=(20,80), dpi=100)
for i in range(8):
    ax = fig.add_subplot(12,1,i+1)
    ax.plot(data[0,:,i])
plt.show()


# In[ ]:




