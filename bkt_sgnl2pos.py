# 此处定义了通过原始信号生成仓位的函数。
# Note：为避免使用未来信息，此处是根据t-1期计算得到的信号值判断t期仓位。
# 交易操作：
    # 开仓：这些函数定义下，若t-1期出现信号，则t期根据信号进行做多（信号为正）或做空（信号为负）操作，无论做多和做空都只持有一手。
    # 调仓：当日内信号符号变化（信号从t期正变为t+1期为负或从负变为正）时进行调仓。
    # 平仓：日内最后一个时点进行平仓。
import numpy as np
import pandas as pd
import os

def read_futdata(file_name):
    data_temp = pd.read_csv("./data/"+file_name+".csv",index_col=0,low_memory=False)
    data_temp.index = pd.to_datetime(data_temp.index)
    data_temp.fillna(method = 'ffill',inplace = True)
    data_temp = data_temp[data_temp.index.isnull() == False]
    return data_temp

def sgnl2pos(sgnl_input, start_time = '09:30:00', end_time = '15:00:00',shift_period = 1):
    sgnl = np.sign(sgnl_input[['sgnl']])
    pos_open_t = pd.Timestamp(start_time).time() # 设置不交易的时间，将信号归零
    pos_end_t = pd.Timestamp(end_time).time()
    sgnl.index = pd.to_datetime(sgnl_input['DateTime']) 
    sgnl.loc[:,'Date'] = sgnl.index.date
    sgnl.loc[:,'Time'] = sgnl.index.time
    sgnl.loc[(sgnl['Time'] <= pos_open_t) | (sgnl['Time'] >= pos_end_t),'sgnl'] = np.nan
    sgnl['sgnl'] = sgnl['sgnl'].shift(1)
    sgnl.loc[(sgnl['sgnl'] == 0),'sgnl'] = np.nan
    sgnl.loc[(sgnl['Time'] <= pos_open_t) | (sgnl['Time'] >= pos_end_t),'sgnl'] = 0 # 收盘仓位归零
    sgnl.fillna(method = 'ffill',inplace = True)
    return sgnl
