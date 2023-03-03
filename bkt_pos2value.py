# 此处定义了仓位生成账户价值的函数。
# Note：为避免使用未来信息，在sgnl2pos函数中根据t-1期计算得到的信号值判断t期仓位，在本文件中，则进一步通过t期仓位根据t期开盘价进行交易。
# 账户价值：
    # 收益率-yield：在yield中，根据开盘价在t期计算计算t-1期投资至t期的收益值。
    # 操作收益-value：在value中，根据仓位pos和yield计算投资获得的收益。
    # 佣金-comm：在comm中，根据仓位pos和yield计算投资获得的收益。
    # 账户净值-value_with_comm：在value_with_comm中，根据value和comm计算考了交易成本的真实收益。
    # Note：在佣金考虑中，股指期货交易收费标准为成交金额的万分之零点二三，其中平今仓手续费为成交金额的万分之三点四五。对于本策略，平今仓定义为先卖后买。
import numpy as np
import pandas as pd
from bkt_sgnl2pos import sgnl2pos

def pos2value(trade_data,ori_sgnl,stop_loss = 0.96, stop_gain = 1.5):
    data_score = trade_data.rename({'future_open':'open','future_high':'high','future_low':'low','future_close':'close'},axis=1)
    data_score['sgnl'] = ori_sgnl['sgnl']
    data_score['pos'] = [*sgnl2pos(data_score)['sgnl']]
    data_score['pos_bak'] = data_score['pos'] # 添加止损、止盈策略前的备份
    Datelst = [x.date() for x in data_score['DateTime']]
    data_score['Date'] = Datelst
    # 止损/止盈思路：记录单独一列是开始投资后的累积收益，默认为1，如果超过(stop_gain)或低于(stop_loss)某一限额就停止
    trade_pos = data_score.loc[(data_score['pos'] != data_score['pos'].shift(1)),:].iloc[1:,:].copy()
    data_score['whether_stop_open'] = trade_pos.loc[trade_pos['pos_bak'] != 0,'open'] * trade_pos['pos_bak']
    data_score['whether_stop_open'] = data_score.groupby('Date')['whether_stop_open'].fillna(method='ffill')
    # 待判断序列
    judge_loc = pd.DataFrame(data_score.loc[data_score['whether_stop_open'].isna() == False,'open']/data_score.loc[data_score['whether_stop_open'].isna() == False,'whether_stop_open'])
    if len(judge_loc):
        judge_loc.loc[(judge_loc[0] > stop_gain) | (judge_loc[0] < stop_loss),'judge'] = 1
        data_score['judge'] = judge_loc['judge']
        data_score['judge'] = data_score.groupby('Date')['judge'].fillna(method='ffill')
        data_score['judge'].fillna(0,inplace = True)
        data_score['pos'] = data_score['pos'].values * (1-data_score['judge'].values)
    data_score['yield'] = data_score['open']/data_score['open'].shift(1)-1 # 这里是表示现在投资在未来能得到的钱
    data_score['value'] = (1 + data_score['pos'].shift(1) * data_score['yield']).cumprod()
    data_score.fillna({'value':1,'yield':0},inplace = True)
    data_score['comm'] = abs(data_score['pos'] - data_score['pos'].shift(1) ) * 0.23 /10000 + ((data_score['pos'].shift(1) == -1) & (data_score['pos'] != -1)) * (3.45-0.23)/10000
    data_score['comm'].fillna(0,inplace = True)
    data_score['value_with_comm'] = data_score['value'] * (1-data_score['comm']).cumprod()
    return data_score

# def pos2value(trade_data,ori_sgnl):
#     data_score = trade_data.rename({'future_open':'open','future_high':'high','future_low':'low','future_close':'close'},axis=1)
#     data_score['sgnl'] = ori_sgnl['sgnl']
#     data_score['pos'] = [*sgnl2pos(data_score)['sgnl']]
#     data_score['yield'] = data_score['open']/data_score['open'].shift(1)-1 # 这里是表示现在投资在未来能得到的钱
#     data_score['value'] = (1 + data_score['pos'].shift(1) * data_score['yield']).cumprod()
#     data_score.fillna({'value':1,'yield':0},inplace = True)
#     data_score['comm'] = abs(data_score['pos'] - data_score['pos'].shift(1) ) * 0.23 /10000 + ((data_score['pos'].shift(1) == -1) & (data_score['pos'] != -1)) * (3.45-0.23)/10000
#     data_score['comm'].fillna(0,inplace = True)
#     data_score['value_with_comm'] = data_score['value'] * (1-data_score['comm']).cumprod()
#     return data_score
