# 此处根据账户价绘制相关图表。
# 输入内容包括：
    # value2sttc_long输出的data_score（也就是长版统计量的详细信息）
# 输出内容包括：
    # 账户价值：净值线和buy and hold线
    # 仓位：1, 0, -1
    # Drawdown
    # 逐笔交易时间直方图
    # 逐笔交易日期直方图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)
if os.name == 'posix': # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC'] #中文字体为宋体
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 在windows系统下显示微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212

def pos_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    fig,ax = plt.subplots(figsize = (20,10))
    ax.plot(long_data['value_with_comm'], label = "策略收益", c='tab:orange',lw = 2)
    ax.plot(long_data['open']/long_data['open'][0], label = "Buy and Hold", c='tab:blue',lw = 2)
    ax.legend()
    ax.grid()
    ax.set_xlabel('时间', fontsize = 30)
    ax.set_ylabel('账户价值', fontsize = 30)
    ax.legend(fontsize = 25,loc = 'best')
    plt.savefig(f"{path}/factor{factor_num}_{factor_type}_account_value.jpg", bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    if plt_show:
        plt.show()

def mxdd_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    maxdd = long_data['value_with_comm']/long_data['value_with_comm'].cummax()-1
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(maxdd, label = "最大回撤比率", c='tab:purple',lw = 2)
    ax.grid()
    ax.set_xlabel('时间', fontsize = 30)
    ax.set_ylabel('账户价值', fontsize = 30)
    ax.legend(fontsize = 25,loc = 'best')
    plt.savefig(f"{path}/factor{factor_num}_{factor_type}_max_dd.jpg", bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    if plt_show:
        plt.show()

def per_return_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    return_hist = long_data['return_per_trade'].dropna().values
    fig,ax = plt.subplots(figsize = (20,10))
    ax.hist(return_hist, bins=30, density=False,  color='tab:blue') # alpha=0.5,
    ax.set_xlabel('逐笔交易对应收益', fontsize = 30)
    ax.set_ylabel('频数', fontsize = 30)
    plt.savefig(f"{path}/factor{factor_num}_{factor_type}_per_return_plot.jpg", bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    if plt_show:
        plt.show()

def per_date_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    pos_analysis = long_data.dropna()
    Date_Num = [x.date() for x in pd.to_datetime(pos_analysis['DateTime'].values)]

    fig,ax = plt.subplots(figsize = (20,10))
    ax.hist(Date_Num, bins=30, density=False,  color='tab:blue') # alpha=0.5,
    ax.set_xlabel('逐笔交易对应日期', fontsize = 30)
    ax.set_ylabel('频数', fontsize = 30)
    plt.savefig(f"{path}/factor{factor_num}_{factor_type}_per_date_plot.jpg", bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    if plt_show:
        plt.show()

def per_time_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    pos_analysis = long_data.dropna()
    Time_Num = pos_analysis['DateTime'].apply(lambda t: pd.Timestamp(t).time()).value_counts()

    fig, ax = plt.subplots(figsize=(10, 10))
    hist_x = [str(x) for x in Time_Num.index]
    hist_y = Time_Num.values
    ax.bar(hist_x, hist_y)
    ax.set_xlabel('逐笔交易对应时间',fontsize = 30)
    ax.set_ylabel('频数',fontsize = 30)
    plt.savefig(f"{path}/factor{factor_num}_{factor_type}_per_time_plot.jpg", bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
    if plt_show:
        plt.show()

def bkt_plot(long_data,factor_type,factor_num,plt_show = 0,path = "./gp_rslt/"):
    pos_plot(long_data,factor_type,factor_num,plt_show,path)
    mxdd_plot(long_data,factor_type,factor_num,plt_show,path)
    per_return_plot(long_data,factor_type,factor_num,plt_show,path)
    per_date_plot(long_data,factor_type,factor_num,plt_show,path)
    per_time_plot(long_data,factor_type,factor_num,plt_show,path)
