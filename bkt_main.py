# 总思路：t-2时刻获得仓位(利用了t-2的bar)，在t-1时刻的open进行交易，持有到t时刻open，计算收益
# 规则：
    # 1.计算信号时，可以用当前(t时刻)的bar
    # 2.信号和目标持仓时同期的
    # 3.利用次日开盘进行交易from tqdm import trange
# TODO:删掉每日最后一个交易点点数据避免未来信息（因为不可交易）
import numpy as np
import pandas as pd
import os
from datetime import datetime
if not os.path.exists('./bkt_rslt/'):   # os：operating system，包含操作系统功能，可以进行文件操作
    os.mkdir('./bkt_rslt/') # 如果存在那就是这个result_path，如果不存在那就新建一个
from bkt_sgnl2pos import read_futdata
from bkt_pos2value import pos2value
from bkt_value2sttc import value2sttc_short
from bkt_value2sttc import value2sttc_long
from bkt_sttc2plot import bkt_plot

if __name__ == '__main__':
    start = datetime.now()
    stp_loss = 0.96
    stp_gain = 1.05
    IC0 = read_futdata('IC期货')
    data = IC0.copy()
    data['DateTime'] = data.index
    data.index = range(len(data))

    data['sgnl'] = pd.read_csv('./data/demo_data_of_bkt_train.csv')['sgnl']  # ['sgnl']
    data_score = pos2value(data, data, stop_loss=stp_loss, stop_gain=stp_gain)
    data_score.to_csv('./bkt_rslt/trade_test.csv')  # 1.418033471	1.410876263
    sttc_short = value2sttc_short(data_score)  # 返回收益率/最大回撤
    print("回测得到总收益与最大回撤比值为",round(sttc_short,4))

    trade_detail, sttc_long = value2sttc_long(data_score)
    trade_detail.to_csv('./bkt_rslt/trade_detail.csv')
    sttc_long.to_csv('./bkt_rslt/sttc.csv')
    print("回测统计结果已保存在bkt_rslt文件夹")

    factor_type = ""
    factor_num = ""
    bkt_plot(trade_detail,factor_type,factor_num,path = "./bkt_rslt")
    print('回测分析图已保存在bkt_rslt文件夹')

    end = datetime.now()
    elapsed = end - start
    print("Time elapsed:", elapsed)