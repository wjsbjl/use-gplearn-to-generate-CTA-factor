# 此处根据gplearn和自编的backtest回测文件尝试了因子的生成。
import numpy as np
import pandas as pd
import dill
from toolkit.backtest import BackTester
from toolkit.DataProcess import load_selecting_data
from toolkit.setupGPlearn import gp_save_factor, my_gplearn
from scipy.stats import spearmanr, pearsonr
from toolkit.IC import get_ic, calculate_ic
import os
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')

def score_func_basic(y, y_pred, sample_weight):  # 因子评价指标
    if len(np.unique(y_pred[-1])) <= 10: # 没办法分组
        # print(len(np.unique(y_pred)))
        ic = -1
    else:
        corr_df = pd.DataFrame(y).corrwith(pd.DataFrame(y_pred), axis=1, method = 'spearman')
        ic = abs(corr_df.mean())
    return ic if not np.isnan(ic) else 0 # pearson

if __name__ == '__main__':
    # 函数集
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log',  # 用于构建和进化公式使用的函数集
                    'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 'max', 'min',      
                    # 'if', 'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk'
                    ] # 最后一行是自己的函数，目前不用自己函数效果更好

    # 数据集
    with open(f'./data/stock_data.pickle', 'rb') as f:
        price = dill.load(f)
    # price = pd.read_csv('./data/stock_data/fadj_close.csv',index_col=[0],parse_dates=[0])
    ret5 = np.log(price.shift(-5) / price.shift(-1)) # 6611天，3553只股
    ret10 = np.log(price.shift(-10) / price.shift(-1))
    y_ret = ret5
    with open(f'./data/factor_data.pickle', 'rb') as f:
        x_dict = dill.load(f)
    x_dict.keys() # 就只要turnover_21和avg_volume_63吧
    feature_names = list(x_dict.keys())
    x_array = np.array(list(x_dict.values()))
    x_array = np.transpose(x_array, axes=(1, 2, 0))
    
    # 生成因子
    factor_num = 2 # 因子编号
    my_cmodel_gp = my_gplearn(function_set, score_func_basic, feature_names=feature_names, pop_num=50, gen_num=3, random_state=0) # 可以通过换random_state来生成不同因子
    my_cmodel_gp.fit(x_array, np.array(y_ret))
    print(my_cmodel_gp)
    
    # 策略结果
    y_pred = my_cmodel_gp.predict(x_array)
    y_pred = pd.DataFrame(y_pred,index = price.index, columns = price.columns)
    IC_df, IC_statistic = get_ic(y_pred, ret5, ret10)
    
    # 保存
    gp_save_factor(my_cmodel_gp, factor_num)
    print(f"因子{factor_num}结果已保存")