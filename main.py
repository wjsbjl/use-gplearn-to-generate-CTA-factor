# 此处根据gplearn和自编的backtest回测文件尝试了因子的生成。
import numpy as np
import pandas as pd
from toolkit.backtest import BackTester
from toolkit.DataProcess import load_timing_data
from toolkit.setupGPlearn import gp_save_factor, my_gplearn
import os
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')

def score_func_basic(y, y_pred, sample_weight):  # 因子评价指标
    try:
        _ = bt.run_(factor=y_pred)
        factor_ret = _['annualized_mean']/_['max_drawdown'] if _['max_drawdown'] != 0 else 0 # 可以把max_drawdown换成annualized_std
    except:
        factor_ret = 0
    return factor_ret

class SymbolicTestor(BackTester): # 回测的设定
    def init(self):
        self.params = {'factor': pd.Series}
        
    @BackTester.process_strategy
    def run_(self, *args, **kwargs) -> dict[str: int]:
        factor = np.array(self.params['factor'])
        long_cond = factor > 0
        short_cond = factor < 0
        self.backtest_env['signal'] = np.where(long_cond, 1, np.where(short_cond, -1, np.nan))
        self.construct_position_(keep_raw=True, max_holding_period=1200, take_profit=None, stop_loss=None)

if __name__ == '__main__':
    # 函数集
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log',  # 用于构建和进化公式使用的函数集
                    'abs', 'neg', 'inv', 'sin', 'cos', 'tan', 'max', 'min',      
                    # 'if', 'gtpn', 'andpn', 'orpn', 'ltpn', 'gtp', 'andp', 'orp', 'ltp', 'gtn', 'andn', 'orn', 'ltn', 'delayy', 'delta', 'signedpower', 'decayl', 'stdd', 'rankk'
                    ] # 最后一行是自己的函数，目前不用自己函数效果更好

    # 数据集
    train_data = pd.read_csv('./data/IC_train.csv', index_col=0, parse_dates=[0])
    test_data = pd.read_csv('./data/IC_test.csv', index_col=0, parse_dates=[0])
    feature_names = list(train_data.columns)
    train_data.loc[:,'y'] = np.log(train_data['Open'].shift(-4)/train_data['Open'].shift(-1))
    train_data.dropna(inplace = True)

    # 回测环境（适应度函数）
    comm = [0/10000, 0/10000] # 买卖费率
    bt = SymbolicTestor(train_data, transact_base='Open',commissions=(comm[0],comm[1])) # 加载数据，根据Close成交,comm是买-卖

    # 生成因子
    factor_num = 1 # 因子编号
    my_cmodel_gp = my_gplearn(function_set, score_func_basic, random_state=0, feature_names=feature_names) # 可以通过换random_state来生成不同因子
    my_cmodel_gp.fit(train_data.loc[:,:'rank_num'].values, train_data.loc[:,'y'].values)
    print(my_cmodel_gp)
    
    # 策略结果
    factor = my_cmodel_gp.predict(test_data.values)
    bt_test = SymbolicTestor(test_data, transact_base='Open',commissions=(comm[0],comm[1])) # 加载数据，根据Close成交,comm是买-卖
    bt_test.run_(factor=factor)
    md = bt_test.summary()
    md.out_stats.to_clipboard()
    print(md.out_stats)
    md.plot_(comm=comm, show_bool=True)
    bt.fees_factor
    out_stats, holding_infos, trading_details = md.get_results()
    md.save_results(file_name=comm)
    
    # 保存
    gp_save_factor(my_cmodel_gp, factor_num)
    print(f"因子{factor_num}结果已保存")