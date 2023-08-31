# 此处根据gplearn和自编的backtest回测文件尝试了因子的生成。
import numpy as np
import pandas as pd
from gplearn import fitness
from gp_func import *
from gplearn.genetic import SymbolicRegressor
from datetime import datetime
import graphviz
from gplearn.functions import make_function
from gp_data_enlarge import data_process
import os
import pickle  # 存数据
from backtest import Displayer, BackTester
import talib
import pickle
import pandas as pd
import numpy as np
import time
import dill
if not os.path.exists('./result/factor/'):
    os.makedirs('./result/factor/')
from functools import partial

def score_func_basic(y, y_pred, sample_weight, train_data):  # 适应度函数：策略评价指标
    data = train_data.copy()
    data.index = range(len(data))
    data['sgnl'] = pd.DataFrame(y_pred, columns=['sgnl'])
    _ = bt.run_(factor=data['sgnl'].values)
    # return _['annualized_mean']/_['annualized_std'] if _['annualized_std'] != 0 else 0
    return _['annualized_mean']/_['max_drawdown'] if _['max_drawdown'] != 0 else 0

def score_func_supplement(y, y_pred, sample_weight):  # 加这行是因为fitness有对于参数个数的逻辑判断
    return y

def my_gplearn(metric, random_state = 42):
    return SymbolicRegressor(population_size=pop_num,  # 每一代公式群体中的公式数量 500，100
                              generations=gen_num,  # 公式进化的世代数量 10，3
                              metric=metric,  # 适应度指标，这里是前述定义的通过 大于0做多，小于0做空的 累积净值/最大回撤 的评判函数
                              tournament_size=tour_num,  # 在每一代公式中选中tournament的规模，对适应度最高的公式进行变异或繁殖 50
                              function_set=(gp_delta, gp_signedpower, gp_decayl, gp_delayy, # gp_stdd, gp_rankk, # gp_corrr, gp_covv, # gp_asin, gp_acos, gp_power,
                                            'add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                            'abs', 'neg', 'inv',
                                            'sin', 'cos', 'tan',
                                            gp_andpn, gp_orpn, gp_ltpn, gp_gtpn,
                                            gp_andn, gp_orn, gp_ltn, gp_gtn,
                                            gp_andp, gp_orp, gp_ltp, gp_gtp, gp_if,
                                            'max', 'min',
                                            ),  # 用于构建和进化公式使用的函数集
                              const_range=(-1.0, 1.0),  # 公式中包含的常数范围
                              parsimony_coefficient='auto',
                              # 对较大树的惩罚,默认0.001，auto则用c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between program size l and program fitness f in the population, and Var(l) is the variance of program sizes.
                              stopping_criteria=100.0,  # 是对metric的限制（此处为收益/回撤）
                              init_depth=(2, 3),  # 公式树的初始化深度，树深度最小2层，最大6层
                              init_method='half and half',  # 树的形状，grow生分枝整的不对称，full长出浓密
                              p_crossover=0.8,  # 交叉变异概率 0.8
                              p_subtree_mutation=0.05,  # 子树变异概率
                              p_hoist_mutation=0.05,  # hoist变异概率 0.15
                              p_point_mutation=0.05,  # 点变异概率
                              p_point_replace=0.05,  # 点变异中每个节点进行变异进化的概率

                              max_samples=1.0,  # The fraction of samples to draw from X to evaluate each program on.

                              feature_names=None, warm_start=False, low_memory=False,

                              n_jobs=1,
                              verbose=1,
                              random_state=random_state)



def gp_save_factor(my_cmodel_gp, factor_num=''):
    with open(f'./result/factor/factor{factor_num}.pickle', 'wb') as f:
        dill.dump(my_cmodel_gp, f)
    with open(f'./result/factor/factor{factor_num}.pickle', 'rb') as f:  # 读结果
        factor_rslt = dill.load(f)
    print(factor_rslt)
    
class SymbolicTestor(BackTester):
    def init(self):
        self.params = {'factor': pd.Series}
        
    @BackTester.process_strategy
    def run_(self, *args, **kwargs) -> dict[str: int]:
        factor = np.array(self.params['factor'])
        long_cond = factor > 0
        short_cond = factor < 0
        self.backtest_env['signal'] = np.where(long_cond, 1, np.where(short_cond, -1, np.nan))
        self.construct_position_(keep_raw=True, max_holding_period=1200, take_profit=None, stop_loss=None)

def read_futdata(file_name):
    data_temp = pd.read_csv(f"./data/{file_name}.csv",index_col=0,low_memory=False)
    data_temp.index = pd.to_datetime(data_temp.index)
    data_temp.fillna(method = 'ffill',inplace = True)
    data_temp = data_temp[data_temp.index.isnull() == False]
    return data_temp

if __name__ == '__main__':
    # 读取数据
    comm, random_state, factor_num, pop_num, gen_num,tour_num = [0/10000, 0/10000], 0, 1, 100, 3, 10 # 500, 5, 50 # 1000, 3, 20 # 1000, 15, 100
    train_data = pd.read_csv('./data/IC_train.csv', index_col=0)
    test_data = pd.read_csv('./data/IC_test.csv', index_col=0)
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)

    # 回测环境
    bt = SymbolicTestor(train_data, transact_base='Open',commissions=(comm[0],comm[1])) # 加载数据，根据Close成交,comm是买-卖

    # 生成因子
    score_func_final = partial(score_func_basic, train_data=train_data)
    score_func_final.__code__ = score_func_supplement.__code__
    metric = fitness.make_fitness(function=score_func_final, # function(y, y_pred, sample_weight) that returns a floating point number.
                         greater_is_better=True,  # 上述y是输入的目标y向量，y_pred是genetic program中的预测值，sample_weight是样本权重向量
                         wrap=False)  # 不保存，运行的更快 # gplearn.fitness.make_fitness(function, greater_is_better, wrap=True)
    my_cmodel_gp = my_gplearn(metric, random_state)
    train_data.loc[:,'y'] = np.log(train_data['Open'].shift(-4)/train_data['Open'].shift(-1))
    train_data.dropna(inplace = True)
    my_cmodel_gp.fit(data_process(train_data.loc[:,:'Volume']), train_data.loc[:,'y'].values)
    print(my_cmodel_gp)
    
    # 测试
    factor = my_cmodel_gp.predict(data_process(test_data))
    my_cmodel_gp.predict(data_process(test_data))
    bt_test = SymbolicTestor(test_data, transact_base='Open',commissions=(comm[0],comm[1])) # 加载数据，根据Close成交,comm是买-卖
    bt_test.run_(factor=factor)

    # 策略结果
    md = bt_test.summary()
    md.out_stats.to_clipboard()
    print(md.out_stats)
    md.plot_(comm=comm, show_bool=True)
    bt.fees_factor
    out_stats, holding_infos, trading_details = md.get_results()
    md.save_results(file_name=comm)
    
    # 保存
    gp_save_factor(my_cmodel_gp, factor_num)
    print(f"因子{factor_num}结果已保存 pop_num, gen_num,tour_num={pop_num}, {gen_num},{tour_num}")