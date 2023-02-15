# 此处根据gplearn和自编的backtest回测文件尝试了因子的生成。
import numpy as np
import pandas as pd
from gplearn import fitness
from gplearn.genetic import SymbolicRegressor
from datetime import datetime
from bkt_sgnl2pos import read_futdata
from bkt_pos2value import pos2value
from bkt_value2sttc import value2sttc_short
from bkt_value2sttc import value2sttc_long
import graphviz 
from gplearn.functions import make_function
from gp_data_enlarge import data_process
import os
import pickle # 存数据
if not os.path.exists('./gp_rslt/'):
    os.mkdir('./gp_rslt/')
from bkt_sttc2plot import bkt_plot

IC0 = read_futdata('IC期货')
IC1 = read_futdata('IC期货(1)')

def score_func_basic(y, y_pred, sample_weight, **kwargs): # 适应度函数：策略评价指标
    global IC0
    data = IC0.copy()
    data['DateTime'] = data.index
    data.index = range(len(data))
    data['sgnl'] = pd.DataFrame(y_pred,columns=['sgnl'])#['sgnl']
    data_score = pos2value(data,data)
    sttc_short = value2sttc_short(data_score) # 返回收益率/最大回撤
    return sttc_short

m = fitness.make_fitness(function=score_func_basic,  # function(y, y_pred, sample_weight) that returns a floating point number. 
                         greater_is_better=True,  # 上述y是输入的目标y向量，y_pred是genetic program中的预测值，sample_weight是样本权重向量
                         wrap=False) # gplearn.fitness.make_fitness(function, greater_is_better, wrap=True)

# 报错
# def _asin(x1):
#     return np.where((x1<=1)&(x1>=-1), np.arcsin(x1), np.arcsin(1*np.sign(x1))) # np.where,条件，若真则，若假则
# def _acos(x1):
#     return np.where((x1<=1)&(x1>=-1), np.arccos(x1), np.arccos(1*np.sign(x1)))
# def _power(x1, x2):
#     res = np.power(abs(x1), abs(x2))
#     return np.where((res>0.0001)&(res<10000), res, 1)
# gp_asin = make_function(function=_asin, name='asin', arity=1)
# gp_acos = make_function(function=_acos, name='acos', arity=1)
# gp_power = make_function(function=_power, name='power', arity=2)
# 怀疑有未来信息
# def _stdd(x1):
#     return np.array([np.std(x1)] * len(x1))
# def _rankk(x1):
#     return x1.argsort()
# gp_stdd = make_function(function=_stdd, name='stdd', arity=1)
# gp_rankk = make_function(function=_rankk, name='rankk', arity=1)


def _andpn(x1, x2):
    return np.where((x1>0)&(x2>0), 1, -1)
def _orpn(x1, x2):
    return np.where((x1>0)|(x2>0), 1, -1)
def _ltpn(x1, x2):
    return np.where(x1<x2, 1, -1)
def _gtpn(x1, x2):
    return np.where(x1>x2, 1, -1)
def _andp(x1, x2):
    return np.where((x1>0)&(x2>0), 1, 0)
def _orp(x1, x2):
    return np.where((x1>0)|(x2>0), 1, 0)
def _ltp(x1, x2):
    return np.where(x1<x2, 1, 0)
def _gtp(x1, x2):
    return np.where(x1>x2, 1, 0)
def _andn(x1, x2):
    return np.where((x1>0)&(x2>0), -1, 0)
def _orn(x1, x2):
    return np.where((x1>0)|(x2>0), -1, 0)
def _ltn(x1, x2):
    return np.where(x1<x2, -1, 0)
def _gtn(x1, x2):
    return np.where(x1>x2, -1, 0)
def _if(x1, x2, x3):
    return np.where(x1>0, x2, x3)

def _delayy(x1): # 这个d咋搞
    return np.nan_to_num(np.concatenate([[np.nan],x1[:-1]]),nan = 0)
def _delta(x1):
    _ = np.nan_to_num(x1, nan=0)
    return _ - np.nan_to_num(_delayy(_), nan=0)
def _signedpower(x1):
    _ = np.nan_to_num(x1, nan=0)
    return np.sign(_) * (abs(_) ** 2)
def _decay_linear(x1):
    _ = pd.DataFrame({'x1':x1}).fillna(0) 
    __ = _.fillna(method = 'ffill').rolling(10).mean() - _
    return np.array(__['x1'].fillna(0))

gp_if = make_function(function=_if, name='if', arity=3)
gp_gtpn = make_function(function=_gtpn, name='gt', arity=2)
gp_andpn = make_function(function=_andpn, name='and', arity=2)
gp_orpn = make_function(function=_orpn, name='or', arity=2)
gp_ltpn = make_function(function=_ltpn, name='lt', arity=2)
gp_gtp = make_function(function=_gtp, name='gt', arity=2)
gp_andp = make_function(function=_andp, name='and', arity=2)
gp_orp = make_function(function=_orp, name='or', arity=2)
gp_ltp = make_function(function=_ltp, name='lt', arity=2)
gp_gtn = make_function(function=_gtn, name='gt', arity=2)
gp_andn = make_function(function=_andn, name='and', arity=2)
gp_orn = make_function(function=_orn, name='or', arity=2)
gp_ltn = make_function(function=_ltn, name='lt', arity=2)
gp_delayy = make_function(function=_delayy, name='delayy', arity=1)
gp_delta = make_function(function=_delta, name='_delta', arity=1)
gp_signedpower = make_function(function=_signedpower, name='_signedpower', arity=1)
gp_decayl = make_function(function=_decay_linear, name='_decayl', arity=1)

def gp_save_factor(cmodel_gp):
        with open(f'./gp_rslt//factor{factor_num}.pickle', 'wb') as f: # 存结果
            pickle.dump(cmodel_gp, f)
        with open(f'./gp_rslt//factor{factor_num}.pickle', 'rb') as f: # 读结果
            factor_rslt = pickle.load(f)
        print(factor_rslt)

def gp_save_plot(factor_num):
    factor_type = 'train'
    long_data = pd.read_csv(f'./gp_rslt//factor{factor_num}_{factor_type}_detail.csv',index_col=0)
    long_data.index = pd.to_datetime(long_data['DateTime'])
    bkt_plot(long_data,factor_type,factor_num)
    factor_type = 'test'
    long_data = pd.read_csv(f'./gp_rslt//factor{factor_num}_{factor_type}_detail.csv',index_col=0)
    long_data.index = pd.to_datetime(long_data['DateTime'])
    bkt_plot(long_data,factor_type,factor_num)
    
if __name__ == '__main__':
    start = datetime.now()
    run_num = 0
    f = open("./gp_rslt//factor_parameter.txt", 'w') # 记录因子结果的文件
    f.write(f"遗传算法生成因子参数记录文档\n")
    f.write(f"第几次运行\t因子编号\t总体数\t世代数\t每代选取优质个体数\n")
    f.write(f"run_num\tfactor_num\tpop_num\tgen_num\ttour_num\n")
    f.close()
    while True: # pop_num, gen_num,tour_num = 100, 3, 10
        run_num += 1
        print("--------------------------------------------------------------")
        print(" ")
        print(f"第{run_num}次循环\n")
        print(f"请输入参数。参数含义分别为:\n")
        print("pop_num(群体数，每代共有多少个公式作为父代),如100")
        print("gen_num(世代数，一共计算几代,如3")
        print("tour_num(进行交叉变异时从群体中选出多少(需要小于pop_num)),如10\n")
        pop_num = int(input("请输入群体数pop_num:"))
        gen_num = int(input("请输入世代数gen_num:"))
        tour_num = int(input("请输入选取群体数tour_num(需要小于pop_num):"))
        print('训练开始')
        train_data = IC0
        test_data = IC1

        cmodel_gp = SymbolicRegressor(population_size=pop_num, # 每一代公式群体中的公式数量 500，100
                                generations=gen_num, # 公式进化的世代数量 10，3
                                metric=m, # 适应度指标，这里是前述定义的通过 大于0做多，小于0做空的 累积净值/最大回撤 的评判函数
                                tournament_size=tour_num, # 在每一代公式中选中tournament的规模，对适应度最高的公式进行变异或繁殖 50
                                function_set= ( gp_delta, gp_signedpower, gp_decayl,
                                                gp_delayy, 
                                                # gp_stdd, 
                                                # gp_rankk, 
                                                # gp_corrr, 
                                                # gp_covv,
                                                'add', 'sub', 'mul', 'div', 'sqrt','log', 
                                                'abs', 'neg', 'inv', 
                                                'sin', 'cos', 'tan', 
                                                # gp_asin, gp_acos, gp_power, 
                                                # gp_andpn, gp_orpn, gp_ltpn, gp_gtpn, 
                                                # gp_andn, gp_orn, gp_ltn, gp_gtn, 
                                                gp_andp, gp_orp, gp_ltp, gp_gtp, 
                                                gp_if,
                                                'max', 'min',
                                                ), # 用于构建和进化公式使用的函数集
                                const_range=(-1.0, 1.0),  # 公式中包含的常数范围
                                parsimony_coefficient='auto',  # 对较大树的惩罚,默认0.001，auto则用c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between program size l and program fitness f in the population, and Var(l) is the variance of program sizes.
                                stopping_criteria=100.0, # 是对metric的限制（此处为收益/回撤）
                                init_depth=(2, 3), # 公式树的初始化深度，树深度最小2层，最大6层
                                init_method='half and half', # 树的形状，grow生分枝整的不对称，full长出浓密
                                p_crossover=0.8, # 交叉变异概率 0.8
                                p_subtree_mutation=0.05, # 子树变异概率
                                p_hoist_mutation=0.05, # hoist变异概率 0.15
                                p_point_mutation=0.05, # 点变异概率
                                p_point_replace=0.05, # 点变异中每个节点进行变异进化的概率
                                
                                max_samples=1.0, # The fraction of samples to draw from X to evaluate each program on.
                                
                                feature_names=None, warm_start=False, low_memory=False,
                                
                                n_jobs=1, 
                                verbose=1, 
                                random_state=0)
                                
        def gp_prdt_s(test_data,gp_data = cmodel_gp):
            """
                test_data: 用来测试的数据,如IC0,IC等
            """
            test_data = data_process(test_data)
            # gp_data.predict(test_data)
            data = test_data.copy()
            data['DateTime'] = data.index
            data.index = range(len(data))
            data['sgnl'] = pd.DataFrame(gp_data.predict(test_data),columns=['sgnl'])
            return value2sttc_short(pos2value(data,data)) # 返回收益率/最大回撤

        def gp_prdt_l(test_data,gp_data = cmodel_gp):
            """
                test_data: 用来测试的数据,如IC0,IC等
            """
            test_data = data_process(test_data)
            # gp_data.predict(test_data)
            data = test_data.copy()
            data['DateTime'] = data.index
            data.index = range(len(data))
            data['sgnl'] = pd.DataFrame(gp_data.predict(test_data),columns=['sgnl'])
            return value2sttc_long(pos2value(data,data)) # 返回收益率/最大回撤

        def gp_sttc(train_data,test_data):
            sttc_short0 = gp_prdt_s(train_data)
            sttc_short1 = gp_prdt_s(test_data)
            print("--------------------------------------------------------------")
            print(" ")
            print("训练集累计收益回撤比",round(sttc_short0,4))
            print("测试集累计收益回撤比",round(sttc_short1,4))
            print(" ")
            print("--------------------------------------------------------------")

        def gp_save_sttc(train_data,test_data,facotr_num):
            trade_detail0, sttc_long0 = gp_prdt_l(train_data)
            trade_detail1, sttc_long1 = gp_prdt_l(test_data)
            print(" ")
            print(" ")
            print("# --- 训练集表现为 --- #\n\n",sttc_long0)
            print(" ")
            print(" ")
            print("--------------------------------------------------------------")
            print(" ")
            print(" ")
            print("# --- 测试集表现为 --- #\n\n",sttc_long1)
            print(" ")
            print(" ")
            print("--------------------------------------------------------------")
            dot_data = cmodel_gp._program.export_graphviz() # 0.2 * 5
            graph = graphviz.Source(dot_data)
            graph.render(directory='gp_rslt').replace('\\', '/')
            os.rename("./gp_rslt//Source.gv",f"./gp_rslt//factor{facotr_num}.gv")
            os.rename("./gp_rslt//Source.gv.pdf",f"./gp_rslt//factor{facotr_num}.gv.pdf")
            trade_detail0.to_csv(f'./gp_rslt//factor{facotr_num}_train_detail.csv')
            trade_detail1.to_csv(f'./gp_rslt//factor{facotr_num}_test_detail.csv')
            sttc_long0['test'] = sttc_long1
            sttc_long0.columns = ['train','test']
            sttc_long0.to_csv(f'./gp_rslt//factor{facotr_num}_sttc.csv')
            print(f"因子{facotr_num}结果已保存")

        cmodel_gp.fit(data_process(train_data),train_data['future_open'].values)
        print(cmodel_gp)
        gp_sttc(train_data,test_data) # 1.3579 0.3646

        user_input = input("是(1)否(0)保存结果")
        if user_input == "1":
            print("--------------------------------------------------------------")
            print(" ")
            factor_num = int(input("请输入保存的因子编号factor_num(无限制,建议从1开始输入):"))
            gp_save_sttc(train_data,test_data,factor_num)
            print("统计量结果保存成功")
            gp_save_plot(factor_num)
            print("图表结果保存成功")
            gp_save_factor
            print("因子结果保存成功")
            print(f"因子{factor_num}结果已保存,pop_num, gen_num,tour_num={pop_num}, {gen_num},{tour_num}")
            print(" ")
            print("--------------------------------------------------------------")
            print(" ")
            f = open("./gp_rslt//factor_parameter.txt", 'w') # 记录因子结果的文件
            f.write(f"{run_num}\t{factor_num}\t{pop_num}\t{gen_num}\t{tour_num}\n")
            f.close()
            whether_continue = int(input("是(1)否(0)继续？"))
            if whether_continue == 0:
                end = datetime.now()
                elapsed = end - start
                print("Time elapsed:", elapsed) 
                break
        elif user_input == "0":
            print("--------------------------------------------------------------")
            print(" ")
            print("重新开始生成因子")
            print(" ")
            print("--------------------------------------------------------------")
        else:
            print("--------------------------------------------------------------")
            print(" ")
            print("输入无效")
            print(" ")
            print("--------------------------------------------------------------")
        # 这时候问是否保存结果，0不保存，重新输入参数，1保存

# 另一个思路：
# ralpha-plus run -f macd_000001.py -a stock 100000 -s 20190101 -e 20191231 -bm 000300.XSHG -p # 
# 类似于python xxx.py [option]，在一开始输入参数值，这部分可以在zsh里输入python --help
# 之后则进行如下循环进行自动化模拟
# for i in range(10):
#   os.system(f"python main.py {i}")