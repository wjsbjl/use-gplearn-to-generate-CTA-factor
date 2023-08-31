# 此文档将简要说明gplearn的使用方法
import numpy as np
import pandas as pd
from gplearn import fitness
from gplearn.genetic import SymbolicRegressor
from datetime import datetime

def score_func_basic(y, y_pred, sample_weight, **args): # 适应度函数：策略评价指标
    return sum((pd.Series(y_pred) - y) ** 2 ) # 这里是最小化残差平方和

m = fitness.make_fitness(function=score_func_basic,  # function(y, y_pred, sample_weight) that returns a floating point number. 
                         greater_is_better=False,  # 上述y是输入的目标y向量，y_pred是genetic program中的预测值，sample_weight是样本权重向量
                         wrap=False) # gplearn.fitness.make_fitness(function, greater_is_better, wrap=True)

cmodel_gp = SymbolicRegressor(population_size=500, # 每一代公式群体中的公式数量 500
                              generations=10, # 公式进化的世代数量 10
                              metric=m, # 适应度指标，这里是前述定义的通过 大于0做多，小于0做空的 累积净值/最大回撤 的评判函数
                              tournament_size=50, # 在每一代公式中选中tournament的规模，对适应度最高的公式进行变异或繁殖 50
                              function_set= ('add', 'sub', 'mul','abs', 'neg', 'sin', 'cos', 'tan'), # 用于构建和进化公式使用的函数集
                              const_range=(-1.0, 1.0),  # 公式中包含的常数范围
                              parsimony_coefficient='auto',  # 对较大树的惩罚,默认0.001，auto则用c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between program size l and program fitness f in the population, and Var(l) is the variance of program sizes.
                              # stopping_criteria=100.0, # 是对metric的限制（此处为收益/回撤）
                              init_depth=(2, 4), # 公式树的初始化深度，树深度最小2层，最大6层
                              init_method='half and half', # 树的形状，grow生分枝整的不对称，full长出浓密
                              p_crossover=0.2, # 交叉变异概率 0.8
                              p_subtree_mutation=0.2, # 子树变异概率
                              p_hoist_mutation=0.2, # hoist变异概率 0.15
                              p_point_mutation=0.2, # 点变异概率
                              p_point_replace=0.2, # 点变异中每个节点进行变异进化的概率
                              max_samples=1.0, # The fraction of samples to draw from X to evaluate each program on.
                              feature_names=None, warm_start=False, low_memory=False,
                              n_jobs=1, 
                              verbose=1, 
                              random_state=0
                             )

if __name__ == '__main__':
    start = datetime.now()
    LenD = 1000
    X1 = pd.DataFrame(data = {'a' : range(LenD), 'b' : np.random.randint(-10,10,LenD)})
    Y1 = X1.sum(axis = 1)#.values
    print("初始策略是Y1=X1.sum(axis=1)")
    cmodel_gp.fit(X1,Y1)
    print(cmodel_gp)
    print("------------------------------------------------------------------------------------")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print("------------------------------------------------------------------------------------")

    LenD = 1000
    X2 = pd.DataFrame(data = {'a' : range(LenD), 'b' : np.random.randint(0,10,LenD)})
    Y2 = np.cos(X2['a']) - np.sin(X2['b'])
    cmodel_gp.fit(X2,Y2)
    print(cmodel_gp)
    print("------------------------------------------------------------------------------------")
    end = datetime.now()
    elapsed = end - start
    print("Time elapsed:", elapsed) 
