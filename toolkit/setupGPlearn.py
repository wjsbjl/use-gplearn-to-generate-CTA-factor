from functools import partial
from gplearn.genetic import SymbolicRegressor
from gplearn import fitness
import os
import graphviz
import dill
from pdf2image import convert_from_path

def gp_save_factor(my_cmodel_gp, factor_num=''):
    with open(f'./result/factor/factor{factor_num}.pickle', 'wb') as f:
        dill.dump(my_cmodel_gp, f)
    with open(f'./result/factor/factor{factor_num}.pickle', 'rb') as f:  # 读结果
        factor_rslt = dill.load(f)
    print(factor_rslt)
    dot_data = my_cmodel_gp._program.export_graphviz()  # 0.2 * 5
    graph = graphviz.Source(dot_data)
    graph.render(directory='result').replace('\\', '/')
    os.rename("./result/Source.gv", f"./result/factor/factor{factor_num}.gv")
    os.rename("./result/Source.gv.pdf", f"./result/factor/factor{factor_num}.pdf")
    images = convert_from_path(f"./result//factor/factor{factor_num}.pdf")
    images[0].save(f'./result/factor/factor{factor_num}.jpg', 'JPEG')

def score_func_supplement(y, y_pred, sample_weight):  # 加这行是因为fitness有对于参数个数的逻辑判断
    return y

def my_gplearn(function_set, score_func_basic, pop_num=100, gen_num=3, tour_num=10, random_state = 42, feature_names=None):
    # pop_num, gen_num, tour_num的几个可选值：500, 5, 50; 1000, 3, 20; 1000, 15, 100
    metric = fitness.make_fitness(function=score_func_basic, # function(y, y_pred, sample_weight) that returns a floating point number.
                        greater_is_better=True,  # 上述y是输入的目标y向量，y_pred是genetic program中的预测值，sample_weight是样本权重向量
                        wrap=False)  # 不保存，运行的更快 # gplearn.fitness.make_fitness(function, greater_is_better, wrap=True)
    return SymbolicRegressor(population_size=pop_num,  # 每一代公式群体中的公式数量 500，100
                              generations=gen_num,  # 公式进化的世代数量 10，3
                              metric=metric,  # 适应度指标，这里是前述定义的通过 大于0做多，小于0做空的 累积净值/最大回撤 的评判函数
                              tournament_size=tour_num,  # 在每一代公式中选中tournament的规模，对适应度最高的公式进行变异或繁殖 50
                              function_set=function_set, 
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
                              feature_names=feature_names, warm_start=False, low_memory=False,
                              n_jobs=1,
                              verbose=1,
                              random_state=random_state)
