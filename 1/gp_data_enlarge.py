# 这里的目标是使输入的数据更大，方式是新增一列难以在func set加入的函数（也就是通过自定义func的方式扩充data set）
# 目前加入bar_num数据，即当前是开盘后第几分钟
def data_process(data_input):
    test_data = data_input.copy()
    test_data['time'] = [x.time() for x in test_data.index]
    keys = sorted(set(test_data['time']))
    values = range(1,len(set(test_data['time']))+1)
    time2rank = {}
    for i, key in enumerate(keys):
        time2rank[key] = values[i]
    test_data['rank_num'] = test_data['time'].map(time2rank)
    test_data.drop('time',axis=1,inplace = True)
    return test_data

# TODO 备份一些函数集和数据集
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