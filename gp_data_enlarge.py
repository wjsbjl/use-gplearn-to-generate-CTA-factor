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
