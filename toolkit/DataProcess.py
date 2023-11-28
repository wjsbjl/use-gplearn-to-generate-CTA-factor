import numpy as np
import pandas as pd
import dill
def load_timing_data():
    train_data = pd.read_csv('./data/IC_train.csv', index_col=0)
    test_data = pd.read_csv('./data/IC_test.csv', index_col=0)
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    feature_names = list(train_data.columns)
    train_data.loc[:,'y'] = np.log(train_data['Open'].shift(-4)/train_data['Open'].shift(-1))
    train_data.dropna(inplace = True)
    return train_data, test_data, feature_names

def load_selecting_data():
    vwap = pd.read_csv('../data/daily_5/vwap.csv',index_col=[0],parse_dates=[0]) # 2290天，5434只股
    close = pd.read_csv('../data/daily_5/close.csv',index_col=[0],parse_dates=[0])
    buy_volume_exlarge_order = pd.read_csv('../data/daily_5/buy_volume_exlarge_order.csv',index_col=[0],parse_dates=[0])
    y_train1 = np.log(vwap.shift(-5) / vwap.shift(-1))
    y_train2 = np.log(vwap.shift(-10) / vwap.shift(-1))
    x_dict = {'close':close,
               'buy_volume_exlarge_order':buy_volume_exlarge_order}
    x_array = np.array(list(x_dict.values()))
    x_array = np.transpose(x_array, axes=(1, 2, 0))
    feature_names = x_dict.keys()    
    return x_array, feature_names

def process_factor_data():
    with open(f'./data/stock_data.pickle', 'rb') as f:
        price = dill.load(f)
    # y_ret = pd.read_csv('./data/stock_data/fadj_close.csv',index_col=[0],parse_dates=[0])
    price = price.loc['2015':,:]
    price.dropna(how='all', axis=1, inplace=True)
    # y_ret.to_csv('./data/stock_data/fadj_close.csv')
    # with open(f'./data/stock_data.pickle', 'wb') as f:
    #     dill.dump(y_ret, f)
    with open(f'./data/stock_data.pickle', 'rb') as f:  # 读结果
        price = dill.load(f)
    # x_dict = {}
    # for filename in tqdm(os.listdir('./data/factor_data/')):
    #     if filename.endswith('.csv'):
    #         file_key = os.path.splitext(filename)[0]
    #         df = pd.read_csv(os.path.join('./data/factor_data/', filename),index_col=[0],parse_dates=[0])
    #         df = df.loc[price.index, price.columns]
    #         x_dict[file_key] = df
    # with open(f'./data/factor_data.pickle', 'wb') as f:
    #     dill.dump(x_dict, f)
    with open(f'./data/factor_data.pickle', 'rb') as f:  # 读结果
        x_dict = dill.load(f)
    # filtered_dict = {key: value for key, value in x_dict.items() if key in ('turnover_21', 'amount_21')}
    for key in list(x_dict.keys())[:-1]:
        x_dict[key] = x_dict[key].loc[price.index, price.columns]
    with open(f'./data/factor_data.pickle', 'wb') as f:
        dill.dump(x_dict, f)   
        
    new_dict = {}
    for key in list(x_dict.keys())[:-1]:
        new_dict[key] = x_dict[key].loc[price.index, price.columns]
    with open(f'./data/what.pickle', 'wb') as f:
        dill.dump(new_dict, f)   
    return x_dict

def delete_factor_data(path):
    data = pd.read_csv(path,index_col=[0],parse_dates=[0])
    data = data.loc['2010':,:]
    data.to_csv(path)    
    
# def read_futdata(file_name):
#     data_temp = pd.read_csv(f"./data/{file_name}.csv",index_col=0,low_memory=False)
#     data_temp.index = pd.to_datetime(data_temp.index)
#     data_temp.fillna(method = 'ffill',inplace = True)
#     data_temp = data_temp[data_temp.index.isnull() == False]
#     return data_temp

# def data_enlarge(data_input): # 这里的目标是使输入的数据更大，方式是新增一列难以在func set加入的函数（也就是通过自定义func的方式扩充data set）
#     test_data = data_input.copy()
#     test_data['time'] = [x.time() for x in test_data.index]
#     keys = sorted(set(test_data['time']))
#     values = range(1,len(set(test_data['time']))+1)
#     time2rank = {}
#     for i, key in enumerate(keys):
#         time2rank[key] = values[i]
#     test_data['rank_num'] = test_data['time'].map(time2rank) # 目前加入bar_num数据，即当前是开盘后第几分钟
#     test_data.drop('time',axis=1,inplace = True)
#     return test_data

# pd.read_csv('./data/factor_data/turnover_21.csv')
# pd.read_csv('./data/factor_data 2/turnover_21.csv')
# pd.read_csv('./data/stock_data/fadj_close.csv',index_col=[0],parse_dates=[0])

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    # process_factor_data()
    # delete_factor_data('./data/stock_data/fadj_close.csv')
    # for filename in tqdm(os.listdir('./data/factor_data')):
    #     if filename.endswith('.csv'):
    #         delete_factor_data(f'./data/factor_data/{filename}')
    # process_factor_data()
    6