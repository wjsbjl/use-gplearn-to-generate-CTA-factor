import dill
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import pickle
from utils.myDataProcess import read_parquet_to_dict
import statsmodels.api as sm
from scipy.stats import rankdata
from scipy.stats import linregress
from scipy.stats import spearmanr, pearsonr
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt  # 是matplotlib的子包
import os
if os.name == 'posix':  # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC']  # 中文字体为宋体
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 在windows系统下显示微软雅黑
mpl.rcParams['figure.dpi'] = 100  # 这个对图形的设置不错，统一了
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False  # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
import os
if not os.path.exists('./result'):
    os.makedirs('./result')

import pandas as pd
import numpy as np
from scipy import stats
from utils.my_plot import my_plot 


class FactorTest():
    def __init__(self, price, log_market = None, industry = None, clip_range = 0.08):
        self.price = price
        self.log_market = log_market
        self.industry = industry
        self.clip_range = clip_range
        self.ret5 = (price.shift(-5) / price.shift(-1) - 1).clip(-clip_range, clip_range) # 6611天，3553只股
        self.ret10 = (price.shift(-10) / price.shift(-1) - 1).clip(-clip_range, clip_range) # 6611天，3553只股

    # ic
    def get_ic(self, factor, method = 'pearson', print_type = 2, factor_name = 1, plt_show = True):
        IC_df = pd.DataFrame()
        IC_df['5日IC'], _ = self.calculate_ic(factor = factor, ret = self.ret5, method = method)
        IC_df['10日IC'], _ = self.calculate_ic(factor = factor, ret = self.ret10, method = method)
        IC_statistic = self.ic_summary(IC_df)
        print(IC_statistic)
        my_plot(IC_df.cumsum(),['IC累计图', '时间', 'IC累计值', f'{factor_name}因子IC图']).line_plot(plt_show = plt_show)
        return IC_df, IC_statistic
    
    def calculate_ic(self, factor, ret=None, method='pearson'):
        if ret is None:
            ret = self.ret5
        corr_df = ret.corrwith(factor, axis=1, method=method)
        return corr_df, abs(corr_df.mean())
    
    def ic_summary(self, IC_df):
        statistics = {
            "IC mean": round(IC_df.mean(), 4),
            "IC std": round(IC_df.std(), 4),
            "IR": round(IC_df.mean() / IC_df.std(), 4),
            "IR_LAST_1Y": round(IC_df[-240:].mean() / IC_df[-240:].std(), 4),
            "IC>0": round(len(IC_df[IC_df > 0].dropna()) / len(IC_df), 4),
            "ABS_IC>2%": round(len(IC_df[abs(IC_df) > 0.02].dropna()) / len(IC_df), 4)
        }
        return pd.DataFrame(statistics)

    # neutralize
    def neutralize(self, factor, index_range=None):
        neutral_factor_df = self.industry.copy()
        resid_df = pd.DataFrame(index = factor.index, columns = self.ret5.columns)

        for idx in tqdm(resid_df.index):
            neutral_factor_df.loc[:,'log_market'] = self.log_market.loc[idx,:] # MARK 可以修改扩展列
            neutral_factor_df.loc[:,'factor'] = factor.loc[idx,:]
            reg_df = neutral_factor_df.dropna()
            
            # 求残差法1，大概是法2一半时间
            X = reg_df.iloc[:, :-1]
            Y = reg_df.iloc[:, -1]
            X = np.column_stack((np.ones(len(X)), X))
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid_df.loc[idx, :] = Y - np.dot(X, beta)

            # 求残差法2
            # resid_df1.loc[idx,:] = sm.OLS(reg_df.iloc[:,-1], sm.add_constant(reg_df.iloc[:,:-1])).fit().resid
        return resid_df

    def get_top_bottom_ret(self, factor, shift_period = 5, group = 20):
        ret_df = self.ret5.loc[factor.index, :].dropna(how='all')
        factor = factor + np.random.normal(loc=0, scale=1e-14, size=factor.shape) # 有时候值相近，不容易分组
        factor_group_df = factor.apply(lambda x: pd.qcut(x, q=group, labels=False, duplicates='drop'), axis = 1)
        group_ret_df = pd.DataFrame()
        group_ret_df['Top - Market'] = ret_df[factor_group_df == group - 1].mean(axis = 1) - ret_df.mean(axis = 1)
        group_ret_df['Bottom - Market'] = ret_df[factor_group_df == 0].mean(axis = 1) - ret_df.mean(axis = 1)
        group_ret_df.fillna(0,inplace=True)
        # pnl_df = (1 + group_ret_df).cumprod()
        return group_ret_df

    def get_group_ret(self, factor, shift_period = 5, group = 20):
        ret_df = self.ret5.loc[factor.index, :].dropna(how='all')

        factor = factor + np.random.normal(loc=0, scale=1e-14, size=factor.shape) # 有时候值相近，不容易分组
        factor_group_df = factor.apply(lambda x: pd.qcut(x, q=group, labels=False, duplicates='drop'), axis = 1)
        group_ret_df = pd.DataFrame()
        for i in range(20):
            # group_ret_df[f'group{i}'] = (ret_df * (factor_group_df == i)).mean(axis = 1) # MARK 错误方法       
            group_ret_df[f'group{i}'] = ret_df[factor_group_df == i].mean(axis = 1)        
        group_ret_df['Top - Bottom'] = group_ret_df.iloc[:,group - 1] - group_ret_df.iloc[:,0]
        group_ret_df['Top - Market'] = group_ret_df.iloc[:,group - 1] - group_ret_df.mean(axis = 1)
        group_ret_df['Bottom - Market'] = group_ret_df.iloc[:,0] - group_ret_df.mean(axis = 1)
        # pnl_df = (1 + group_ret_df).cumprod()
        return group_ret_df
    
    def plot_key_period(self, pnl_df):
        my_plot((pnl_df.loc['2020-11':'2021-03',:] + 1).cumprod(), ['202011-202103', 'Xlabel', 'Ylabel', 'save_name']).line_plot()
        my_plot((pnl_df.loc['2021-10':'2021-12',:] + 1).cumprod(), ['202011-202103', 'Xlabel', 'Ylabel', 'save_name']).line_plot()
        my_plot((pnl_df.loc['2023-03':'2023-05',:] + 1).cumprod(), ['202303-202305', 'Xlabel', 'Ylabel', 'save_name']).line_plot()      

def gp_factor_test(FT, factor_name = 2, x_dict = None, show_resid = False, ipython = True, print_feature_name = False, use_resid = False, generative_method = 'gplearn', plt_show = True):    
    if generative_method == 'gplearn':    
        if not os.path.exists(f'./result/factor/factor{factor_name}.parquet'):
            with open(f'./result/factor/factor{factor_name}.pickle', 'rb') as f:  # 读结果
                my_cmodel_gp = dill.load(f)
            feature_names = my_cmodel_gp.feature_names
            if print_feature_name:
                print(feature_names)
            x_dict_factor = {k: x_dict[k] for k in feature_names}
            x_array = np.transpose(np.array(list(x_dict_factor.values())), axes=(1, 2, 0))
            factor_df = pd.DataFrame(my_cmodel_gp.predict(x_array),index = price.index, columns = price.columns)
            factor_df.to_parquet(f'./result/factor/factor{factor_name}.parquet')
        else:
            factor_df = pd.read_parquet(f'./result/factor/factor{factor_name}.parquet')
    elif generative_method == 'csv':
        factor_df = pd.read_csv(f'./{factor_name}.csv', index_col=0, parse_dates=[0])
    else:
        factor_df = pd.read_parquet(f'./{factor_name}.parquet').dropna(how = 'all')

    FT.get_ic(factor_df.iloc[-200:,:])
    
    if use_resid:
        resid_df = FT.neutralize(factor_df.iloc[:,:])
        factor_df = resid_df
        corr_df, ic = FT.calculate_ic(resid_df.iloc[-200:,:].astype(float)) # 直接取因子值计算IC
        print(ic)
    elif show_resid:
        resid_df = FT.neutralize(factor_df.iloc[-200:,:]) # 中性化：y_pred与其他因子做回归，取残差
        corr_df, ic = FT.calculate_ic(resid_df.astype(float)) # 直接取因子值计算IC
        print(ic)

    # 回测
    group_ret_df = FT.get_group_ret(factor_df.iloc[::5,:])
    my_plot((group_ret_df.iloc[:,-3:]+1).cumprod(),[f'全时段内收益 因子{factor_name}', 'X', 'Y', f'全时段内收益_因子{factor_name}']).line_plot(plt_show = plt_show)
    my_plot((group_ret_df.iloc[-40:,-3:]+1).cumprod(),[f'测试集内收益 {factor_name}', 'X', 'Y', f'测试集内收益_因子{factor_name}']).line_plot(plt_show = plt_show)
    my_plot(group_ret_df.iloc[-40:, :-3].mean(), [f'20分组收益 因子{factor_name}', 'X', 'Y', f'20分组收益_因子{factor_name}']).bar_plot(rotation_angel = 30, plt_show = plt_show)
    FT.plot_key_period(group_ret_df.iloc[:,-3:]) # 三个时点
    if generative_method == 'gplearn':
        if ipython:
            from IPython.display import Image
            return Image(filename=f'./result/factor/factor{factor_name}.jpg', width=600)
        else:
            from PIL import Image
            return Image.open(f'./result/factor/factor{factor_name}.jpg').show()
                
if __name__ == '__main__':
    vwap = pd.read_parquet('../data/daily/vwap.parquet')
    adj_factor = pd.read_parquet('../data/daily/adj_factor.parquet')
    industry = pd.read_parquet('../data/中性化因子/industry_dummy.parquet').sort_index()
    log_market = pd.read_parquet('../data/中性化因子/log_market.parquet').sort_index()    
    price = vwap * adj_factor
    FT = FactorTest(price, log_market, industry)
    gp_factor_test(FT, 'specific_volatility_df', generative_method='parquet')
    x_dict = read_parquet_to_dict('../data/daily/')
    gp_factor_test(20231207, show_resid=True, print_feature_name=True, use_resid=True)
    # gp_factor_test(20231207, show_resid=True, print_feature_name=True, use_resid=True)
    