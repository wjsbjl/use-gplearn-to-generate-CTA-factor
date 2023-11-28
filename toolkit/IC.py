import pandas as pd
import numpy as np
from scipy import stats
from toolkit.my_plot import my_plot 

def get_ic(factor, ret5, ret10, method = 'spearman', print_type = 2, factor_name = 1):
    IC_df = pd.DataFrame()
    IC_df['5日IC'] = calculate_ic(ret5, factor, method)
    IC_df['10日IC'] = calculate_ic(ret10, factor, method)
    IC_statistic = ic_summary(IC_df)
    print(IC_statistic)
    my_plot(IC_df.cumsum(),['IC累计图', '时间', 'IC累计值', f'{factor_name}因子IC图']).line_plot()
    return IC_df, IC_statistic
   
def calculate_ic(ret, factor, method='spearman'):
    return ret.corrwith(factor, axis=1, method=method)
 
def ic_summary(IC_df):
    statistics = {
        "IC mean": round(IC_df.mean(), 4),
        "IC std": round(IC_df.std(), 4),
        "IR": round(IC_df.mean() / IC_df.std(), 4),
        "IR_LAST_1Y": round(IC_df[-240:].mean() / IC_df[-240:].std(), 4),
        "IC>0": round(len(IC_df[IC_df > 0].dropna()) / len(IC_df), 4),
        "ABS_IC>2%": round(len(IC_df[abs(IC_df) > 0.02].dropna()) / len(IC_df), 4)
    }
    return pd.DataFrame(statistics)

if __name__ == '__main__':
    vwap = pd.read_csv('../data/daily_5/vwap.csv',index_col=[0],parse_dates=[0])
    price = vwap
    ret5 = np.log(price.shift(-5) / price.shift(-1))
    ret10 = np.log(price.shift(-10) / price.shift(-1))
    IC_df, IC_statistic = get_ic(vwap) # 自相关系数
