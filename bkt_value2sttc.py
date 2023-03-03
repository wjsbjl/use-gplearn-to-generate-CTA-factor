# 此处定义了根据账户价值计算统计信息的函数。
# 统计结果：
# short：计算投资收益率与最大回撤比值。若最大回撤（分母）为0则返回10。
# long：分别输出包含逐笔交易的交易过程和基本统计信息，其中统计信息包括
# 逐笔收益，操作胜率，收益率(年化),标准差(年化),夏普比率(年化),最大回撤水平,最大回撤比率,平均每日开仓次数,胜率
import numpy as np
import pandas as pd

def value2sttc_short(data_score):
    cash_value = data_score[['value_with_comm']]
    rate_of_return = (cash_value.iloc[-1, 0] / cash_value.iloc[0, 0] - 1)
    max_drawdown = (cash_value.cummax() - cash_value).max()[0]
    # sum(data_score['pos'] != data_score['pos'].shift(1)) # 应该是调仓次数
    return rate_of_return / max_drawdown if max_drawdown else 0


def value2sttc_long(data_score):
    # 每笔收益: 开多仓的费率是0.23*2/10000; 开空仓的费率是(3.45 + 0.23)/10000
    data_per_trade = data_score[['pos', 'open']].copy()
    # data_per_trade = data_score[['DateTime','pos','value','open']].copy()
    value_per_trade = data_per_trade.loc[(data_per_trade['pos'] != data_per_trade['pos'].shift(1)), :]
    value_per_trade = value_per_trade.iloc[1:, :]  # 开盘和na不同，但这时候并不记为调仓
    Ilong = (value_per_trade['pos'] == 1)  # pos为1则计算rlong，pos为-1则计算rshort
    Ishort = (value_per_trade['pos'] == -1)
    V1 = value_per_trade['open'].shift(-1)
    V0 = value_per_trade['open']
    _ = ((1 - 0.23 / 10000) * (Ilong * V1 + Ishort * V0) - (1 + 0.23 / 10000) * Ilong * V0 - (
                1 + 3.45 / 10000) * Ishort * V1) / (V0 * (1 + 0.23 / 10000))
    data_score.loc[:, 'return_per_trade'] = _
    data_score.loc[(data_score['pos'] == 0), 'return_per_trade'] = np.nan  # 费率-仅在非0部分标记

    # adjustcount = len(value_per_trade) - 1 #因为加了一列
    daycount = len(set(x.date() for x in data_score['DateTime']))
    cash_value = data_score[['value_with_comm']]
    cash_value.index = data_score['DateTime']
    cashflow_temp = cash_value.copy()
    cashflow_temp = cashflow_temp.resample('d').last().dropna()  # 转为日数据
    cashflow_temp['simple_return'] = cashflow_temp / cashflow_temp.shift(1) - 1
    cash_ref = data_score['open'].values
    return_ref = (cash_ref[-1] / cash_ref[0] - 1) * 365 / (cash_value.index[-1] - cash_value.index[0]).days
    return_annual = (cashflow_temp['value_with_comm'][-1] / cashflow_temp['value_with_comm'][0] - 1) * 365 / (
                cash_value.index[-1] - cash_value.index[0]).days
    std_annual = cashflow_temp['simple_return'].std() * np.sqrt(252)
    sharpe_annual = (return_annual - 0.02) / std_annual

    sttc_per_trade = data_score[['return_per_trade', 'pos']].copy()  # 逐笔收益 * 仓位
    sttc_per_trade.dropna(inplace=True)
    adjustcount = len(sttc_per_trade)
    # (sttc_per_trade.loc[:,'return_per_trade'] > 0).sum() / (sttc_per_trade['return_per_trade']).count()

    sttc_rslt = pd.DataFrame(data=[return_annual, return_ref, std_annual, sharpe_annual,
                                   np.max(pd.DataFrame(cash_value).cummax() - pd.DataFrame(cash_value), axis=0)[0],
                                   np.max((pd.DataFrame(cash_value).cummax() - pd.DataFrame(cash_value)) /
                                          pd.DataFrame(cash_value).cummax(), axis=0)[0],
                                   adjustcount / daycount,
                                   (data_score.loc[:, 'return_per_trade'] > 0).sum() / adjustcount,
                                   (sttc_per_trade['pos'] == 1).sum() / daycount,
                                   (sttc_per_trade.loc[(sttc_per_trade['pos'] == 1), 'return_per_trade'] > 0).sum() / (
                                               sttc_per_trade['pos'] == 1).sum() if (
                                               sttc_per_trade['pos'] == 1).sum() else np.inf,
                                   (sttc_per_trade['pos'] == -1).sum() / daycount,
                                   (sttc_per_trade.loc[(sttc_per_trade['pos'] == -1), 'return_per_trade'] > 0).sum() / (
                                               sttc_per_trade['pos'] == -1).sum() if (
                                               sttc_per_trade['pos'] == -1).sum() else np.inf,
                                   ], dtype=float,
                             index=['收益率(年化)', '收益率(Buy and Hold, 参考)', '标准差(年化)', '夏普比率(年化)',
                                    '最大回撤水平', '最大回撤比率', '平均每日开仓次数', '胜率',
                                    '平均每日开多仓次数', '多仓胜率', '平均每日开空仓次数', '空仓胜率'])
    sttc_rslt = sttc_rslt.round(4)
    sttc_rslt.loc[(sttc_rslt.index.str.contains("率")) & (sttc_rslt[0] != np.inf)] = sttc_rslt.loc[
        (sttc_rslt.index.str.contains("率")) & (sttc_rslt[0] != np.inf)].applymap(lambda x: "{:.2%}".format(x))

    return data_score, sttc_rslt
