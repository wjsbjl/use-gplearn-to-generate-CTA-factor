import pandas as pd
import numpy as np
from abc import abstractmethod, ABCMeta
from collections import abc
import pickle
from itertools import zip_longest
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import os
import warnings
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
if not os.path.exists('./result/backtest/'):   #os：operating system，包含操作系统功能，可以进行文件操作
    os.makedirs('./result/backtest/') #如果存在那就是这个result_path，如果不存在那就新建一个
if os.name == 'posix': # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC'] #中文字体为宋体,还可以选['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 在windows系统下显示微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212
import matplotlib.gridspec as gridspec
import tqdm
import itertools
import multiprocessing
import dill
from functools import wraps

class SiMuPaiPaiWang():
    """
    排排网的配色
    """
    colors = {'strategy': '#de3633',
              'benchmark': '#80b3f6',
              'excess': '#f4b63f'}

    def __getitem__(self, key):
        return self.colors[key]

    def __repr__(self):
        return self.colors.__repr__()

# TODO: 增加stop_loss


class DataSet(object):
    def __init__(self):
        pass


class Displayer(object):
    """
    Display a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        assert isinstance(df.index, pd.DatetimeIndex)
        assert 'benchmark_curve' in df.columns
        assert 'strategy_curve' in df.columns
        assert 'position' in df.columns
        assert 'signal' in df.columns
        self.df = df
        self.freq_base = self.df['benchmark'].resample('d').apply(lambda x: len(x))[0]
        self.holding_infos = self._calc_holding_infos()
        self.out_stats = self._calc_stats()

    def get_results(self):
        return self.out_stats, self.holding_infos, self.df

    def save_results(self, file_name=''):        
        self.out_stats.to_csv(f'./result/backtest/{file_name}_stats.csv')
        self.holding_infos.to_csv(f'./result/backtest/{file_name}_holdings.csv')
        self.df.to_csv(f'./result/backtest/{file_name}_trading_details.csv')
        # pd.concat([self.df,data['FutPredict']], axis = 1).to_csv(f'./backtest/{model_name}{comm}_df.csv')

    def _calc_holding_infos(self): # TODO
        state = self.df['position'].copy(deep=True)
        assert not all(state == 0), "没有进行交易, 故不进行统计"

        time_info = [num for num, count in enumerate(np.abs((state - state.shift(1)).fillna(0)))
                     for i in range(int(count))]
        open_time = time_info[::2] # 切片得用熟
        exit_time = time_info[1::2]
        holding_infos = pd.DataFrame(zip_longest(open_time, exit_time, fillvalue=None))
        holding_infos.columns = ['open_time', 'exit_time']
        holding_infos.fillna(len(self.df), inplace=True)
        holding_infos['direction'] = state[list(holding_infos['open_time'])].values
        holding_infos['holding_time'] = holding_infos['exit_time'] - holding_infos['open_time']
        holding_infos['returns'] = holding_infos.apply(lambda x:
                                                       np.log(self.df['strategy_curve'].iloc[int(x['exit_time']) - 1] /
                                                              self.df['strategy_curve'].iloc[int(x['open_time']) - 1]),
                                                       axis=1)
        holding_infos.loc[:,'open_time_stamp'] = self.df.index[holding_infos.loc[:,'open_time'].values-1]
        holding_infos.loc[:,'exit_time_stamp'] = self.df.index[holding_infos.loc[:,'exit_time'].values.astype(int)-1]
        return holding_infos

    def _calc_stats(self): # TODO
        output_stat = {}
        strategy_returns = np.log(self.df['strategy_curve'] / self.df['strategy_curve'].shift(1))
        benchmark_returns = np.log(self.df['benchmark_curve'] / self.df['benchmark_curve'].shift(1))
        excess_returns = strategy_returns - benchmark_returns
        output_stat['Annualized_Mean'] = 252 * strategy_returns.groupby(strategy_returns.index.date).sum().mean() # 252 * self.freq_base * np.mean(strategy_returns)
        output_stat['Annualized_Std'] = np.sqrt(252) * strategy_returns.groupby(strategy_returns.index.date).sum().std() # np.sqrt(252 * self.freq_base) * np.std(strategy_returns)
        output_stat['Sharpe'] = output_stat['Annualized_Mean'] / output_stat['Annualized_Std']
        output_stat['Excess_Annualized_Mean'] = 252 * self.freq_base * np.mean(excess_returns)
        output_stat['Excess_Annualized_Std'] = np.sqrt(252 * self.freq_base) * np.std(excess_returns)
        output_stat['Excess_sharpe'] = output_stat['Excess_Annualized_Mean'] / output_stat['Excess_Annualized_Std']
        output_stat['MaxDrawDown'] = ((self.df['strategy_curve'].cummax() - self.df['strategy_curve']) / self.df[
            'strategy_curve'].cummax()).max()
        try:
            output_stat['LongCounts'] = self.holding_infos['direction'].value_counts()[1]
            output_stat['MeanLongTime'] = \
            self.holding_infos['holding_time'].groupby(self.holding_infos['direction']).mean()[1]
            output_stat['PerLongReturn'] = \
            self.holding_infos['returns'].groupby(self.holding_infos['direction']).mean()[1]
            
        except KeyError:
            output_stat['LongCounts'] = 0
            output_stat['MeanLongTime'] = 0
            output_stat['PerLongReturn'] = 0

        try:
            output_stat['ShortCounts'] = self.holding_infos['direction'].value_counts()[-1]
            output_stat['MeanShortTime'] = \
            self.holding_infos['holding_time'].groupby(self.holding_infos['direction']).mean()[-1]
            output_stat['PerShortReturn'] = \
            self.holding_infos['returns'].groupby(self.holding_infos['direction']).mean()[-1]
        except KeyError:
            output_stat['ShortCounts'] = 0
            output_stat['MeanShortTime'] = 0
            output_stat['PerShortReturn'] = 0
        try: 
            temp_p = self.holding_infos['returns'][self.holding_infos['returns'] > 0].mean()
            temp_n = self.holding_infos['returns'][self.holding_infos['returns'] < 0].mean()
            output_stat['PnL'] = np.abs(temp_p / temp_n)

        except ZeroDivisionError:
            output_stat['PnL'] = np.inf

        output_stat['WinRate'] = (self.holding_infos['returns'] > 0).sum() / len(self.holding_infos['returns'])

        return pd.Series(output_stat)

    def plot_(self, comm='', tick_count=9, plot_name='', rotation_angle=0, plot_PnL=True, show_bool=False): # 12
        datetime_index = self.df.index
        if plot_PnL:
            strategy_returns = self.df['strategy_curve']#.pct_change() # 策略收益
            benchmark_returns = self.df['benchmark_curve']#.pct_change()
            excess_returns = (1+strategy_returns.pct_change() - benchmark_returns.pct_change()).cumprod()
            y_hlines = 1
        else:
            strategy_returns = self.df['strategy_curve'].pct_change() # 策略收益
            benchmark_returns = self.df['benchmark_curve'].pct_change()
            excess_returns = strategy_returns - benchmark_returns
            strategy_returns = strategy_returns.cumsum().fillna(0)
            benchmark_returns = benchmark_returns.cumsum().fillna(0)
            excess_returns = excess_returns.cumsum().fillna(0)
            y_hlines = 0
        fig = plt.figure(figsize=(10, 6))  # 创建画布
        gs = gridspec.GridSpec(12, 1)  # 定义网格，4行1列
        ax1 = fig.add_subplot(gs[:8, :])  # 第一张图占据前3行
        ax2 = fig.add_subplot(gs[8:10, :])  # 第二张图占据最后一行
        ax3 = fig.add_subplot(gs[10:, :])  # 第二张图占据最后一行
        fontsize = 12
        plot_df1 = pd.concat([strategy_returns, benchmark_returns, excess_returns], axis = 1)
        plot_df1.columns = ['strategy', 'benchmark', 'excess']
        label_list = ['策略', '基准', '超额']
        x = range(len(self.df))
        y_labels = plot_df1.columns
        for i in range(len(y_labels)):
            clmn = y_labels[i]
            ax1.plot(x, plot_df1.loc[:, clmn].values, label=label_list[i], color=SiMuPaiPaiWang()[clmn])  # 在第一个子图上绘图
        ax1.hlines(y=y_hlines, xmin=0, xmax=len(self.df), color='grey', linestyles='dashed')
        step = int(len(self.df) / tick_count)
        if step <= 3:
            step = 3
        ax1.set_xlim(0, len(self.df))
        ax1.grid(True)  # 显示网格线
        ax1.set_xticks(range(len(self.df))[::step])
        ax1.set_xticklabels(['']*len(ax1.get_xticks()))  # 设置x轴的标签为空字符串
        ax1.set_title(f'{plot_name} 费率{comm}', fontsize=fontsize)
        ax1.legend()
        ax1.set_ylabel(f'累计份额', fontsize=fontsize)
        ax2.plot(range(len(self.df)), self.df.position, color='tab:grey') # 仓位
        # position.scatter(range(len(self.df)), self.df.position, color='red', s=0.001)
        ax2.set_xlim(0, len(self.df))
        # ax2.set_xticks(range(len(self.df))[::step])
        ax2.set_xticklabels(['']*len(ax1.get_xticks()))  # 设置x轴的标签为空字符串
        ax2.set_ylabel(f'仓位', fontsize=fontsize)
        strategy_curve = np.array(self.df.strategy_curve.fillna(method='ffill').fillna(1))
        max_drawdowns = (np.maximum.accumulate(strategy_curve) - strategy_curve) / np.maximum.accumulate(strategy_curve)
        stacked = np.stack((max_drawdowns,) * 3, axis=-1)
        from matplotlib.ticker import FixedFormatter
        ax3.plot(range(len(self.df)), -max_drawdowns, color='tab:purple')
        ax3.set_xlim(0, len(self.df))
        ax3.set_xticks(range(len(self.df))[::step])
        ytick_list = [-np.round(max(max_drawdowns),3),-np.round(max(max_drawdowns) * 0.5,3), 0]
        ax3.set_yticks(ytick_list)
        ax3.set_yticklabels([f'{val:.0%}' for val in ytick_list])
        ax3.set_xticklabels(datetime_index.strftime('%Y-%m-%d')[::step])
        ax3.set_ylabel(f'最大回撤', fontsize=fontsize)

        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)

        plt.xticks(rotation = 0)
        plt.subplots_adjust(right=0.85) 
        plt.tight_layout()  # 为了避免子图之间的重叠，我们可以使用 tight_layout
        plt.savefig(f"./result/backtest/{plot_name} {comm}.png", bbox_inches='tight', dpi=300, pad_inches=0.0)
        if show_bool == True:
            plt.show()
        return fig # TODO

class BackTester(object):
    """The Vectorized BackTester class.
    仅支持ALL-IN
    # TODO: 增加position_size
    BackTester is a vectorized backtest for quantitative trading strategies.

    Methods:
        run_():
        optimize_():
    Note:
        1. data在创建时只是浅拷贝, 而在创建回测环境时, 我们进行了一次深拷贝
    """
    __metaclass__ = ABCMeta # The ABCMeta metaclass allows the class to be treated as an abstract base class (ABC), which can define abstract methods and enforce their implementation in subclasses.

    @staticmethod # 静态方法，不需要访问类的任何特定状态或属性
    def process_strategy(func):
        @wraps(func) # 装饰器，用来保证函数属性不变
        def wrapper(*args, **kwargs):
            self = args[0] # 此行代码是获取传入参数的第一个元素，通常是类的实例引用。注意这在静态方法中是非标准的，但在这种特定的装饰器上下文中，这是一种技巧，使得装饰器可以在实例方法上工作。
            self.create_backtest_env()
            if set(self.params).issubset(set(kwargs.keys())):
                for name in self.params:
                    self.params[name] = kwargs[name]
            else:
                for idx, name in enumerate(self.params):
                    self.params[name] = args[1:][idx]
            # 上面是对self进行的修改
            func(self, *args[1:], **kwargs) # 前面和后面是对函数附加的处理 # 调用原始的被装饰的函数，并传递修改后的参数给它。
            assert 'signal' in self.backtest_env.columns, "未计算信号"
            assert 'position' in self.backtest_env.columns, "未填充下期持仓信息, 请重新填写"
            assert self.backtest_env['position'].isnull().sum() == 0, "无任何交易, 结束统计"
            # 计算strategy和benchmark的收益率以及净值
            self.backtest_env['benchmark'] = np.log(
                self.backtest_env['transact_base'] / self.backtest_env['transact_base'].shift(1))
            self.backtest_env['strategy'] = self.backtest_env['position'] * self.backtest_env['benchmark']
            self.backtest_env['benchmark_curve'] = self.backtest_env['benchmark'].cumsum().apply(np.exp)
            self.backtest_env['strategy_curve'] = self.backtest_env['strategy'].cumsum().apply(np.exp)

            # 计算交易费用
            self.commission_lst = list()
            if self.buy_commission is not None and self.sell_commission is not None:
                fees_factor = pd.Series(np.nan, index=self.backtest_env.index)
                fees_factor[:] = np.where((self.backtest_env.position - self.backtest_env.position.shift(1)) > 0,
                                          -(self.backtest_env.position - self.backtest_env.position.shift(
                                              1)) * self.buy_commission, np.nan) # TODO: 这里似乎有疏忽，应该根据变动大小来定。
                fees_factor[:] = np.where((self.backtest_env.position - self.backtest_env.position.shift(1)) < 0,
                                          (self.backtest_env.position - self.backtest_env.position.shift(
                                              1)) * self.sell_commission, fees_factor)
                fees_factor.fillna(0, inplace=True)
                fees_factor += 1
                # self.commission_lst.append()
                self.fees_factor = fees_factor
                fees_factor.value_counts()
                self.backtest_env['strategy_curve'] *= fees_factor.cumprod()
                self.backtest_env['strategy'] = np.log(self.backtest_env['strategy_curve']/self.backtest_env['strategy_curve'].shift(1))
                self.backtest_env['strategy'].fillna(0, inplace=True)
                # 风险评估
            result = dict()
            result['params'] = tuple(self.params.values())
            result['annualized_mean'] = 252 * self.backtest_env['strategy'].groupby(self.backtest_env.index.date).sum().mean() # result['annualized_mean'] = 252 * self.freq_base * self.backtest_env['strategy'].mean()
            result['annualized_std'] = np.sqrt(252) * self.backtest_env['strategy'].groupby(self.backtest_env.index.date).sum().std() # np.sqrt(252 * self.freq_base) * self.backtest_env['strategy'].std()
            if result['annualized_std'] != 0:
                result['sharpe_ratio'] = result['annualized_mean'] / result['annualized_std']
            elif result['annualized_std'] == 0 and result['annualized_mean'] == 0:
                result['sharpe_ratio'] = 0
            elif result['annualized_std'] == 0 and result['annualized_mean'] < 0:
                result['sharpe_ratio'] = -999
            elif result['annualized_std'] == 0 and result['annualized_mean'] > 0:
                result['sharpe_ratio'] = 999
            cummax_value = np.maximum.accumulate(self.backtest_env['strategy_curve'].fillna(1))
            result['max_drawdown'] = np.max((cummax_value - self.backtest_env['strategy_curve'])/cummax_value)
            result['signal_counts'] = np.sum(np.abs(self.backtest_env['signal']))
            return result

        return wrapper # 返回修改后的函数

    def __init__(self,
                 symbol_data: pd.DataFrame,
                 transact_base='PreClose',
                #  commissions=(None, None),
                #  commissions=(0.000023, 0.000023),
                 commissions=(0.23, 0.23),
                 slippage_rate=None):

        assert isinstance(symbol_data, pd.DataFrame)
        assert isinstance(symbol_data.index, pd.DatetimeIndex)
        for attr in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert attr in symbol_data.columns

        self.data = symbol_data
        self.freq_base = self.data['Close'].resample('d').apply(lambda x: len(x))[0] # 每天有多长
        self.transact_base = transact_base
        self.backtest_env = ... # 占位符
        self.params = ...
        self.buy_commission = commissions[0]
        self.sell_commission = commissions[1]
        self.slippage_rate = slippage_rate
        self.init()
        self.check_signals = False

    def create_backtest_env(self) -> None:
        self.backtest_env = self.data.copy(deep=True)
        # if self.transact_base == 'PreClose':
        if self.transact_base == 'PreClose':
            self.backtest_env['transact_base'] = self.data['Close']
        elif self.transact_base == 'Open':
            self.backtest_env['transact_base'] = self.data['Open'].shift(-1)
            self.backtest_env['transact_base'].fillna(method='ffill') # 最后一期没数据
        else:
            raise ValueError(f'transact_base must be "PreClose" or "Open", get {self.transact_base}')
        self.backtest_env['signal'] = np.nan
        self.backtest_env['position'] = np.nan

    @abstractmethod
    def init(self):
        """
        在调用类的构造函数时，自动调用该函数
        """
        self.params = ...
        raise NotImplementedError

    @property
    def params_name(self):
        try:
            return list(self.params.keys())
        except AttributeError:
            self.init()
            return list(self.params.keys())

    @abstractmethod
    @process_strategy.__get__(object)
    def run_(self, *args, **kwargs) -> dict[str: int]: # 这里run是可以在外部更新的函数
        """Add the signal and position to the column of the backtest_env.
        and calculate the risk indicators.
        """
        self.backtest_env.position = ...
        raise NotImplementedError("run_ must be implemented")

    def construct_position_(self,
                            keep_raw=False,
                            min_holding_period=None,
                            max_holding_period=None,
                            take_profit=None,
                            stop_loss=None):
        """Modify the position of the backtest_env.
        """
        assert 'signal' in self.backtest_env.columns, '未计算信号'
        self.backtest_env['position'] = self.backtest_env['signal'].shift(1)

        if take_profit is not None and stop_loss is not None:
            mark = pd.Series(np.nan, index=self.backtest_env.index)
            mark[:] = np.where(((self.backtest_env['position'] == 1) +
                                (self.backtest_env['position'] == -1)) > 0, self.backtest_env['transact_base'], np.nan)
            mark.fillna(method='ffill', inplace=True)
            up_band = mark * (1 + take_profit)
            low_band = mark * (1 - stop_loss)
            self.backtest_env['position'] = np.where(self.backtest_env['transact_base'] > up_band, 0,
                                                     self.backtest_env['position'])
            self.backtest_env['position'] = np.where(self.backtest_env['transact_base'] < low_band, 0,
                                                     self.backtest_env['position'])

        if keep_raw: # 决定是否为0就平仓
            self.backtest_env['position'].fillna(0, inplace=True)
        else:
            if max_holding_period is not None:
                self.backtest_env['position'].fillna(method='ffill', limit=max_holding_period, inplace=True)
                self.backtest_env['position'].fillna(0, inplace=True)
            else:
                raise ValueError('max_holding_period should not be None if keep_raw is False')
        self.backtest_env.loc[self.backtest_env.index[0], 'position'] = 0 # 起始仓位为0

    def optimize_(self,
                  goal='sharpe_ratio',
                  method='grid',
                  n_jobs=1,
                  **kwargs):
        """
        :param goal: 优化的目标
        :param method:
        :param n_jobs: 进程数
        :return: The best parameters of the backtest_env.
        """
        assert goal in ['annualized_mean', 'annualized_std', 'sharpe_ratio']  # TODO: 增加其它的指标
        for name in self.params:
            assert name in kwargs
            assert isinstance(kwargs[name], abc.Iterable)

        temp = itertools.product(*[kwargs[x] for x in self.params])
        if method == 'grid':
            if n_jobs > 1:
                print('调用并行')
                with multiprocessing.Pool(n_jobs) as p:
                    results = p.starmap(self.run_, temp)
            else:
                print('不调用并行')
                results = [self.run_(*args) for args in temp]
            rlt = max(results, key=lambda x: x[goal])
            return rlt

    def summary(self, *args, **kwargs) -> Displayer:
        return Displayer(self.backtest_env)

    def clear(self):
        del self.backtest_env

    @staticmethod
    def cross_up(series1, series2):
        assert isinstance(series1, pd.Series)
        assert isinstance(series2, pd.Series)
        return (series1 > series2) * (series1.shift(1) < series2.shift(1))

    @staticmethod
    def cross_down(series1, series2):
        assert isinstance(series1, pd.Series)
        assert isinstance(series2, pd.Series)
        return (series1 < series2) * (series1.shift(1) > series2.shift(1))
