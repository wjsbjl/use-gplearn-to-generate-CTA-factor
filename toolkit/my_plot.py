import sys
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
import matplotlib.ticker as mticker
from itertools import product
from scipy import stats
import os
import colorsys
import random
if not os.path.exists('./result/plot/'):   #os：operating system，包含操作系统功能，可以进行文件操作
    os.makedirs('./result/plot/') #如果存在那就是这个result_path，如果不存在那就新建一个
if os.name == 'posix': # 如果系统是mac或者linux
    plt.rcParams['font.sans-serif'] = ['Songti SC'] #中文字体为宋体,还可以选['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 在windows系统下显示微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 负号用 ASCII 编码的-显示，而不是unicode的 U+2212
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
# pd.set_option('display.max_rows', 20)
# pd.set_option('display.max_columns', 10)
# 把轴隐藏掉
# ax.axes.get_xaxis().set_visible(False) 
# ax.axes.get_yaxis().set_visible(False) 
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_color_gradient(index, total):
    hue = index / total
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)  # 调整饱和度和亮度
    return f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})'

class my_plot():  # 后面再封装一些其他函数
    def __init__(self, plot_df, plot_name = ['Title', 'Xlabel', 'Ylabel', 'save_name']):
        self.plot_df = pd.DataFrame(plot_df)
        self.plot_name = plot_name
        
    def save_plot(self):
        plt.savefig(f"./result/plot/{self.plot_name[3]}.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)

    def line_plot(self, type = ['-'], scale = [''], legend = True, ncol = 0, font_scale = 0.7, x_label_num = 6,
                  save_bool = True,
                  legend_loc = 'best', rotation_angel = 0, x_log = False, y_log = False):  # name包括title，xlabel，ylabel，save_name, type:, 'linear'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        plt.rcParams['axes.unicode_minus'] = False
        fontsize = 12
        x = self.plot_df.index
        y_labels = self.plot_df.columns
        for i in trange(len(y_labels)):
            clmn = y_labels[i]
            ax.plot(x, self.plot_df.loc[:, clmn].values, type[0], label=clmn)
        ax.grid()
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        if ncol > 0:
            font_scale = 0.75
            plt.legend(fontsize = fontsize * font_scale, loc = legend_loc,ncol = ncol, bbox_to_anchor=(1, -0.1))
            # plt.legend(bbox_to_anchor=(0.663, -0.1), ncol = 10)
        else:
            plt.legend()
        self.set_log_axis(ax, x_log, y_log, x_label_num)
        plt.xticks(rotation = rotation_angel)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        if save_bool:
            self.save_plot()
        plt.show()
        
    def line_go_area(self, rainbow_len = None):
        fig = go.Figure()
        if rainbow_len is None:
            for col in self.plot_df.columns:
                fig.add_trace(go.Scatter(x=list(self.plot_df.index), y=list(self.plot_df[col]), name=col, line=dict()))
        else:
            for i, col in enumerate(self.plot_df.columns):
                color = get_color_gradient(i, rainbow_len)
                fig.add_trace(go.Scatter(x=list(self.plot_df.index), y=list(self.plot_df[col]), name=col, line=dict(color=color)))

        fig.update_layout(
            title_text=f"{self.plot_name[0]}",
            xaxis=dict(title_text=f"{self.plot_name[1]}"),
            yaxis=dict(title_text=f"{self.plot_name[2]}")
        )
        fig.write_html(f"./result/plot/{self.plot_name[3]}.html")
        fig.show()
                
    def line_go_drag(self, is_date = False):
        data = []
        for column in self.plot_df.columns:
            data.append(go.Scatter(x=self.plot_df.index, y=self.plot_df[column], name=column))
        if is_date == True:
            type="date"
        else:
            type = None
            
        layout = go.Layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")])),
                rangeslider=dict(visible=True),
                type = type))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(title_text=f"{self.plot_name[0]}")
        fig.update_xaxes(title_text=f"{self.plot_name[1]}")
        fig.update_yaxes(title_text=f"{self.plot_name[2]}")
        # pio.write_image(fig, f'./result/plot/{self.plot_name[3]}', format='png')
        # fig.write_image(f'./result/plot/{self.plot_name[3]}.png', scale = 5)
        fig.write_html(f"./result/plot/{self.plot_name[3]}.html")
        fig.show()

    def hist_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        fontsize = 12
        ax.hist(self.plot_df,bins = 20)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        plt.savefig(f"./result/plot/{self.plot_name[3]}_hist_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
    
    def quantiles_plot(self):
        qtl_df = self.plot_df
        qtl_df.dropna(inplace=True)
        qtl_df = qtl_df.rank() / len(qtl_df)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        fontsize = 12
        ax.scatter(qtl_df['x'], qtl_df['y'], s=0.00001)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        ax.grid()
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.savefig(f"./result/plot/{self.plot_name[3]}_quantiles_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.show()
    
    def scatter_plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        fontsize = 12
        ax.scatter(self.plot_df.index, self.plot_df.values, s=10)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        ax.grid()
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.savefig(f"./result/plot/{self.plot_name[3]}_quantiles_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)

    def bar_plot(self, type = ['-'], scale = [''], legend = True, ncol = 100, font_scale = 0.7, legend_loc = 'best', rotation_angel = 0, width = 0.3):  # name包括title，xlabel，ylabel，save_name, type:, 'linear'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
        fontsize = 12
        x = self.plot_df.index
        y_labels = self.plot_df.columns
        for i in trange(len(y_labels)):
            clmn = y_labels[i]
            ax.bar(x, self.plot_df.loc[:, clmn].values, label=clmn, alpha=0.3, width=width)
        ax.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        ax.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        plt.xticks(rotation = rotation_angel)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.savefig(f"./result/plot/{self.plot_name[3]}_bar_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        
    def multi_hist_plot(self, nrows, ncols):
        figsize = (ncols * ncols + nrows, nrows * nrows + ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,dpi = 300)

        fontsize = 12
        pairs = product(range(0, nrows), range(0, ncols))
        for i,  k in zip(pairs,range(nrows * ncols)):
            plot_series = self.plot_df.iloc[:,k]
            axes[i].hist(plot_series, density = True, bins = 20)
            axes[i].set_title(f'{self.plot_df.columns[k]}{self.plot_name[0]}频率直方图', fontsize=fontsize)
            axes[i].set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
            axes[i].set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
            mu, sigma = plot_series.mean(), plot_series.std()# plot_list.describe()['mean'], plot_list.describe()['std']
            norm_x = np.sort(plot_series.dropna())
            norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (norm_x - mu))**2))
            axes[i].plot(norm_x, norm_y, '--', color='tab:orange')
        plt.savefig(f"./result/plot/{self.plot_name[3]}_hist_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        
    def multi_line_plot(self, nrows, ncols):  # name包括title，xlabel，ylabel，save_name, type:, 'linear'
        figsize = (ncols * ncols + nrows, nrows * nrows + ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,dpi = 300)
        fontsize = 12
        x = self.plot_df.index
        pairs = product(range(0, nrows), range(0, ncols))
        for i,  k in zip(pairs,range(nrows * ncols)):
            plot_series = self.plot_df.iloc[:,k]
            param1 = plot_series.mean()
            param2 = plot_series.std() 
            
            y = np.sort(plot_series.values) # 纵轴是真实分布
            y_rank = np.sort(plot_series.rank(pct=True))
            x = np.sort(stats.norm.ppf(y_rank, param1, param2)) # 横轴是理论分布

            axes[i].plot(x, y, '.')
            axes[i].set_title(f'{self.plot_df.columns[k]}{self.plot_name[0]}QQ图', fontsize=fontsize)
            axes[i].set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
            axes[i].set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
            axes[i].grid()
            axes[i].plot(y, y, '--', color='tab:orange')
        plt.savefig(f"./result/plot/{self.plot_name[3]}_multi_line_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)

    def price_volume_create(self, plot_name = 'Ylabel2'):
        fig = plt.figure(figsize=(10, 6))  # 创建画布
        gs = gridspec.GridSpec(4, 1)  # 定义网格，4行1列
        ax1 = fig.add_subplot(gs[:3, :])  # 第一张图占据前3行
        ax2 = fig.add_subplot(gs[3, :])  # 第二张图占据最后一行
        fontsize = 12
        ax1.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        for i in range(len(self.plot_df.T)):
            clmn = self.plot_df.columns[i]
            ax1.plot(self.plot_df.index, self.plot_df.loc[:, clmn].values, label=clmn)  # 在第一个子图上绘图
        ax1.legend()
        ax1.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        ax2.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax2.set_ylabel(f'{plot_name}', fontsize=fontsize)
        sys.stdout = open(os.devnull, 'w')
        return ax1, ax2
    
    # 量价图 调整为最后一列是副轴 TODO: 后续考虑一下怎么略去重复的
    def price_volume_plot(self, ax1, ax2, plot_name = 'Ylabel2', width = 0.4, x_log = False, y_log1 = False, y_log2 = False,
                          rotation_angle = 0, ax2_type = 'bar', x_label_num = 6,
                          plt_save = True):
        color_set = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fontsize = 12
        plot_df1 = self.plot_df.copy()
        plot_df1.iloc[:,-1] = np.nan
        x = plot_df1.index
        y_labels = plot_df1.columns
        for i in range(len(y_labels)):
            clmn = y_labels[i]
            ax1.plot(x, plot_df1.loc[:, clmn].values, label=clmn, color = color_set[i])  # 在第一个子图上绘图
        self.set_log_axis(ax1, x_log, y_log1, x_label_num)
        ax1.grid(True)  # 显示网格线
        # ax1.set_xticks(ax1.get_xticks())  # Remove x-tick labels by setting them to an empty list
        ax1.set_xticklabels(['']*len(ax1.get_xticks()))  # 设置x轴的标签为空字符串
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        
        plot_df2 = self.plot_df.copy()
        plot_df2.iloc[:,:-1] = np.nan
        if ax2_type == 'linear':
            for i in range(len(y_labels)):
                clmn = y_labels[i]
                ax2.plot(x, plot_df2.loc[:, clmn].values, label=clmn, color = color_set[i])  # 在第一个子图上绘图
            ax2.grid(True)
        elif ax2_type == 'bar':
            ax2.bar(x, self.plot_df.iloc[:, -1].values, label=self.plot_df.columns[-1], alpha=0.3, width=width)  # 在第二个子图上绘图，并设置颜色为skyblue
        self.set_log_axis(ax2, x_log, y_log2, x_label_num)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        ax2.set_xlim(ax1.get_xlim())
        plt.xticks(rotation = rotation_angle)
        plt.subplots_adjust(right=0.85) 
        plt.tight_layout()  # 为了避免子图之间的重叠，我们可以使用 tight_layout
        if plt_save:
            plt.savefig(f"./result/plot/{self.plot_name[3]}_bar_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)


    # 量价图 调整为最后一列是副轴 TODO: 后续考虑一下怎么略去重复的
    def price_volume_plot_origin(self, plot_name = 'Ylabel2', width = 0.4, x_log = False, y_log1 = False, y_log2 = False,
                          rotation_angle = 0, ax2_type = 'bar', x_label_num = 6):
        fig = plt.figure(figsize=(10, 6))  # 创建画布
        gs = gridspec.GridSpec(4, 1)  # 定义网格，4行1列
        ax1 = fig.add_subplot(gs[:3, :])  # 第一张图占据前3行
        ax2 = fig.add_subplot(gs[3, :])  # 第二张图占据最后一行
        fontsize = 12
        plot_df1 = self.plot_df.copy()
        plot_df1.iloc[:,-1] = np.nan
        x = plot_df1.index
        y_labels = plot_df1.columns
        for i in range(len(y_labels)):
            clmn = y_labels[i]
            ax1.plot(x, plot_df1.loc[:, clmn].values, label=clmn)  # 在第一个子图上绘图
        ax1.set_title(f'{self.plot_name[0]}', fontsize=fontsize)
        self.set_log_axis(ax1, x_log, y_log1, x_label_num)
        ax1.legend()
        ax1.grid(True)  # 显示网格线
        ax1.set_ylabel(f'{self.plot_name[2]}', fontsize=fontsize)
        # ax1.set_xticks(ax1.get_xticks())  # Remove x-tick labels by setting them to an empty list
        ax1.set_xticklabels(['']*len(ax1.get_xticks()))  # 设置x轴的标签为空字符串
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        
        plot_df2 = self.plot_df.copy()
        plot_df2.iloc[:,:-1] = np.nan
        if ax2_type == 'linear':
            for i in range(len(y_labels)):
                clmn = y_labels[i]
                ax2.plot(x, plot_df2.loc[:, clmn].values, label=clmn)
            ax2.grid()
        elif ax2_type == 'bar':
            ax2.bar(x, self.plot_df.iloc[:, -1].values, label=self.plot_df.columns[-1], alpha=0.3, width=width)  # 在第二个子图上绘图，并设置颜色为skyblue
        ax2.set_xlabel(f'{self.plot_name[1]}', fontsize=fontsize)
        ax2.set_ylabel(f'{plot_name}', fontsize=fontsize)
        self.set_log_axis(ax2, x_log, y_log2, x_label_num)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        ax2.set_xlim(ax1.get_xlim())
        plt.xticks(rotation = rotation_angle)
        plt.subplots_adjust(right=0.85) 
        plt.tight_layout()  # 为了避免子图之间的重叠，我们可以使用 tight_layout
        plt.savefig(f"./result/plot/{self.plot_name[3]}_bar_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)
        
    def set_log_axis(self, ax, x_log, y_log, x_label_num):
        if x_log == True:
            step_size = len(self.plot_df.index) // (x_label_num)
            selected_index = list(self.plot_df.index[::step_size])
            ax.set_xscale('log')
            ax.set_xticks(selected_index)
            ax.set_xticklabels( [f'{i:.2e}' for i in ax.get_xticks()])
        if y_log == True:
            ax.set_yscale('log')
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels( [f'{i:.2e}' for i in ax.get_yticks()])
            # ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            # ax2.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%1.2e'))
    
    def tab_color(): # 这下面都是写好的，作备份
        {'tab:blue',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan'}
        
    def heat_map(self):
        fig_width = 6.2 * len(self.plot_df) / 8
        fig_height = 5 * len(self.plot_df) / 8
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height),dpi = 100, )
        sns.heatmap(self.plot_df.sort_index(ascending=False), annot=True, cmap="plasma_r", ax=axes)
        # plasma_r是黄到蓝，coolwarm是蓝到红，viridis_r是黄到黑
        axes.set_title(f'{self.plot_name[0]}', fontsize = 11)
        axes.set_xlabel(f'{self.plot_name[1]}', fontsize = 11)
        axes.set_ylabel(f'{self.plot_name[2]}', fontsize = 11)
        plt.savefig(f'./result/plot/{self.plot_name[3]}.png')
	    
    def bar_plot_bak(self): # 这下面是备份的
        fig, ax = plt.subplots(figsize=(10, 10))
        hist_x = [str(x) for x in self.plot_df.index]
        hist_y = self.plot_df.values
        ax.bar(hist_x, hist_y)
        ax.set_xlabel('逐笔交易对应时间',fontsize = 30)
        ax.set_ylabel('频数',fontsize = 30)
        plt.savefig(f"./result/plot/{self.plot_name[3]}_bar_plot.jpg", bbox_inches='tight', dpi=300, pad_inches=0.0)

    def word_cloud_plot(self):
        self.plot_df.columns = ['words','sizes']
        word_sizes = dict(zip(self.plot_df['words'], self.plot_df['sizes']))
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50,random_state=230412,
                    relative_scaling=0.5, prefer_horizontal=0.8, colormap='viridis').generate_from_frequencies(word_sizes)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud of {self.plot_name[0]}", fontsize=30)
        plt.tight_layout(pad=0)
        plt.savefig(f'./result/{self.plot_name[3]}_word_plot.jpg', bbox_inches = 'tight' , dpi=300, pad_inches = 0.0)
        plt.show()

if __name__ == '__main__':  
    df = pd.DataFrame(columns=["a","b"])
    plot_instance = my_plot(df)
    ax1, ax2 = plot_instance.price_volume_create('Ylabel2')
    for i in range(100):
        df.loc[i,:] = [np.sqrt(i), i**2]  
        plot_instance.plot_df = df
        plot_instance.price_volume_plot(ax1, ax2, 'Ylabel2', width=0.15, x_log=False, y_log1=False, ax2_type='linear', x_label_num=6)
        plt.pause(0.1)  # 暂停时间可以根据需要调整
        clear_output(wait=True)
        display(plt.gcf())  # 显示当前图形
      
    row = 100
    col = 10
    random_df1 = pd.DataFrame(data = np.array([random.uniform(-1,1) for i in range(row*col)]).reshape(row,col), index = range(row), columns = range(col))
    df = random_df1.copy()
    df['x'] = np.linspace(2, col, row)
    df['y'] = [np.log(i) for i in df['x']]
    df['x'] = (df['x']-2)*12.5 / 5 / 100
    df['y'] = ((df['y'] - df['y'].min()) ) /1.75*55 + 50
    for i in range(col):
        df.iloc[:,i] = random_df1.iloc[:,i].cumsum() + df['y']
    df = df.set_index('x')
    plot_df = df
    plot_name = ['Title', 'Xlabel', 'Ylabel', 'save_name']
    my_plot(plot_df).line_plot(legend = [1.03, -0.25])
    my_plot(plot_df, plot_name).price_volume_plot('Ylabel2', width = 0.15, x_log = False, y_log1 = False, ax2_type = 'linear', x_label_num=6)
    my_plot(plot_df, plot_name).line_plot(x_log = True)
    my_plot(plot_df, plot_name).line_go_area()
    my_plot(plot_df, plot_name).line_go_drag()
    # my_plot(plot_df.iloc[:,0], plot_name).bar_plot(width = 0.1)
    # my_plot(plot_df, plot_name).line_plot(legend = [1,1], ncol = 1, font_scale = 0.8)
    plot_df2 = plot_df.diff().dropna(how = 'all')
    plot_df2 = np.ceil(np.abs((plot_df2/0.5))) * np.sign(plot_df2)
    plot_df3 = plot_df2.iloc[:,0].value_counts()
    my_plot(plot_df3, plot_name).bar_plot()    

    random_df2 = pd.DataFrame(np.random.random((5, 5)))
    my_plot(random_df2).heat_map()