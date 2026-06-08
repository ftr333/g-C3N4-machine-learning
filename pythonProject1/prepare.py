import pandas as pd
import numpy as np
import time
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy import interpolate

from sklearn import metrics
import matplotlib.pyplot as plt
import os
from fancyimpute import KNN

from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from torch.ao.nn.quantized.functional import interpolate


def draw_partial_dependence_2d(model,x_train,config,save = False):
    pdp = partial_dependence(model,x_train,features=config['features'],kind='average',grid_resolution=config['grid_resolution'])
    #提取坐标
    plot_x = pdp['grid_values'][0]
    plot_y = pdp['average'].ravel()

    # 生成新的网格
    xnew = np.linspace(plot_x.min(),plot_x.max(),300)

    #样条插值
    spl = make_interp_spline(plot_x,plot_y,k = 3 )
    ynew = spl(xnew)

    #绘制曲线
    plt.plot(xnew,ynew,linewidth=1,color=config['color'],antialiased=True)

    #设置字体
    font = {'family' : 'Times New Roman',
            'size' : 16,'weight' : 'bold',
            'color' : 'black'}

    #保存和显示
    if save:
        os.makedirs(f'{config["model_path"]}',exist_ok=True)
        plt.savefig(f'{config["model_path"]}/{config["model_name"]}.jpg',dpi=300)

    plt.show()

columns_need_fill = ['Urea','Melamine','Dcda',
                     'Time(h)','PM(g)','Heat(℃)',
                     'HR(℃/min)','ratio(%)',
                    'load(g)','NO(ppm)','rate(mL/min)'
                     ,'Xe','Wu','LED','area(cm²)',
                    'Intensity(W)','BET(m²/g)','Eg(eV)','NO₂(%)']
train_properties_dict = {'Urea':'Urea',
                         'Melamine':'Mela',
                         'Dcda':'DCDA',
                         'Time(h)':'T',
                         'area(cm²)':'area',
                         'PM(g)':'PM',
                         'HR(℃/min)':'HR',
                         'Heat(℃)':'H',
                         'ratio(%)':'ratio',
                         'load(g)':'LD',
                         'NO(ppm)':'NO',
                         'rate(mL/min)':'Rate',
                         'Xe':'Xe',
                         'Wu':'Wu',
                         'LED':'Led',
                         'NO₂(%)':'NO₂',
                         'Intensity(W)':'W',
                         'BET(m²/g)':'BET(m²/g)',
                         'Eg(Eg)':'Eg',
                         }

heatmap_feature_name = ['Urea','Mela','DCDA','T','PM','HR','H','ratio','NO2',
                        'LD','NO','Rate','Xe','Wu','Led','area','W','BET','Eg']

def train_properties_and_names():
    properties = []
    names = []
    for k, v in train_properties_dict.items():
        properties.append(k)
        names.append(v)

    return (properties, names)


train_properties, feature_names = train_properties_and_names()

def read_csv(filepath, encoding='utf8', index='id'):
    datas = pd.read_csv(filepath, encoding=encoding)
    '''
    检查索引是否在我们的数据中，不在
    插入一个index，即新的列，index='id',代表新一列的名称为id
    '''
    if index not in datas.columns:
        datas.insert(0, index, (range(len(datas))), allow_duplicates=False)
    return datas

def save_csv(datas, filepath, encoding='utf8'):
    datas = datas.loc[:, ~datas.columns.str.contains("Unnamed")]
    datas.to_csv(filepath, encoding=encoding, header=True, index=False)

def merge_data(datas, filled_data):
    merged_data = datas.copy()
    for col in filled_data.columns:
        merged_data[col] = filled_data[col]

    merged_data = merged_data.loc[:, ~merged_data.columns.str.contains("Unnamed")]
    return merged_data

def fill_data(datas, imputer=KNN(k=9, verbose=False)):
    start_time = time.time()
    print("Begin to process fill data.")
    selected_data = pd.DataFrame(datas, columns=columns_need_fill)
    filled_datas = pd.DataFrame((imputer.fit_transform(selected_data)), columns=(selected_data.columns))
    merged_datas = merge_data(datas, filled_datas)
    end_time = time.time()
    print("Finished! Speed time: {} s.".format(end_time - start_time))
    return merged_datas

def draw_heatmap(data, method, cmap,size = 15, save=False):
    corr = data.corr(method=method)

    # 动态调整图形尺寸
    n_cols = len(data.columns)
    figsize = (n_cols * 1.5, n_cols * 1.5)

    # 设置全局字体和图形
    sns.set(font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots(figsize=figsize)

    # 生成掩码（排除对角线）
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # 绘制热力图
    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap=cmap,
        square=True,
        ax=ax,
        annot=True,
        annot_kws={'size': 15, 'weight': 'bold', 'color': 'black'},
        cbar_kws={"shrink": 0.75},
        mask=mask
    )
    #自定义标签
    x_ticklabels = ['Urea','Melamine','Dcda','PM(g)','Time(h)', 'Heat(℃)','HR(℃/min)',
                    'hata','modefiy', 'dope', 'ratio(%)','load(g)','area(cm²)','NO(ppm)',
                    'rate(mL/min)', 'Xe', 'Wu', 'LED','Intensity(W)','BET(m²/g)',
                    'Eg(eV)','NO₂(%)']

    # 调整坐标轴标签
    ax.set_xticklabels(
        x_ticklabels,
        rotation = 45,
        ha = 'center',
        fontsize = size,
    )
    ax.set_yticklabels(
        x_ticklabels,
        rotation=0,
        ha='right',
        fontsize = size,
    )

    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)


    # 保存图像
    if save:
        output_dir = 'heatmap_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir,f'{method}.jpg')
        plt.savefig(output_path,format = "jpg",dpi = 600,
                    bbox_inches = "tight",
                    pil_kwargs = {'quality' : 95})
        print(f"热力图已经保存到：{output_path}")
    plt.show()



def split_without_z_score_normalize(properties, target, test_size=0.2, seed=55):
    x_train, x_test, y_train, y_test = train_test_split(properties, target, test_size=test_size, random_state=seed)
    train_ids, test_ids = x_train["id"], x_test["id"]
    x_train.drop(columns="id", inplace=True)
    x_test.drop(columns="id", inplace=True)
    return (x_train, x_test, y_train, y_test, train_ids, test_ids)
def split_and_z_score_normalize(properties, target, test_size=0.2, seed=55):
    x_train, x_test, y_train, y_test = train_test_split(properties, target, test_size=test_size, random_state=seed)
    train_ids, test_ids = x_train["id"], x_test["id"]
    x_train.drop(columns="id", inplace=True)
    x_test.drop(columns="id", inplace=True)
    scaler = StandardScaler().fit(x_train)
    x_train_scaler_data, x_test_scaler_data = scaler.transform(x_train), scaler.transform(x_test)
    x_train_scaler, x_test_scaler = pd.DataFrame(x_train_scaler_data), pd.DataFrame(x_test_scaler_data)
    x_train_scaler.columns, x_test_scaler.columns = x_train.columns, x_test.columns
    return (x_train_scaler, x_test_scaler, y_train, y_test, train_ids, test_ids)
def split_train_test(datas, test_size=0.2, seed=55, normalize=True, target_col=["NO(%)"]):
    train_properties_columns = train_properties.copy()
    train_properties_columns.append("id")
    properties = pd.DataFrame(datas, columns=train_properties_columns)
    target = pd.DataFrame(datas, columns=target_col)
    if normalize:
        return split_and_z_score_normalize(properties, target, test_size, seed)
    return split_without_z_score_normalize(properties, target, test_size, seed)


def get_mse_rmse_mae_mape_smape_r2(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    r2 = metrics.r2_score(y_true, y_pred)
    return (mse, rmse, mae, mape, smape, r2)


def model_performance_metrics(label, true, pred):
    mse, rmse, mae, mape, smape, r2 = get_mse_rmse_mae_mape_smape_r2(true.values.ravel() / 100, pred / 100)
    print("%s mse: %.5f | rmse: %.5f | mae: %.5f | mape: %.5f | smape: %.5f | r2: %.5f\n" % (
     label, mse, rmse, mae, mape, smape, r2))
    return r2


def cross_validation(regressor, x_train, y_train, k=5):
    scores = cross_val_score(regressor, x_train, (y_train.values.ravel()),
      cv=k, scoring="r2")
    score = np.mean(scores)
    print("cv train R2: %.3f (+/- %.3f) \n" % (score, scores.std()))


def process_regressor(regressor, x_train, x_test, y_train, y_test, cv=True):
    if cv:
        cross_validation(regressor, x_train, y_train)
    regressor.fit(x_train, y_train.values.ravel())
    return (
     regressor.predict(x_train), regressor.predict(x_test))


def draw_scatter(train_true, train_pred, test_true, test_pred, title,
                  actual_text='actual', pred_text='prediction',save=False):

    # 统一字体设置
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.sans-serif'] = ['Arial']  # 备用字体

    # 设置整张图背景颜色
    '''
    1、创建Figure时直接设置
    plt.figure(facecolor='w'))
    
    2、对现有的Figure设置
    plt.gcf()  #获取当前的图
    fig.set_facecolor('white')
    '''
    #设置的是整个图的颜色
    #plt.figure(facecolor='lightgrey')  #还可以使用RGB元组(0.8,0.8,0.8)取值范围0~1

    # 定义字体样式
    font_common = {
        'family': 'Times New Roman',
        'color': '#2F4F4F',  # 深灰色
        'weight': 'bold',
        'size': 18
    }

    # 绘制散点图
    sns.regplot(x=test_true, y=test_pred, color="orange", label="test")
    sns.regplot(x=train_true, y=train_pred, color="dodgerblue", label="train")

    # 设置坐标轴标签和范围
    plt.xlabel(actual_text, fontdict=font_common,weight='bold')
    plt.ylabel(pred_text, fontdict=font_common,weight='bold')
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # 设置图例
    legend_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.legend(loc="upper right", markerscale=1.5, prop=legend_font,frameon=True, bbox_to_anchor=(0.3, 1))
    '''
    frameon:设置图例是否具有边框，bbox_to_anchor调整图例位置
    loc:控制图例在绘图区内的基准位置，可以配合bbox_to_anchor使用
    loc:    best   0自动化选择最好位置
            upper right 右上角1
            upper left  左上角2
            lower left  左下3
            lower right 右下4
            right  垂直居中 右侧5            center left 左侧垂直居中6
    '''


    #获取图中绘制的图
    ax = plt.gca()

    # 设置主刻度线 - 更明显的显示
    ax.tick_params(
        axis='both',          # 同时设置x轴和y轴
        which='major',        # 主刻度线
        direction='out',       # 刻度线方向：向内
        length=8,            # 增加刻度线长度
        width=2,             # 增加刻度线宽度
        color='black',       # 刻度线颜色
        labelsize=14,        # 标签大小
        bottom=True,         # 显示底部刻度线
        left=True,           # 显示左侧刻度线
        top=False,            # 显示顶部刻度线
        right=False           # 显示右侧刻度线
    )
    #设置颜色
    ax.set_facecolor('white')

    #设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    #设置图内网格
    plt.grid(True,
             axis='both',
             linestyle='--',
             linewidth=0.5,
             color='gray',
             alpha=0.6)

    # 添加统计信息文本
    _, _, _, _, _, train_r2 = get_mse_rmse_mae_mape_smape_r2(train_true.values.ravel() / 100, train_pred / 100)
    _, test_rmse, _, _, _, test_r2 = get_mse_rmse_mae_mape_smape_r2(test_true.values.ravel() / 100, test_pred / 100)

    plt.text(60, 26, f"Train R² = {(int(train_r2*1000)/1000):.3f}", **font_common)
    plt.text(60, 18, f"Test R² = {test_r2:.3f}", **font_common)
    plt.text(60, 10, f"RMSE = {test_rmse:.3f}", **font_common)

    # 添加标题
    plt.text(50, 90, title, **font_common, horizontalalignment="center")

    # 保存图像
    if save:
        out_put = 'scatter_images'
        if not os.path.exists(out_put):
            os.makedirs(out_put)
        output_path = os.path.join(out_put, f'{title}.jpg')
        plt.savefig(output_path, format="jpg", dpi=600,
                    bbox_inches="tight",
                    pil_kwargs={'quality': 95})
        print(f"测试图已保存：{output_path}")


    plt.show()

def draw_violin(data, config, which, bw_method='scott', save=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 初始化设置
    plt.rcParams['font.family'] = 'Times New Roman'
    cfg = config[which]
    bw_method = cfg.get('bw_method', bw_method)

    # 数据预处理
    data = data[which].values.ravel()
    scale_data = data * cfg['scale']

    # 自动计算区间范围

    min_val, max_val = scale_data.min(), scale_data.max()
    step = cfg['step'] * cfg['scale']

    scale_begin = max(0, int(min_val - step)) if cfg['begin'] < 0 else cfg['begin']
    scale_end = int(max_val + step + 1) if cfg['end'] < 0 else cfg['end']

    bins = range(scale_begin, scale_end, int(step))
    segments = pd.cut(scale_data, bins)

    mid_mapping = {
        interval: interval.mid / cfg['scale']
        for interval in segments.categories
    }

    # 将数据进行进一步集中计数
    mid_points = segments.rename_categories(mid_mapping).astype(float)

    # 绘图核心
    plt.figure(figsize=cfg.get('figsize', (8, 6)))

    # 使用上下文管理器优化绘图参数
    with sns.axes_style("whitegrid"):
        ax = sns.violinplot(
            x=mid_points,
            inner='box',
            bw_method=bw_method,
            color=cfg['color'],
            linewidth=cfg.get('linewidth', 1),
            saturation=cfg.get('saturation', 0.75),
            alpha = cfg.get('alpha',0.7)
        )

    # 添加文本注释cfg['text_x'],cfg['text_y']设定第三个参数出现位置，分别位于x，y处
    plt.text(x = cfg['text_x'],y =cfg['text_y'],s = cfg['text'] ,family='Times New Roman', weight='bold',
             size=(cfg['xtick_size']),
             horizontalalignment='center',
             transform=plt.gcf().transFigure,
             verticalalignment='top',)
    # 设置x轴刻度
    plt.xticks(fontproperties='Times New Roman', size=cfg['xtick_size'], weight='bold')

    # 隐藏y轴刻度和标签
    plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
    '''
    plt.ylabel(
        "Amplitude (m)",  # 标签文本
        fontsize=12,  # 字体大小
        color='blue',  # 字体颜色
        fontweight='bold',  # 字体粗细 ('normal', 'bold', 'heavy')
        fontfamily='Times New Roman',  # 字体类型
        rotation=90,  # 旋转角度（0~360 或 'vertical'/'horizontal'）
        labelpad=30  # 标签与轴的距离（像素）
    )
    '''
    # 设置刻度样式
    plt.tick_params(direction="in", width=2)

    # 设置y轴范围
    plt.ylim(cfg['ylim_left'], cfg['ylim_right'])

    # 获取坐标轴对象ax
    ax = plt.gca()

    #设置x轴名称标签
    ax.set_xlabel(
    cfg['xlabel'],
    fontproperties='Times New Roman',  # 字体
    size=20,                           # 字体大小
    weight='bold',                     # 字体加粗
    labelpad=2                        # 标题与 x 轴的距离
                )
    #设置x轴刻度标签
    ax.set_xticklabels(
    ax.get_xticklabels(),
    fontproperties='Times New Roman',  # 字体
    size=20,                           # 字体大小
    weight='bold',                     # 字体加粗
    rotation=0,                       # 旋转角度
    ha='right',                        # 水平对齐方式
    va='top',                          # 垂直对齐方式
    )
    ax.xaxis.set_tick_params(pad=15)   # 标签与 x 轴的距离
    #设置y轴刻度标签
    ax.set_yticks([cfg['ylim_left'],cfg['ylim_left']/2,0,cfg['ylim_right']/2,cfg['ylim_right']])  # 主刻度位置
    ax.set_yticklabels([cfg['ylim_left'],cfg['ylim_left']/2,0,cfg['ylim_right']/2,cfg['ylim_right']])  # 主刻度标签
    # 设置坐标轴边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(cfg['bound_width'])
        spine.set_color('black')
    #设置图内网格
    plt.grid(True,
             axis='both',
             linestyle='--',
             linewidth=0.5,
             color='gray',
             alpha=0.6)
    # 调整图形布局
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

    # 保存图像

    if save:
        output_dir = 'violin_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'{cfg['image_name']}.jpg')
        plt.savefig(output_path, format="jpg", dpi=600,
                    bbox_inches="tight",
                    pil_kwargs={'quality': 95})
        print(f"小提琴图已保存：{output_path}")
    # 显示图形
    plt.show()

def mid_mapping(data,cfg,key):
    cf = cfg[key]

    #数据预处理
    data = data[key].values.ravel()
    scale_data = data * cf['scale']

    # 自动计算区间范围

    min_val, max_val = scale_data.min(), scale_data.max()
    step = cf['step'] * cf['scale']

    scale_begin = max(0, int(min_val - step)) if cf['begin'] < 0 else cf['begin']
    scale_end = int(max_val + step + 1) if cf['end'] < 0 else cf['end']

    bins = range(scale_begin, scale_end, int(step))
    segments = pd.cut(scale_data, bins)

    mid_mapping = {
        interval: interval.mid / cf['scale']
        for interval in segments.categories
    }
    # 将数据进行进一步集中计数
    mid_points = segments.rename_categories(mid_mapping).astype(float)
    return mid_points

def data_concatenate(df_1,df_2,cfg,key):
    data = []
    data.append(pd.DataFrame({
        'Category': '100%',
        'Group': '100%',
        'Value': mid_mapping(df_1,cfg,key),
    }))
    data.append(pd.DataFrame({
        'Category': '70%',
        'Group': '70%',
        'Value': mid_mapping(df_2,cfg,key),
    }))
    df = pd.concat(data, ignore_index=True)
    return df


def draw_violin_concatenate(data_1, data_2, config, which,i,bw_method='scott', save=False):
    cfg = config[which]
    data = data_concatenate(data_1, data_2, config, which)

    # ====================== 颜色控制优化 ======================
    # 自定义颜色方案（示例使用莫兰迪色系）
    custom_palette = [
        "#54778C",  # 雾霾蓝
        "#7A9D7E"   # 橄榄绿
    ]

    # 从配置中获取颜色（如果存在）
    if 'color' in cfg:
        custom_palette = cfg['color']
    # ========================================================

    # 初始化设置
    plt.rcParams['font.family'] = 'Times New Roman'
    bw_method = cfg.get('bw_method', bw_method)
    plt.figure(figsize=cfg.get('figsize', (8, 6)))
    # ====================== 样式优化 ======================
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })
    # =====================================================

    ax = sns.violinplot(
        x="Category",
        y="Value",
        hue="Group",
        data=data,
        dodge=False,        # 强制并排显示
        split=False,
        palette=custom_palette,  # 使用自定义颜色
        bw_method=cfg.get('bw_method',0.2),
        linewidth=cfg.get('linewidth', 1.5),  # 增加边框线宽
        saturation=1,    # 提高颜色饱和度
        width=0.9,         # 控制整体宽度（新增参数）
        gap=0.01,           # 新增分组间距控制（需要seaborn 0.12+）
        inner='box',       # 显示内部箱线图
        scale='width',        # 统一宽度模式
        alpha = cfg.get('alpha',0.6),
    )

    # ====================== 间距调整优化 ======================
    # 手动调整分组间距（适用于旧版seaborn）
    for violin, alpha in zip(ax.collections[::2], [0.8, 0.8]):
        violin.set_alpha(alpha)  # 设置透明度分层



    # 文本标注优化（原代码保持不变）
    plt.text(x=cfg['text_x'], y=cfg['text_y'], s=cfg['text'],
             family='Times New Roman', weight='bold',
             size=cfg['xtick_size'],
             horizontalalignment='center',
             transform=plt.gcf().transFigure,
             verticalalignment='top')

    # ====================== 坐标轴优化 ======================
    ax.set_xlabel("")  # ✅ 去掉横坐标轴名称
    ax.set_ylabel("")  # ✅ 去掉纵坐标轴名称

    plt.xticks(fontproperties='Times New Roman',
              size=cfg['xtick_size'],
              weight='bold')

    # Y轴标签优化
    plt.yticks(
        fontproperties='Times New Roman',
        size=20,
        weight='bold',
        rotation=0,     # 添加旋转参数
        ha='right',     # 水平对齐方式
        va='center_baseline'  # 垂直对齐优化
    )
    ax.tick_params(
        axis='both',          # 同时设置x轴和y轴
        which='major',        # 主刻度线
        direction='out',      # 刻度线方向：向外
        length=6,            # 刻度线长度
        width=1.5,           # 刻度线宽度
        color='black',       # 刻度线颜色
        bottom=True,         # 显示底部刻度线
        left=True,           # 显示左侧刻度线
        top=False,           # 不显示顶部刻度线
        right=False          # 不显示右侧刻度线
    )
    # 优化坐标轴标签间距
    ax.yaxis.set_tick_params(pad=15)
    ax.xaxis.set_tick_params(pad=10)
    # =======================================================

    # 设置坐标轴边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(cfg['bound_width'])
        spine.set_color('black')
    #设置图内网格
    '''
    #丢弃上和左边框线
    sns.despine(left=True)
    sns.despine(top=True)
    '''
    plt.grid(True,
             axis='both',
             linestyle='--',
             linewidth=0.5,
             color='gray',
             alpha=0.6)
    # 调整图形布局
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)


    if save:
        output_dir = 'violin_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'{cfg['image_name']+ i}.jpg')
        plt.savefig(output_path, format="jpg", dpi=600,
                    bbox_inches="tight",
                    pil_kwargs={'quality': 95})
        print(f"小提琴图已保存：{output_path}")


    plt.show()

def draw_scatter_3(train_true, train_pred,val_true,val_pred,test_true, test_pred,title,
                  actual_text='actual', pred_text='prediction',save=False):

    # 统一字体设置
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.sans-serif'] = ['Arial']  # 备用字体

    # 设置整张图背景颜色
    '''
    1、创建Figure时直接设置
    plt.figure(facecolor='w'))

    2、对现有的Figure设置
    plt.gcf()  #获取当前的图
    fig.set_facecolor('white')
    '''
    #设置的是整个图的颜色
    #plt.figure(facecolor='lightgrey')  #还可以使用RGB元组(0.8,0.8,0.8)取值范围0~1

    # 定义字体样式
    font_common = {
        'family': 'Times New Roman',
        'color': '#2F4F4F',  # 深灰色
        'weight': 'bold',
        'size': 18
    }

    # 绘制散点图
    #scatter_kws使用此函数控制透明度和点大小
    sns.regplot(x=train_true, y=train_pred,
                color="dodgerblue", label= 'train',
                scatter_kws = {'alpha': 0.5, 's': 18})
    sns.regplot(x=test_true, y=test_pred, color= '#FF69B4', label= 'test',
                scatter_kws = {'alpha': 0.8, 's': 30})
    sns.regplot(x=val_true, y= val_pred,
                color="orange", label="val",
                scatter_kws = {'alpha': 0.5, 's': 10})

    # 设置坐标轴标签和范围
    plt.xlabel(actual_text, fontdict=font_common,weight='bold')
    plt.ylabel(pred_text, fontdict=font_common,weight='bold')
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # 设置图例
    legend_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.legend(loc="upper right", markerscale=1.5, prop=legend_font,frameon=True, bbox_to_anchor=(0.3, 0.85))
    '''
    frameon:设置图例是否具有边框，bbox_to_anchor调整图例位置
    loc:控制图例在绘图区内的基准位置，可以配合bbox_to_anchor使用
    loc:    best   0自动化选择最好位置
            upper right 右上角1
            upper left  左上角2
            lower left  左下3
            lower right 右下4
            right  垂直居中 右侧5            center left 左侧垂直居中6
    '''

    # 设置刻度
    plt.tick_params(direction="in", width=2, labelsize=14)

    #获取图中绘制的图
    ax = plt.gca()
    #设置颜色
    ax.set_facecolor('white')

    #设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    #设置图内网格
    plt.grid(True,
             axis='both',
             linestyle='--',
             linewidth=0.5,
             color='gray',
             alpha=0.6)

    # 添加统计信息文本
    _, _, _, _, _, train_r2 = get_mse_rmse_mae_mape_smape_r2(train_true.values.ravel() / 100, train_pred / 100)
    _, _, _, _, _, val_r2 = get_mse_rmse_mae_mape_smape_r2(val_true.values.ravel() / 100, val_pred / 100)
    _, _, _, _, _, test_r2 = get_mse_rmse_mae_mape_smape_r2(test_true.values.ravel() / 100, test_pred / 100)

    plt.text(60, 26, f"Train R² = {(int(train_r2*1000)/1000):.3f}", **font_common)
    plt.text(60, 18, f"val   R² = {int(val_r2*1000)/1000:.3f}", **font_common)
    plt.text(60, 10, f"test  R² = {int(test_r2*1000)/1000:.3f}", **font_common)

    # 添加标题
    plt.text(50, 90, title, **font_common, horizontalalignment="center")

    # 保存图像
    if save:
        out_put = 'scatter_images'
        if not os.path.exists(out_put):
            os.makedirs(out_put)
        output_path = os.path.join(out_put, f'{title}.jpg')
        plt.savefig(output_path, format="jpg", dpi=600,
                    bbox_inches="tight",
                    pil_kwargs={'quality': 95})
        print(f"测试图已保存：{output_path}")


    plt.show()


def draw_partial_dependence_3d(model, x_train, config, save=False):
    # ========== 1. 计算部分依赖 ==========
    pdp = partial_dependence(
        model, x_train,
        features=config['features'],
        kind='average',
        grid_resolution=config['grid_resolution']
    )

    # ========== 2. 数据准备与验证 ==========
    # 提取原始网格点（一维数组）
    x_grid = pdp['grid_values'][0]  # 形状 (nx,)
    y_grid = pdp['grid_values'][1]  # 形状 (ny,)
    Z = pdp['average']              # 形状 (nx, ny)
    XX, YY = np.meshgrid(x_grid, y_grid, indexing='ij')
    tck = interpolate.bisplrep(XX,YY, Z, s=len(x_grid)*len(y_grid)*0.01)


    # ========== 3. 高密度插值 ==========
    # 生成高分辨率网格（100x100）
    x_highres = np.linspace(x_grid.min(), x_grid.max(), 100)
    y_highres = np.linspace(y_grid.min(), y_grid.max(), 100)
    xx, yy = np.meshgrid(x_highres, y_highres, indexing='ij')  # 形状 (100, 100)

    # 双样条插值（修复关键参数）
    zz = interpolate.bisplev(x_highres, y_highres, tck)  # 形状 (100, 100)

    # 强制统一维度
    if zz.shape != xx.shape:
        zz = zz.reshape(xx.shape)  # 确保形状一致

    # ========== 4. 可视化优化 ==========
    plt.rc('font', family='Times New Roman', weight='bold')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 动态颜色映射
    norm = plt.Normalize(zz.min(), zz.max())
    surf = ax.plot_surface(
        xx, yy, zz,
        cmap=config.get('cmap', 'viridis'),
        norm=norm,          # 颜色标准化
        rstride=1,          # 全分辨率渲染
        cstride=1,
        linewidth=0.2,
        antialiased=True,
        edgecolor='grey',
        alpha=0.95
    )

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax,
                        location = 'left',      #关键参数
                        shrink=0.8,             #设置高度缩放
                        anchor = (-0.2,0.5),    #设置坐标点位置
                        aspect=20,              #宽度比例控制
                        pad = -0.15              #间距调整
                        )
    cbar.set_label('Partial Dependence',
                   fontsize=16,
                   weight='bold',
                   labelpad=10,                    #标签与颜色条距离
                   rotation=90,                    #垂直方向
                   horizontalalignment='center',    #水平对齐方式
                   verticalalignment='center',     #垂直对齐方式
                   )
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    #优化主图位置

    #ax.set_position([0.05, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
    # 添加整体布局微调
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

    #坐标标签调整
    plt.xticks(
        fontproperties='Times New Roman',
        size=14,
        weight='bold',
        rotation=0,     # 添加旋转参数
        ha='right',     # 水平对齐方式
        va='center_baseline'  # 垂直对齐优化
    )
    plt.yticks(
        fontproperties='Times New Roman',
        size=14,
        weight='bold',
        rotation=0,     # 添加旋转参数
        ha='right',     # 水平对齐方式
        va='center_baseline'  # 垂直对齐优化
    )



    # 坐标轴标签
    ax.set_xlabel(config['feature1_label'], labelpad=15, fontsize=16,weight='bold')
    ax.set_ylabel(config['feature2_label'], labelpad=15, fontsize=16,weight='bold')
    ax.set_zlabel('NO(%)', labelpad=15, fontsize=16,weight='bold')

    ax.view_init(elev=35, azim=-45)  # 优化视角
    ax.zaxis.set_tick_params(
    labelsize=14,                     # 刻度字号
    labelcolor='black',             # 刻度颜色
    labelrotation= 0,                 # 刻度标签旋转角度（防重叠）
    pad=8                             # 刻度与轴线间距
        )
    # ========== 5. 保存与输出 ==========
    if save:
        os.makedirs(config['model_path'], exist_ok=True)
        plt.savefig(
            f"{config['model_path']}/{config['model_name']}.png",
            dpi=600, bbox_inches='tight'
        )
    plt.show()




def calculate_mre(model,x_train,x_test,y_train, y_test,):
    """
    计算平均相对误差 (MRE)

    参数:
    y_true (array-like): 真实值数组
    y_pred (array-like): 预测值数组

    返回:
    float: MRE 值
    """
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    train_pred = np.asarray(train_pred)
    test_pred = np.asarray(test_pred)


    # 计算相对误差并取平均
    relative_errors = np.abs(y_train - train_pred) / y_train
    mre_1 = np.mean(relative_errors)

    relative_errors = np.abs(y_test - test_pred) / y_test
    mre_2 = np.mean(relative_errors)
    return mre_1,mre_2