#!/usr/bin/env python
# coding: utf-8

# ![](./img/dl_banner.jpg)

# # 新零售预估--连锁超市销量预估案例
# 
# 
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料
# **代码来源于kaggle比赛[rank 5th solution](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47556)**<br>
# 
# 代码整理by [@寒小阳](https://blog.csdn.net/han_xiaoyang)
# 
# ![](./img/favorita-grocery-sales-forecasting.png)
# 
# ### 背景
# 这是kaggle上的一个比赛：[连锁超市销量预估案例](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
# 
# 实体店中的采购和销售需要保持平衡。稍微将销售预测过量，店里就会积存许多货物，要是积压了不易储存的商品会更加糟糕。而要是将销量预测过低，设想一下，当顾客把钱都付了，却发现没货，那这家店的口碑将会急剧下降。
# 
# 此外，随着零售商增加新的门店，那里的顾客又可能存在独特的需求，比如他们想要新的产品，口味随季节而变化，那问题将变得更加复杂，产品市场如何真的很难预知。
# 
# 在这次比赛中，主办方希望参赛者建立一个更准确的预测产品销量的模型。作者的目标是建立一个可能应用于实际，并具有最高准确度的销量预测模型。
# 
# ### 数据集描述
# 这次比赛是预测商品销量，数据被分为两部分——训练数据和测试数据。训练数据用于模型训练，测试数据被分为几部分，分别用于在public和private的排行榜上进行模型的准确性评估。这场比赛中，Corporacion Favorita 提供 125,497,040 个训练观察值和 3370,464 个测试观察值。给出的训练数据为<单位销量，日期，商店ID，商品ID，推销活动标签>，其中单位销量是待预测值，基本上属于回归问题。同时给出的额外数据表有：
# 
# - “商店信息表”—<商店ID，所在城市，所在州，类型，聚类簇>
# - “商品表”—<商品ID，所属类别，所属子类别，易腐烂标签>
# - “交易信息表”—<商店ID，日期，总交易笔数>
# - “石油价格”—<日期、石油价格>
# - “节假日信息”
# 
# 值得注意的是，同一商品可能在不同的商店均有销售，而最后的测试数据为<日期，商店ID，商品ID，推销活动标签>，我们需要预测的是某商品在指定商店在某天的销量。训练数据给出了从2013年-2017年8月15日近一千余天的数据条目，每一天都包含各类商品在各个商店的数据条路，而很多商品只在某一些时间阶段出现过。测试数据需要预测的是2017年8月16日—31日销量情况。这就使得数据的组织十分重要。
# 
# ### 评估准则
# $$NWRMSLE = \sqrt{ \frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}_i + 1) - \ln(y_i +1)  \right)^2  }{\sum_{i=1}^n w_i}}$$

# ### 0.总体解法说明

# 这是一个很典型的时间序列预估的问题，这类问题常见的解决方案有`监督学习模型`和`时间序列模型`两种。
# 
# 其中`监督学习模型`思路是把所有能收集的影响最后销量的信息汇总成特征X，销量(有时候会做变换)作为y，构建监督学习模型，可以选用的监督学习算法很多，包括树模型(随机森林/GBDT/xgboost/LightGBM)和深度学习(多层感知器、wide&deep models)等。注意在时间序列问题中，对销量预估有用的信息除掉**静态**的信息(比如商品的品类、品牌、地理位置等)以外，还包括历史销量数据，所以我们通常会以时间滑窗的方式采集很多统计值和趋势信息也一起作为预估的辅助信息。
# 
# `时间序列模型`主要指的是基于历史销量值和变化趋势等构建模型进行预估，包括Arima，facebook prophet等传统统计方法，以及LSTM等基于RNN的解决方案。
# 
# 下面总结一下这位同学解法的核心。
# - 关于模型
#     - 构建了3个模型：lightGBM，CNN + DNN和seq2seq RNN模型。最终模型是这些模型的加权平均值（其中每个模型通过使用不同的随机种子多次训练来稳定，然后取平均值），对促销信息做特殊处理。每个单模型都可以在私人排行榜中保持最高1％(这个挺了不起的)。
#     - LGBM：常规的boosting模型，效果来源于特征工程。
#     - CNN + DNN：这是传统的NN模型，其中CNN部分是受WaveNet启发的扩张因果卷积(dilated causal convolution)网络，DNN部分是连接到原始销售序列的2个FC层。然后将输入与分类嵌入和未来的促销拼接在一起，并直接输出到未来16天的预测。
#     - RNN：这是一个seq2seq模型。编码器和解码器都是GRU。编码器的隐藏状态通过FC层连接器传递给解码器。对最后结果起到非常大的作用。
#     
# - 关于特征工程
#     - 对于LGB，对每个时间段统计包括平均销售额，促销数和“零”数等值。利用不同分割方式都计算了这些特征，分割方式包括 有/无促销，每个工作日，商品/商店组统计等。类别型的特征用label encoding处理。
#     - 对于NN，基于item的销售额平均值和一年前/四分之一前的取值作为序列输入模型。类别型特征和时间特征（工作日，日期）以嵌入(embedding)方式提供
# 
# - 关于训练集验证集构建
#     - 取比赛日期为2017.7.26~2017.8.10的数据为验证集，2017-1.1~2017.7.5期间随机抽样作为训练集。同时会在训练集中过滤出测试集中出现了的store-item对这部分数据。

# ## 工具函数构建

# In[ ]:


# 导入工具库
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc

# 加载数据
def load_data():
    df_train = pd.read_csv('train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': bool},
                           converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                           parse_dates=["date"])
    df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                          parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])

    # 切分数据子集
    df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,1,1)]

    # 促销
    promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train

    df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)

    # items
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")

    return df_2017, promo_2017, items, stores


def save_unstack(df, promo, filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df.columns = df.columns.astype('str')
    df.reset_index().to_feather(df_name)
    promo.columns = promo.columns.astype('str')
    promo.reset_index().to_feather(promo_name)

def load_unstack(filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df_2017 = pd.read_feather(df_name).set_index(['store_nbr','item_nbr'])
    df_2017.columns = pd.to_datetime(df_2017.columns)
    promo_2017 = pd.read_feather(promo_name).set_index(['store_nbr','item_nbr'])
    promo_2017.columns = pd.to_datetime(promo_2017.columns)
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")

    return df_2017, promo_2017, items, stores

# 构建验证集和测试集
def create_dataset(df, promo_df, items, stores, timesteps, first_pred_start, is_train=True, aux_as_tensor=False, reshape_output=0):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    return create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, first_pred_start, reshape_output, aux_as_tensor, is_train)

def train_generator(df, promo_df, items, stores, timesteps, first_pred_start,
    n_range=1, day_skip=7, is_train=True, batch_size=2000, aux_as_tensor=False, reshape_output=0, first_pred_start_2016=None):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()
    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    while 1:
        date_part = np.random.permutation(range(n_range))
        if first_pred_start_2016 is not None:
            range_diff = (first_pred_start - first_pred_start_2016).days / day_skip
            date_part = np.concat([date_part, np.random.permutation(range(range_diff, int(n_range/2) + range_diff))])

        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx,:]
            promo_df_tmp = promo_df.iloc[keep_idx,:]
            cat_features_tmp = cat_features[keep_idx]

            pred_start = first_pred_start - timedelta(days=int(day_skip*i))

            # Generate a batch of random subset data. All data in the same batch are in the same period.
            yield create_dataset_part(df_tmp, promo_df_tmp, cat_features_tmp, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, True)

            gc.collect()

def create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, is_train, weight=False):

    item_mean_df = item_group_mean.reindex(df.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df.index.get_level_values(0))
    # store_family_mean_df = store_family_group_mean.reindex(df.index)

    X, y = create_xy_span(df, pred_start, timesteps, is_train)
    is0 = (X==0).astype('uint8')
    promo = promo_df[pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)].values
    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    dom = np.tile([d.day-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)
    # store_family_mean, _ = create_xy_span(store_family_mean_df, pred_start, timesteps, False)
    # month_tmp = np.tile([d.month-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
    #                       (X_tmp.shape[0],1))
    yearAgo, _ = create_xy_span(df, pred_start-timedelta(days=365), timesteps+16, False)
    quarterAgo, _ = create_xy_span(df, pred_start-timedelta(days=91), timesteps+16, False)

    if reshape_output>0:
        X = X.reshape(-1, timesteps, 1)
    if reshape_output>1:
        is0 = is0.reshape(-1, timesteps, 1)
        promo = promo.reshape(-1, timesteps+16, 1)
        yearAgo = yearAgo.reshape(-1, timesteps+16, 1)
        quarterAgo = quarterAgo.reshape(-1, timesteps+16, 1)
        item_mean = item_mean.reshape(-1, timesteps, 1)
        store_mean = store_mean.reshape(-1, timesteps, 1)
        # store_family_mean = store_family_mean.reshape(-1, timesteps, 1)

    w = (cat_features[:, 2] * 0.25 + 1) / (cat_features[:, 2] * 0.25 + 1).mean()

    cat_features = np.tile(cat_features[:, None, :], (1, timesteps+16, 1)) if aux_as_tensor else cat_features

    # Use when only 6th-16th days (private periods) are in the training output
    # if is_train: y = y[:, 5:]

    if weight: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y, w)
    else: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y)

def create_xy_span(df, pred_start, timesteps, is_train=True, shift_range=0):
    X = df[pd.date_range(pred_start-timedelta(days=timesteps), pred_start-timedelta(days=1))].values
    if is_train: y = df[pd.date_range(pred_start, periods=16)].values
    else: y = None
    return X, y

# Not used in the final model
def random_shift_slice(mat, start_col, timesteps, shift_range):
    shift = np.random.randint(shift_range+1, size=(mat.shape[0],1))
    shift_window = np.tile(shift,(1,timesteps)) + np.tile(np.arange(start_col, start_col+timesteps),(mat.shape[0],1))
    rows = np.arange(mat.shape[0])
    rows = rows[:,None]
    columns = shift_window
    return mat[rows, columns]

# Calculate RMSE scores for all 16 days, first 5 days (fror public LB) and 6th-16th days (for private LB) 
def cal_score(Ytrue, Yfit):
	print([metrics.mean_squared_error(Ytrue, Yfit), 
	metrics.mean_squared_error(Ytrue[:,:5], Yfit[:,:5]),
	metrics.mean_squared_error(Ytrue[:,5:], Yfit[:,5:])])

# Create submission file
def make_submission(df_index, test_pred, filename):
	df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
	df_preds = pd.DataFrame(
	    test_pred, index=df_index,
	    columns=pd.date_range("2017-08-16", periods=16)
	).stack().to_frame("unit_sales")
	df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

	submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
	submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
	submission.to_csv(filename, float_format='%.4f', index=None)


# ## LightGBM建模

# In[ ]:


from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc

from Utils import *

df_2017, promo_2017, items = load_unstack('1617')

promo_2017 = promo_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df_2017 = df_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_2017 = promo_2017.astype('int')
df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df_2017.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

df_2017 = df_2017.loc[df_2017.index.get_level_values(1).isin(item_inter)]
promo_2017 = promo_2017.loc[promo_2017.index.get_level_values(1).isin(item_inter)]


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True, one_hot=False):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "day_2_2017": get_timespan(df_2017, t2017, 2, 1).values.ravel(),
        "day_3_2017": get_timespan(df_2017, t2017, 3, 1).values.ravel(),
#         "day_4_2017": get_timespan(df_2017, t2017, 4, 1).values.ravel(),
#         "day_5_2017": get_timespan(df_2017, t2017, 5, 1).values.ravel(),
#         "day_6_2017": get_timespan(df_2017, t2017, 6, 1).values.ravel(),
#         "day_7_2017": get_timespan(df_2017, t2017, 7, 1).values.ravel(),
#         "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
#         "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
#         "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
#         "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
#         "median_30_2017": get_timespan(df_2017, t2017, 30, 30).median(axis=1).values,
#         "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
        'promo_3_2017': get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "last_year_mean": get_timespan(df_2017, t2017, 365, 16).mean(axis=1).values,
        "last_year_count0": (get_timespan(df_2017, t2017, 365, 16)==0).sum(axis=1).values,
        "last_year_promo": get_timespan(promo_2017, t2017, 365, 16).sum(axis=1).values
    })
    
    for i in [7, 14, 21, 30, 60, 90, 140, 365]:
        X['mean_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['median_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['max_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).max(axis=1).values
        X['mean_{}_haspromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==1].mean(axis=1).values
        X['mean_{}_nopromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==0].mean(axis=1).values
        X['count0_{}_2017'.format(i)] = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).values
        X['promo_{}_2017'.format(i)] = get_timespan(promo_2017, t2017, i, i).sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_{}_mean'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        item_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('item_nbr').mean().to_frame('item_count0')
        X['item_{}_count0_mean'.format(i)] = df_2017.join(item_count0)['item_count0'].values
        store_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('store_nbr').mean().to_frame('store_mean')
        X['store_{}_mean'.format(i)] = df_2017.join(store_mean)['store_mean'].values
        store_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('store_nbr').mean().to_frame('store_count0')
        X['store_{}_count0_mean'.format(i)] = df_2017.join(store_count0)['store_count0'].values
        
    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_10_dow{}'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['count0_10_dow{}'.format(i)] = (get_timespan(df_2017, t2017, 70-i, 10)==0).sum(axis=1).values
        X['promo_10_dow{}'.format(i)] = get_timespan(promo_2017, t2017, 70-i, 10, freq='7D').sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_mean_10_dow{}'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        X['mean_20_dow{}'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values
    
    if one_hot:
        family_dummy = pd.get_dummies(df_2017.join(items)['family'], prefix='family')
        X = pd.concat([X, family_dummy.reset_index(drop=True)], axis=1)
        store_dummy = pd.get_dummies(df_2017.reset_index().store_nbr, prefix='store')
        X = pd.concat([X, store_dummy.reset_index(drop=True)], axis=1)
#         X['family_count'] = df_2017.join(items).groupby('family').count().iloc[:,0].values
#         X['store_count'] = df_2017.reset_index().groupby('family').count().iloc[:,0].values
    else:
        df_items = df_2017.join(items)
        df_stores = df_2017.join(stores)
        X['family'] = df_items['family'].astype('category').cat.codes.values
        X['perish'] = df_items['perishable'].values
        X['item_class'] = df_items['class'].values
        X['store_nbr'] = df_2017.reset_index().store_nbr.values
        X['store_cluster'] = df_stores['cluster'].values
        X['store_type'] = df_stores['type'].astype('category').cat.codes.values
#     X['item_nbr'] = df_2017.reset_index().item_nbr.values
#     X['item_mean'] = df_2017.join(item_mean)['item_mean']
#     X['store_mean'] = df_2017.join(store_mean)['store_mean']

#     store_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('store_nbr').mean().to_frame('store_promo_90_mean')
#     X['store_promo_90_mean'] = df_2017.join(store_promo_90_mean)['store_promo_90_mean'].values
#     item_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('item_nbr').mean().to_frame('item_promo_90_mean')
#     X['item_promo_90_mean'] = df_2017.join(item_promo_90_mean)['item_promo_90_mean'].values
    
    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y
    return X


print("Preparing dataset...")
X_l, y_l = [], []
t2017 = date(2017, 7, 5)
n_range = 14
for i in range(n_range):
    print(i, end='..')
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 - delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'max_bin':128,
    'num_threads': 8
}

print("Training and predicting models...")
MAX_ROUNDS = 700
val_pred = []
test_pred = []
# best_rounds = []
cate_vars = ['family', 'perish', 'store_nbr', 'store_cluster', 'store_type']
w = (X_val["perish"] * 0.25 + 1) / (X_val["perish"] * 0.25 + 1).mean()

for i in range(16):

    print("Step %d" % (i+1))

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=None)
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=w,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], verbose_eval=100)

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True)[:15]))
    best_rounds.append(bst.best_iteration or MAX_ROUNDS)

    val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    gc.collect();

cal_score(y_val, np.array(val_pred).T)

make_submission(df_2017, np.array(test_pred).T)


# ## CNN建模

# In[ ]:


import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2015,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2015,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

timesteps = 200

# preparing data
train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 5),
                                           n_range=16, day_skip=7, batch_size=2000, aux_as_tensor=False, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=False, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=False, is_train=False, reshape_output=2)

w = (Xval[7][:, 2] * 0.25 + 1) / (Xval[7][:, 2] * 0.25 + 1).mean() # validation weight: 1.25 if perishable and 1 otherwise per competition rules

del df, promo_df; gc.collect()

print('current no promo 2') # log info

latent_dim = 32

# Define input
# seq input
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))
weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(6,))
item_family = Lambda(lambda x: x[:, 0, None])(cat_features)
item_class = Lambda(lambda x: x[:, 1, None])(cat_features)
item_perish = Lambda(lambda x: x[:, 2, None])(cat_features)
store_nbr = Lambda(lambda x: x[:, 3, None])(cat_features)
store_cluster = Lambda(lambda x: x[:, 4, None])(cat_features)
store_type = Lambda(lambda x: x[:, 5, None])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=1)(item_family)
# class_embed = Embedding(337, 8, input_length=1)(item_class)
store_embed = Embedding(54, 8, input_length=1)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=1)(store_cluster)
type_embed = Embedding(5, 2, input_length=1)(store_type)

encode_slice = Lambda(lambda x: x[:, :timesteps, :])
# encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in], axis=2)
# encode_features = encode_slice(encode_features)

x_in = concatenate([seq_in, encode_slice(promo_in), item_mean_in], axis=2)

# Define network
# c0 = TimeDistributed(Dense(4))(x_in)
# # c0 = Conv1D(4, 1, activation='relu')(sequence_in)
c1 = Conv1D(latent_dim, 2, dilation_rate=1, padding='causal', activation='relu')(x_in)
c2 = Conv1D(latent_dim, 2, dilation_rate=2, padding='causal', activation='relu')(c1)
c2 = Conv1D(latent_dim, 2, dilation_rate=4, padding='causal', activation='relu')(c2)
c2 = Conv1D(latent_dim, 2, dilation_rate=8, padding='causal', activation='relu')(c2)
# c2 = Conv1D(latent_dim, 2, dilation_rate=16, padding='causal', activation='relu')(c2)

c4 = concatenate([c1, c2])
# c2 = MaxPooling1D()(c2)

conv_out = Conv1D(8, 1, activation='relu')(c4)
# conv_out = GlobalAveragePooling1D()(c4)
conv_out = Dropout(0.25)(conv_out)
conv_out = Flatten()(conv_out)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
promo_pred = decode_slice(promo_in)
# qAgo_pred = decode_slice(quarterAgo_in)
# yAgo_pred = decode_slice(yearAgo_in)


# Raw sequence in results overfitting!!!
dnn_out = Dense(512, activation='relu')(Flatten()(seq_in))
dnn_out = Dense(256, activation='relu')(dnn_out)
# dnn_out = BatchNormalization()(dnn_out)
dnn_out = Dropout(0.25)(dnn_out)

x = concatenate([conv_out, dnn_out,
                 Flatten()(promo_pred), Flatten()(family_embed), Flatten()(store_embed), Flatten()(cluster_embed), Flatten()(type_embed), item_perish])
# x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# x = Dense(256, activation='relu')(x)
# x = BatchNormalization()(x)
# x = concatenate([x, seq_in])
output = Dense(16, activation='relu')(x)

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], output)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1000, workers=4, use_multiprocessing=True, epochs=10, verbose=2,
                    validation_data=(Xval, Yval, w))
                    

val_pred = model.predict(Xval)
cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'cnn_no-promo2.csv')
# gc.collect()

# model.save('save_models/cnn_model')


# ## seq2seq建模

# In[ ]:


import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

timesteps = 365

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2014,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2014,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 9),
                                           n_range=380, day_skip=1, batch_size=1000, aux_as_tensor=True, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=True, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=True, is_train=False, reshape_output=2)

w = (Xval[7][:, 0, 2] * 0.25 + 1) / (Xval[7][:, 0, 2] * 0.25 + 1).mean()

del df, promo_df; gc.collect()

# Note
# current best: add item_mean, dim: 50, all as tensor ~ 3500 (~3630 in new cv)
print('1*100, train on private 7, nrange 380 timestep 200, data 1000*1500 \n')

latent_dim = 100

# seq input
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))
weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(timesteps+16, 6))
item_family = Lambda(lambda x: x[:, :, 0])(cat_features)
item_class = Lambda(lambda x: x[:, :, 1])(cat_features)
item_perish = Lambda(lambda x: x[:, :, 2])(cat_features)
store_nbr = Lambda(lambda x: x[:, :, 3])(cat_features)
store_cluster = Lambda(lambda x: x[:, :, 4])(cat_features)
store_type = Lambda(lambda x: x[:, :, 5])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=timesteps+16)(item_family)
class_embed = Embedding(337, 8, input_length=timesteps+16)(item_class)
store_embed = Embedding(54, 8, input_length=timesteps+16)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=timesteps+16)(store_cluster)
type_embed = Embedding(5, 2, input_length=timesteps+16)(store_type)

# Encoder
encode_slice = Lambda(lambda x: x[:, :timesteps, :])
encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
encode_features = encode_slice(encode_features)

# conv_in = Conv1D(8, 5, padding='same')(concatenate([seq_in, encode_features], axis=2))
# conv_raw = concatenate([seq_in, encode_slice(quarterAgo_in), encode_slice(yearAgo_in), item_mean_in], axis=2)
# conv_in = Conv1D(8, 5, padding='same')(conv_raw)
conv_in = Conv1D(4, 5, padding='same')(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=1)(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=2)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=4)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=8)(conv_in_deep)
# conv_in_quarter = Conv1D(4, 5, padding='same')(encode_slice(quarterAgo_in))
# conv_in_year = Conv1D(4, 5, padding='same')(encode_slice(yearAgo_in))
# conv_in = concatenate([conv_in_seq, conv_in_deep, conv_in_quarter, conv_in_year])

x_encode = concatenate([seq_in, encode_features, conv_in, item_mean_in], axis=2)
                        # store_mean_in, is0_in, store_family_mean_in], axis=2)
# encoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# encoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# encoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
encoder = CuDNNGRU(latent_dim, return_state=True)
print('Input dimension:', x_encode.shape)
_, h= encoder(x_encode)
# s1, h1 = encoder1(x_encode)
# s1 = Dropout(0.25)(s1)
# s2, h2 = encoder2(s1)
# _, h3 = encoder3(s2)

# Connector
h = Dense(latent_dim, activation='tanh')(h)
# h1 = Dense(latent_dim, activation='tanh')(h1)
# h2 = Dense(latent_dim, activation='tanh')(h2)

# Decoder
previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
decode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
decode_features = decode_slice(decode_features)

# decode_idx_train = np.tile(np.arange(16), (Xtrain.shape[0], 1))
# decode_idx_val = np.tile(np.arange(16), (Xval.shape[0], 1))
# decode_idx = Input(shape=(16,))
# decode_id_embed = Embedding(16, 4, input_length=16)(decode_idx)
# decode_features = concatenate([decode_features, decode_id_embed])

# aux_features = concatenate([dom_embed_decode, store_embed_decode, family_embed_decode], axis=2)
# aux_features = decode_slice(aux_features)

# decoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# decoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
decoder = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder_dense1 = Dense(128, activation='relu')
decoder_dense2 = Dense(1, activation='relu')
# dp = Dropout(0.25)
slice_at_t = Lambda(lambda x: tf.slice(x, [0,i,0], [-1,1,-1]))
for i in range(16):
    previous_x = Reshape((1,1))(previous_x)
    
    features_t = slice_at_t(decode_features)
    # aux_t = slice_at_t(aux_features)

    decode_input = concatenate([previous_x, features_t], axis=2)
    # output_x, h1 = decoder1(decode_input, initial_state=h1)
    # output_x = dp(output_x)
    # output_x, h2 = decoder2(output_x, initial_state=h2)
    # output_x, h3 = decoder3(output_x, initial_state=h3)
    output_x, h = decoder(decode_input, initial_state=h)
    # aux input
    # output_x = concatenate([output_x, aux_t], axis=2)
    # output_x = Flatten()(output_x)
    # decoder_dense1 = Dense(64, activation='relu')
    # output_x = decoder_dense1(output_x)
    # output_x = dp(output_x)
    output_x = decoder_dense2(output_x)

    # gather outputs
    if i == 0: decoder_outputs = output_x
    elif i > 0: decoder_outputs = concatenate([decoder_outputs, output_x])

    previous_x = output_x

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], decoder_outputs)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1500, workers=5, use_multiprocessing=True, epochs=18, verbose=2,
                    validation_data=(Xval, Yval, w))

# val_pred = model.predict(Xval)
# cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'seq-private_only-7.csv')

# model.save('save_models/seq2seq_model-withput-promo-2')


# ![](./img/xiniu_neteasy.png)
