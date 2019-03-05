#!/usr/bin/env python
# coding: utf-8

# ![](./img/dl_banner.jpg)

# # 时间序列预测与facebook prophet
# **[facebook prophet](https://facebook.github.io/prophet/)**
# **[github地址](https://github.com/facebook/prophet)**
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)
# 
# 
# ## 1.时间序列问题与prophet
# ![](./img/prophet.png)
# 时间序列是我们日常生活中最常见的数据类型之一。金融产品价格、天气、家庭能源使用量、甚至体重都具有变化规律。几乎每个数据科学家都会在日常工作中遇到时间序列，学习如何对时间序列进行建模是一项重要的数据科学技能。用于分析和预测周期性数据时，一种强大而简单的方法是加法模型（additive model）。这个想法很简单：将时间序列表示为每日、每周、每季度、每年度和节假日等不同时间维度的组合模式，并加以整体趋势。你的能源使用量可能会在夏天上升，在冬天下降，但是随着你家庭能源使用效率的提高，能源使用量总体呈下降趋势。加法模型可以向我们展示数据的模式/趋势，并根据这些观察结果进行预测。
# 
# Prophet是FaceBook开源的时序框架。非常简单实用，你不需要理解复杂的公式，看图，调参，调用十几行代码即可完成从数据输入到分析的全部工作，可谓懒人之利器。Prophet曾在多个典型时间序列预测数据科学比赛中取得很好的结果。
# 
# Prophet的原理是分析各种时间序列特征：周期性、趋势性、节假日效应，以及部分异常值。在趋势方面，它支持加入变化点，实现分段线性拟合。在周期方面，它使用傅里叶级数（Fourier series）来建立周期模型(sin+cos)，在节假和突发事件方面，用户可以通过表的方式指定节假日，及其前后相关的N天。可将Prophet视为一种针对时序的集成解决方案。
# ![](./img/additive_model.png)
#  使用Prophet具体使用步骤就是：根据格式要求填入训练数据，节假日数据，指定要预测的时段，然后训练即可。除了预测具体数值，Prophet还将预测结果拆分成trend, year, season, week等成份，并提供了各成份预测区间的上下边界。不仅是预测工具，也是一个很好的统计分析工具。
#  
#  当然Prophet也有它的弱项，比如可调节的参数不多，不支持与其时序特征结合等等，不过这些也可以通过预测处理和模型融合来解决。

# ## 2 安装方式
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳]
# 
# 在云平台可通过以下命令安装prophet：
# ```
# !pip install fbprophet
# ```
# 
# 可以通过以下命令下载源码（下面例程中用到了源码中的数据，请先下载源码）
# ```
# !git clone https://github.com/facebookincubator/prophet.git
# ```

# In[4]:


get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pystan fbprophet')


# In[5]:


get_ipython().system('git clone https://github.com/facebookincubator/prophet.git')


# ## 3 使用方法示例
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang/article/details/84205701)
# 
# 这里我们使用pandas DataFrame组织我们的时间序列数据，引入prophet工具库，并将我们数据中的列重新命名为正确的格式。日期列必须被称为「ds」，数值列被称为「y」。创建 prophet 模型后传入数据训练拟合，就像 Scikit-Learn 机器学习模型一样。

# ### 3.1导入工具库

# In[1]:


import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 3.2 读入数据

# In[2]:


# 读取时间数据文件，文件也在源码目录中
file_path= 'prophet/examples/example_wp_log_peyton_manning.csv'  
df = pd.read_csv(file_path)  


# In[11]:


df.head(10)


# ### 3.3 参数设定与附加信息输入

# In[12]:


# 输入节假日数据，注意lower_window, upper_window是前后影响天数
playoffs = pd.DataFrame({  
  'holiday': 'playoff',  
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',  
                        '2010-01-24', '2010-02-07', '2011-01-08',  
                        '2013-01-12', '2014-01-12', '2014-01-19',  
                        '2014-02-02', '2015-01-11', '2016-01-17',  
                        '2016-01-24', '2016-02-07']),  
  'lower_window': 0,  
  'upper_window': 1,  
})


# In[ ]:


superbowls = pd.DataFrame({  
  'holiday': 'superbowl',  
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),  
  'lower_window': 0,  
  'upper_window': 1,  
})  
holidays = pd.concat((playoffs, superbowls)) 


# ### 3.4 训练与预测

# In[ ]:


# prophet训练和预测
prophet = Prophet(daily_seasonality=True) 
df['y'] = np.log(df['y'])  
prophet.fit(df)  
future = prophet.make_future_dataframe(freq='D',periods=10)  # 测试之后十天
forecasts = prophet.predict(future)  


# In[15]:


forecasts.head()


# ### 3.5 结果可视化

# In[13]:


# 训练结果作图
prophet.plot(forecasts).show()  
prophet.plot_components(forecasts).show()  
plt.show()


# ### 版权归 © 稀牛学院 所有 保留所有权利
# ![](./img/xiniu_neteasy.png)
