import pandas as pd
import numpy as np 

dates = pd.date_range('20181024',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['A','B','C','D'])
print(df)
print(df.columns)
print(len(df.columns))
# 选择某一列
# print(df['A'],df.A)

# 选择第0,1行
# print(df[0:2])
# print(df['2018-10-24':'2018-10-25'])

# select by label: loc
# print(df.loc['2018-10-27'])
# print(df.loc[:,['A','B']]) # 保留所有行的内容，取特定列的数据
# print(df.loc['2018-10-27',['A','B']])  # 保留特定行的内容，取特定列的数据
    
# select by positionn: iloc
# print(df.iloc[3]) # 第三行的数据
# print(df.iloc[3,0]) # 第三行,第0列的数据
# print(df.iloc[3:5,0:3]) # 第3到5行,第0到2列的数据
# print(df.iloc[[0,2,-1],:]) # 第0,2,-15行,所有列数据

# mixed selection: ix
# print(df.ix[:3, ['A','C']]) # index 和 label 一起使用

# Boolean indexing
# print(df[df.A > 1]) # 这个可以哦
