import pandas as pd
import numpy as np 

dates = pd.date_range('20181024',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['A','B','C','D'])
print(df)

# 修改特定元素
# df.iloc[2,2] = 111 # 使用idex来修改
# df.loc['2018-10-25','A'] = 1234 # 使用label来修改
# df[df.A > 0] = 0 # 使用Boolean的方式修改
# df.A[df.A > 0] = 0 # 使用Boolean的方式修改 只修改A这一列
# print(df)

# 添加新的列
df['F'] = np.nan
# 利用index对齐
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20181024',periods=6))
print(df)

