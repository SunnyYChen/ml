import numpy as np
import pandas as pd

# df = [1,2,3,4]
# print(df[:-1])
# print(df[-1])


p = [0.1, 0.2, 0.3, 0.4]
y = [0.2, 0.4, 0.6, 0.8]
# mat = [p, y]
# mat = np.transpose(mat)
# df = pd.DataFrame(mat)
# print(df)
# data = df[[0, 1]]
# print(data)
# print(data.corr())


trade_date = np.arange(len(p))
print(trade_date)
print(p)
print(y)
# data = new_df[[new_df_columns[i], 'label']]