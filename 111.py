import numpy as np
import pandas as pd
from datetime import date

#  数据提取
f = pd.read_csv(r'xgb_preds.csv',header=0)

f.loc[:0] = f.loc[:0].astype(int)
f.loc[:1] = f.loc[:1].astype(int)
f.loc[:2] = f.loc[:2].astype(int)
f.to_csv("xgb_preds.csv",header = 0)