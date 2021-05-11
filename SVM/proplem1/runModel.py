import pandas as pd
from Model import  Model
data = pd.read_csv('../data.txt', sep=" ", header=None)

cl = Model(data, C=1000)
print(cl.get_result())
