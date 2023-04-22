import scipy.io
import pandas as pd
import numpy as np
data = scipy.io.loadmat("G:\毕业设计\数据集\速变数据\\traindata\\T082_common_channel_DATA.mat")
del data['__header__']
del data['__version__']
del data['__globals__']
del data['x_sequence']

file_handle=open('list.txt',mode='w')
keys = data.keys()
for key in keys:
    file_handle.write(key)
    file_handle.write('\n')

file_handle.close()

