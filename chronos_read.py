import pickle
import datasets
import numpy as np
import pandas as pd

import os
import pickle

def load_all_datasets(directory):
    all_datasets_list = []

    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            
            # 读取pickle文件
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                
                # 检查sliding_windows是否在数据中
                # if 'sliding_windows' in data:
                for key in list(data.keys()):
                    all_datasets_list.extend(data[key])
    
    return all_datasets_list

# 使用示例
directory = 'datasets/chronos'
all_datasets_list = load_all_datasets(directory)
print(len(all_datasets_list))