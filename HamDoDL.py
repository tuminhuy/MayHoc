import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

dulieuload = pd.read_csv("winequality-red.csv",delimiter=';')
print(dulieuload)
dulieu_x = dulieuload.iloc[:,0:-1]
dulieu_y = dulieuload.iloc[:,-1]
#tham số
print("số lượng phần tử",len(dulieu_x))
#đếm số lượng Y
DanhSachY_PhanBiet = np.unique(dulieu_y)
print("DS Y",DanhSachY_PhanBiet)
print("SL Y: ", len(DanhSachY_PhanBiet))
print("SL của mỗi lớp: ", dulieu_y.value_counts())

#chia tập dữ liệu 
X_train, X_Test, Y_Train, Y_Test =  train_test_split(dulieu_x,dulieu_y, test_size=1/4.0, random_state=10)
print("SL Train: ",len(X_train))
print("SL Test: ",len(X_Test))
