import numpy as np # chuyển loại dữ liệu
import pandas as pd # thư viện pandas

#tải dữ liệu và chia dữ liệu
from sklearn.datasets import load_iris
from sklearn.datasets import load_files
# hàm để chua dữ liệu thành 2 phần train và test
from sklearn.model_selection import train_test_split

#KNN
from sklearn.neighbors import KNeighborsClassifier
#đánh giá độ chính xác
from sklearn.metrics import accuracy_score
#ma trận Confusion
from sklearn.metrics import confusion_matrix
#đánh giá phân lớp prec, rec, f1, acc
from sklearn.metrics import precision_recall_fscore_support
#Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#--DecisionTree---
from sklearn.tree import DecisionTreeClassifier
#----đánh giá nghi thức K-Fold--
from sklearn.model_selection import KFold

#đọc dữ liệu từ thư viện

dl = load_iris()
dulieu_x = dl.data[0:] # thuộc tính tập iris
dulieu_y = dl.target[0:] #class
print("X = ",dulieu_x)
print("Y = ",dulieu_y) 

'''
#đọc dữ liệu từ file có sẵn
dl2 = pd.read_csv("winequality-red.csv",delimiter=';')
dulieu_x = dl2.iloc[:,0:-1]
dulieu_y = dl2.iloc[:,-1]
print("X= ",dulieu_x )
print("Y= ",dulieu_y)
'''


#x_Train, x_Test, y_Train, y_Test = train_test_split(dulieu_x,dulieu_y, test_size=1/3.0, random_state=30, shuffle=True )

x_Train, x_Test, y_Train, y_Test = train_test_split(dulieu_x,dulieu_y, test_size=0.25, random_state=30, shuffle=True )

print(len(x_Train))
print(len(x_Test))

K=4

#Huấn luyện mô hình
model = KNeighborsClassifier(n_neighbors=K)
model.fit(x_Train,y_Train)
#model.fit(x_Train.values, y_Train.values)
#model.fit(x_Train.to_numpy(), y_Train.to_numpy())


Y_DuDoan = model.predict(x_Test)
print(Y_DuDoan, "Dự đoán: ")
print(y_Test, "Real Thực tế:")

DoChinhXac = accuracy_score(Y_DuDoan, y_Test)*100
print("Độ chính xác:", DoChinhXac)

MaTranKetQua =  confusion_matrix(Y_DuDoan, y_Test)
print (MaTranKetQua)

prec, rec, f1, sup = precision_recall_fscore_support(Y_DuDoan, y_Test)
print("precision", prec)
print("Recall", rec)
print("F1", f1)
print("Support", sup)

from sklearn.ensemble import RandomForestClassifier
modelRungNgauNhien =  RandomForestClassifier(n_estimators=111, random_state=30)
modelRungNgauNhien.fit(x_Train, y_Train)

Y_DuDoan = modelRungNgauNhien.predict(x_Test)
DoChinhXac = accuracy_score(Y_DuDoan, y_Test)*100
print("Độ chính xác rừng ngãu nhiên: ",DoChinhXac)

#Decision Tree
modelDecisionTree = DecisionTreeClassifier(random_state=70)
modelDecisionTree.fit(x_Train, y_Train)
Y_DuDoan_DT = modelDecisionTree.predict(x_Test)
DoChinhXac_DT = accuracy_score(Y_DuDoan_DT, y_Test)*100
print("Độ chính xác Decision Tree: ", DoChinhXac_DT)