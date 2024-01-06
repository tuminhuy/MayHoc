import pandas as pd 
#--------------load du lieu va chia du lieu---------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#------------------KNN-----------------
from sklearn.neighbors import KNeighborsClassifier

#----------------- danh gia mo hinh - do chinh xac
from sklearn.metrics import accuracy_score

#----------------------Confunsion Matrix---
from sklearn.metrics import confusion_matrix

#---------------Bayes------------
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#------------DecisionTree----------
from sklearn.tree import DecisionTreeClassifier

#---------------danh gia nghi thuc K-Fold
from sklearn.model_selection import KFold

#doc du lieu
dulieuload = pd.read_csv("winequality-red.csv",delimiter=';')
print(dulieuload)
dulieu_x = dulieuload.iloc[:,0:-1]
dulieu_y = dulieuload.iloc[:,-1]

# chia tap du lieu - danh gia nghi thu k-fold
kf = KFold(n_splits=15)
for idTrain, idTest in kf.split(dulieuload):
    X_Train = dulieu_x.iloc[idTrain, ]
    X_Test = dulieu_x.iloc[idTest, ]
    Y_Train = dulieu_y.iloc[idTrain, ]
    Y_Test = dulieu_y.iloc[idTest, ]
    print ("SL Train",len(X_Train))
    print ("SL Test",len(X_Test))

   

#load mo hinh
    #MoHinhDT = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
   # MoHinhDT.fit(X_Train)
    

#Du doan mo hinh
    #Y_Dudoan = MoHinhDT.predict(X_Test)
  #  print("KQ du doan: ", Y_Dudoan)
    
#MHKNN = KNeighborsClassifier(n_neighbors=5)
#MHKNN.fit(X_Test)
    
    from sklearn.neighbors import KNeighborsClassifier

# Tạo mô hình KNN
MoHinhKNN = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình trên tập dữ liệu đào tạo
MoHinhKNN.fit(X_Train, Y_Train)
