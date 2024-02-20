import pandas as pd # thư viện pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Đọc dữ liệu từ URL trên GitHub
url = 'https://github.com/tuminhuy/MayHoc/raw/main/CarEvaluation1.xlsx'
dl = pd.read_excel(url)

# Chuyển đổi các giá trị chuỗi thành các giá trị số
encoder = LabelEncoder()
dl_encoded = dl.apply(encoder.fit_transform)

# Chia dữ liệu thành features (X) và target (y)
dulieu_x = dl_encoded.iloc[:, :-1]
dulieu_y = dl_encoded.iloc[:, -1]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_Train, x_Test, y_Train, y_Test = train_test_split(dulieu_x, dulieu_y, test_size=1/3.0, random_state=30, shuffle=True)

# Khởi tạo mô hình Random Forest
model = RandomForestClassifier()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_Train, y_Train)

# Dự đoán nhãn của tập kiểm tra
y_pred = model.predict(x_Test)

# Đánh giá accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Đánh giá mô hình Random Forest ", accuracy)

# Khởi tạo mô hình Decision Tree
model = DecisionTreeClassifier()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_Train, y_Train)

# Dự đoán nhãn của tập kiểm tra
y_pred = model.predict(x_Test)

# Đánh giá accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Đánh giá mô hình Decision Tree :", accuracy)

# Khởi tạo mô hình K-NN với số lân cận là 5
model = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_Train, y_Train)

# Dự đoán nhãn của tập kiểm tra
y_pred = model.predict(x_Test)

# Đánh giá accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Đánh giá mô hình K-NN ", accuracy)

# Khởi tạo mô hình Naive Bayes với Gaussian distribution
model = GaussianNB()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_Train, y_Train)

# Dự đoán nhãn của tập kiểm tra
y_pred = model.predict(x_Test)

# Đánh giá accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Đánh giá mô hình Naive Bayes:", accuracy)

# Khởi tạo mô hình K-NN với số lân cận là 5
model = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_Train, y_Train)

# Dự đoán nhãn của tập kiểm tra
y_pred = model.predict(x_Test)

# Đánh giá accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Accuracy:", accuracy)

# Đánh giá F1-score
f1 = f1_score(y_Test, y_pred, average='weighted')
print("F1 Score:", f1)

# Đánh giá precision
precision = precision_score(y_Test, y_pred, average='weighted')
print("Precision:", precision)

# Đánh giá recall
recall = recall_score(y_Test, y_pred, average='weighted')
print("Recall:", recall)