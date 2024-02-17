import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file Excel
#df = pd.read_excel("https://github.com/tuminhuy/MayHoc/raw/main/CarEvaluation1.xlsx")

# Đường dẫn tới tập tin Excel trong thư mục của bạn
file_path = 'D:\DH-CT\CT312-KhaiKhoanDuLieu\DoAn\MayHoc\CarEvaluation1.xlsx'

# Đọc tập tin Excel bằng Pandas
data = pd.read_excel(file_path)

dulieu_x = data.iloc[:,0:-1]
dulieu_y = data.iloc[:,-1]
print("X= ",dulieu_x )
print("Y= ",dulieu_y)

# Lấy danh sách tên các cột thuộc tính
column_names = data.columns.tolist()

# Tính toán số lượng hàng và cột cho lưới subplot
num_cols = 3
num_rows = (len(column_names) + num_cols - 1) // num_cols

# Đặt kích thước của đồ thị
plt.figure(figsize=(15, 10))

# Duyệt qua từng cột và vẽ biểu đồ countplot
for i, column in enumerate(column_names):
    plt.subplot(num_rows, num_cols, i+1)  # Tạo subplot cho mỗi cột
    sns.countplot(data=data, x=column)  # Trực quan hóa số lượng mẫu trong từng nhóm của cột
    plt.title(f'Số lượng mẫu của {column}')  # Đặt tiêu đề cho biểu đồ

plt.tight_layout()  # Tự động điều chỉnh layout
plt.show()  # Hiển thị đồ thị
