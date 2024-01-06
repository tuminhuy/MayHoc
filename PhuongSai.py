import numpy as np

# Nhập dữ liệu từ người dùng
data_input = input("Nhập dữ liệu (cách nhau bởi dấu cách): ")
data = list(map(float, data_input.split()))

# Tính giá trị trung bình
mean_value = np.mean(data)

# Tính phương sai
variance = np.var(data)

# Tính độ lệch chuẩn
standard_deviation = np.std(data)

# Xuất kết quả
print(f"Giá trị trung bình: {mean_value}")
print(f"Độ lệch chuẩn: {standard_deviation}")
print(f"Phương sai: {variance}")

