import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Nhập giá trị kỳ vọng (mean) và độ lệch chuẩn (standard deviation) từ người dùng
mu = float(input("Nhập giá trị kỳ vọng (mean): "))
sigma = float(input("Nhập độ lệch chuẩn (standard deviation): "))

# Tạo dãy số x từ -5 đến 5 với khoảng cách 0.1
x = np.arange(-5, 5, 0.1)

# Tính giá trị mật độ xác suất tại các điểm x
pdf_values = norm.pdf(x, mu, sigma)

# Vẽ đồ thị hàm mật độ xác suất
plt.plot(x, pdf_values, label='Phân phối chuẩn')
plt.title(f'Hàm Mật độ Xác suất của Phân phối Chuẩn\n(mu={mu}, sigma={sigma})')
plt.xlabel('Giá trị')
plt.ylabel('Mật độ xác suất')
plt.legend()
plt.show()
