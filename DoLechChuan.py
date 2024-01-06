from scipy.stats import norm


# Nhập giá trị x từ người dùng
x_value = float(input("Nhập giá trị x: "))
9
# Nhập giá trị kỳ vọng (mean) và độ lệch chuẩn (standard deviation) từ người dùng
mu = float(input("Nhập giá trị kỳ vọng (mean): "))
sigma = float(input("Nhập độ lệch chuẩn (standard deviation): "))


# Tính giá trị mật độ xác suất tại giá trị x
pdf_value = norm.pdf(x_value, mu, sigma)

# Xuất giá trị mật độ xác suất
print(f"Giá trị mật độ xác suất tại {x_value} là: {pdf_value}")
