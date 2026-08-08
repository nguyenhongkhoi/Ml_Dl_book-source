import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[45, 50, 65, 70, 85, 100, 115, 120]]).T
y = np.array([1.5, 1.8, 2.3, 2.5, 3.0, 3.5, 4.0, 4.2])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w_0 = w[0]
w_1 = w[1]

print("--- KẾT QUẢ TỪ NUMPY ---")
print(f"Phương trình tìm được: y = {w_1:.4f} * x + {w_0:.4f}")

area_new = 80
price_predicted = w_1 * area_new + w_0
print(f"Giá dự đoán cho căn hộ {area_new} m2 là: {price_predicted:.2f} tỷ VNĐ")

plt.scatter(X, y, color='red', label='Dữ liệu thực tế')

x_line = np.array([40, 130])
y_line = w_1 * x_line + w_0
plt.plot(x_line, y_line, color='blue', label='Đường hồi quy tìm được')

plt.title('Dự đoán giá nhà theo diện tích')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (Tỷ VNĐ)')
plt.legend()
plt.grid(True)
plt.show()