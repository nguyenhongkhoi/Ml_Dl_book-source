import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

file_path = r'D:\data_system\Ml_Dl_book-source\Ml_Base_code\supervised\KNN\car_sale_extends\car-sales-extended.csv'
df = pd.read_csv(file_path)

# Tách biến độc lập (X) và biến mục tiêu (y)
X = df.drop('Price', axis=1)
y = df['Price']


# Mã hóa One-Hot cho Make và Colour
X_encoded = pd.get_dummies(X, columns=['Make', 'Colour'], drop_first=True, dtype=int)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

#chuan hoa du lieu(do knn dung khoang cach euclid)
scaler = StandardScaler()
# Fit và transform trên tập Train
X_train_scaled = scaler.fit_transform(X_train)
# Chỉ transform trên tập Test (dùng lại thông số từ Train)kk
X_test_scaled = scaler.transform(X_test)

#k = 5
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Dự đoán
y_pred = knn_model.predict(X_test_scaled)

# Đánh giá chỉ số
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH KNN (K=5) ===")
print(f"R2 Score: {r2:.4f}")
print(f"MAE:      {mae:.2f} USD")

#so sanh gia tri thuc te va ly thuyet
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Đường hoàn hảo (y=x)')
plt.xlabel('Giá thực tế (Actual Price)')
plt.ylabel('Giá dự đoán (Predicted Price)')
plt.title('So sánh Giá thực tế vs Giá dự đoán')
plt.legend()
plt.grid(True)
plt.show()

#thu nghiem tu 1->21
k_values = range(1, 21)
mae_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    mae_scores.append(mean_absolute_error(y_test, preds))

plt.figure(figsize=(8, 4))
plt.plot(k_values, mae_scores, marker='o', color='green', linestyle='--')
plt.xlabel('Giá trị K')
plt.ylabel('Chỉ số MAE ')
plt.title('Đồ thị chọn giá trị K tối ưu')
plt.grid(True)
plt.show()