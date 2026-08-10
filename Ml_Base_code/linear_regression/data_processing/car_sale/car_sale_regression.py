import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đường dẫn file gốc
file_path = r'D:\data_system\Ml_Dl_book-source\Ml_Base_code\data_processing\car_sale\car-sales.csv'

# 2. Đọc và tiền xử lý dữ liệu
df = pd.read_csv(file_path)
df['Price'] = df['Price'].str.replace(r'[\$,]', '', regex=True).astype(float)

X = df.drop('Price', axis=1)
y = df['Price']

# Mã hóa dữ liệu phân loại
X_encoded = pd.get_dummies(X, columns=['Make', 'Colour'], drop_first=True, dtype=int)

# 3. Chia tập dữ liệu (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.2, 
    random_state=42
)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH REGRESSION
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# ==========================================
# 5. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE):  {mse:.2f}")
print(f"R2 Score:                  {r2:.2f}\n")

# In so sánh Giá thực tế và Giá dự đoán
comparison_df = pd.DataFrame({'Giá thực tế': y_test.values, 'Giá dự đoán': y_pred})
print("So sánh chi tiết trên tập Test:")
print(comparison_df)

# ==========================================
# 6. VẼ ĐỒ THỊ
# ==========================================
plt.figure(figsize=(12, 5))

# Đồ thị 1: Mối quan hệ giữa Số KM đã đi (Odometer) và Giá xe (Price)
plt.subplot(1, 2, 1)
sns.regplot(x=df['Odometer (KM)'], y=df['Price'], color='blue', line_kws={'color': 'red'})
plt.title('Đường Hồi quy: Odometer vs Price')
plt.xlabel('Odometer (KM)')
plt.ylabel('Price ($)')

# Đồ thị 2: So sánh Giá thực tế vs Giá dự đoán
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='blue', s=80, label='Dự đoán')
# Vẽ đường tham chiếu y = x (nếu dự đoán chuẩn 100% thì các điểm sẽ nằm trên đường này)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Đường lý tưởng (y = x)')

plt.title('Giá thực tế vs Giá dự đoán')
plt.xlabel('Giá thực tế ($)')
plt.ylabel('Giá dự đoán ($)')
plt.legend()

plt.tight_layout()
plt.show()