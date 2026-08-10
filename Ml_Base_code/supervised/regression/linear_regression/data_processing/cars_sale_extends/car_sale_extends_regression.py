import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

file_path = r'D:\data_system\Ml_Dl_book-source\Ml_Base_code\supervised\regression\linear_regression\data_processing\cars_sale_extends\car-sales-extended.csv'

df = pd.read_csv(file_path)

# Tách X và y
X = df.drop('Price', axis=1)
y = df['Price']

X_encoded = pd.get_dummies(X, columns=['Make', 'Colour'], drop_first=True, dtype=int)

#train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.2, 
    random_state=42
)

#model
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập Test
y_pred = model.predict(X_test)

# print
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}$")

# ==========================================
# 6. VẼ ĐỒ THỊ
# ==========================================
plt.figure(figsize=(12, 5))

# dthi 1: qhe 
plt.subplot(1, 2, 1)
sns.regplot(
    x=df['Odometer (KM)'], 
    y=df['Price'], 
    scatter_kws={'alpha': 0.3, 'color': 'blue'}, 
    line_kws={'color': 'red'}
)
plt.title('Đường Hồi quy: Odometer (KM) vs Price')
plt.xlabel('Odometer (KM)')
plt.ylabel('Price ($)')

#dthi 2: so sanh
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Dự đoán')

# Đường lý tưởng y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường lý tưởng (y = x)')

plt.title('Giá thực tế vs Giá dự đoán ')
plt.xlabel('Giá thực tế ($)')
plt.ylabel('Giá dự đoán ($)')
plt.legend()

plt.tight_layout()
plt.show()