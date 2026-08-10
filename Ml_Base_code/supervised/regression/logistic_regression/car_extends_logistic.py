import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Đọc dữ liệu
df = pd.read_csv('exported-patient-data.csv')

# Loại bỏ cột Unnamed: 0 nếu có
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 2. Tách X (features) và y (target)
X = df.drop('target', axis=1)
y = df['target']

# 3. Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Chuẩn hóa dữ liệu (Quan trọng đối với Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Dự đoán và Đánh giá
y_pred = model.predict(X_test_scaled)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Trực quan hóa
plt.figure(figsize=(12, 5))

# Đồ thị 1: Biểu đồ mối quan hệ giữa Tuổi & Nhịp tim tối đa
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=df['age'], y=df['thalach'], hue=df['target'],
    palette={0: 'blue', 1: 'red'}, s=60, alpha=0.8
)
plt.title('Tuổi (age) vs Nhịp tim tối đa (thalach)')
plt.xlabel('Tuổi')
plt.ylabel('Nhịp tim tối đa')
plt.legend(title='Bệnh tim', labels=['Không (0)', 'Có (1)'])

# Đồ thị 2: Confusion Matrix
# plt.subplot(1, 2, 2)
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Khỏe mạnh (0)', 'Bệnh tim (1)'],
#             yticklabels=['Khỏe mạnh (0)', 'Bệnh tim (1)'])
# plt.title('Confusion Matrix')
# plt.xlabel('Dự đoán')
# plt.ylabel('Thực tế')

# plt.tight_layout()
# plt.show()