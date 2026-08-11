import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('exported-patient-data.csv')

if 'Unnamed: 0' in df.columns:
  df = df.drop('Unnamed: 0', axis=1)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f'Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ax = sns.scatterplot(
    x=df['age'],
    y=df['thalach'],
    hue=df['target'],
    palette={0: 'blue', 1: 'red'},
    s=60,
    alpha=0.8,
)
plt.title('Tuổi (age) vs Nhịp tim tối đa (thalach)')
plt.xlabel('Tuổi')
plt.ylabel('Nhịp tim tối đa')

handles, _ = ax.get_legend_handles_labels()
plt.legend(handles=handles, title='Bệnh tim', labels=['Không (0)', 'Có (1)'])

plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Khỏe mạnh (0)', 'Bệnh tim (1)'],
    yticklabels=['Khỏe mạnh (0)', 'Bệnh tim (1)'],
)
plt.title('Confusion Matrix')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')

plt.tight_layout()
plt.show()