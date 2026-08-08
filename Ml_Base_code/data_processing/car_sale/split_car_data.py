import os
import pandas as pd
from sklearn.model_selection import train_test_split

#path file goc
file_path = r'D:\data_system\Ml_Dl_book-source\zero-to-mastery-ml\data\car-sales.csv'


#read + processing data
df = pd.read_csv(file_path)

#xoa $ + change -> float
df['Price'] = df['Price'].str.replace(r'[\$,]', '', regex=True).astype(float)

X = df.drop('Price', axis=1)  # X : Make, Colour, Odometer (KM), Doors
y = df['Price']               # y la Price


# encoding
X_encoded = pd.get_dummies(X, columns=['Make', 'Colour'], drop_first=True)



X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.2, 
    random_state=42
)

print("--- CHIA TẬP DỮ LIỆU THÀNH CÔNG ---")
print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước X_test:  {X_test.shape}")
print(f"Kích thước y_train: {y_train.shape}")
print(f"Kích thước y_test:  {y_test.shape}\n")


#save

output_dir = os.path.dirname(file_path)

# Đặt đường dẫn đầu ra cho từng file
path_X_train = os.path.join(output_dir, 'car_sales_X_train.csv')
path_X_test  = os.path.join(output_dir, 'car_sales_X_test.csv')
path_y_train = os.path.join(output_dir, 'car_sales_y_train.csv')
path_y_test  = os.path.join(output_dir, 'car_sales_y_test.csv')

# Xuất ra file CSV (index=False để không lưu thêm cột số thứ tự mặc định)
X_train.to_csv(path_X_train, index=False)
X_test.to_csv(path_X_test, index=False)
y_train.to_csv(path_y_train, index=False)
y_test.to_csv(path_y_test, index=False)

print("--- ĐÃ XUẤT THÀNH CÔNG 4 FILE CSV ---")
print(f"1. {path_X_train}")
print(f"2. {path_X_test}")
print(f"3. {path_y_train}")
print(f"4. {path_y_test}")