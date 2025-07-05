import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load Data
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Latih Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Simpan Model & Scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("model.pkl dan scaler.pkl berhasil dibuat sesuai environment kamu!")
