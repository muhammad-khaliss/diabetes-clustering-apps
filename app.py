import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# Load Data
diabetes_df = pd.read_csv('diabetes.csv')
kopi_df = pd.read_csv('lokasi_gerai_kopi_clean.csv')

# Load Model & Scaler
with open('model.pkl', 'rb') as model_file:
    knn = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Sidebar
menu = st.sidebar.radio("Pilih Menu", ["Klasifikasi Diabetes", "Clustering Gerai Kopi"])

# Halaman Klasifikasi
if menu == "Klasifikasi Diabetes":
    st.title("Klasifikasi Diabetes - KNN")
    st.write("""
    ## Tentang Aplikasi

    Aplikasi ini dikembangkan untuk membantu memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa parameter kesehatan. Dengan memanfaatkan algoritma K-Nearest Neighbors (KNN), aplikasi ini dapat memproses data seperti kadar glukosa, tekanan darah, indeks massa tubuh, usia, dan faktor keturunan untuk menghasilkan prediksi.

    Model ini dibangun berdasarkan dataset diabetes Pima Indian yang sering digunakan untuk riset dan pembelajaran klasifikasi kesehatan.

    **Catatan:** Prediksi ini bersifat simulasi untuk edukasi dan tidak dapat dijadikan sebagai diagnosis medis resmi.
    """)

    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']
    X_scaled = scaler.transform(X)
    y_pred = knn.predict(X_scaled)

    st.subheader("Metrik Klasifikasi")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Input Data Baru")
    pregnancies = st.number_input('Jumlah Kehamilan', 0, 20)
    glucose = st.number_input('Kadar Glukosa', 0.0, 300.0)
    blood_pressure = st.number_input('Tekanan Darah', 0.0, 200.0)
    skin_thickness = st.number_input('Tebal Lipatan Kulit', 0.0, 100.0)
    insulin = st.number_input('Kadar Insulin', 0.0, 1000.0)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', 0.0, 100.0)
    dpf = st.number_input('Fungsi Keturunan Diabetes', 0.0, 2.5)
    age = st.number_input('Usia', 0, 120)

    if st.button('Prediksi'):
        data_baru = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        hasil = knn.predict(data_baru)
        st.success(f"Hasil Prediksi: {'Positif Diabetes' if hasil[0] == 1 else 'Negatif Diabetes'}")

# Halaman Clustering
else:
    st.title("Clustering Lokasi Gerai Kopi - KMeans")
    st.write("""
    Aplikasi ini mengelompokkan lokasi gerai kopi berdasarkan koordinat x dan y menggunakan algoritma KMeans.
    """)

    X_kopi = kopi_df[['x', 'y']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kopi_df['Cluster'] = kmeans.fit_predict(X_kopi)

    st.subheader("Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=kopi_df, x='x', y='y', hue='Cluster', palette='Set2', s=100, ax=ax)
    st.pyplot(fig)

    st.subheader("Input Lokasi Baru")
    lokasi_x = st.number_input('Koordinat X', 0.0, 100.0)
    lokasi_y = st.number_input('Koordinat Y', 0.0, 100.0)

    if st.button('Cek Cluster'):
        cluster_pred = kmeans.predict([[lokasi_x, lokasi_y]])
        st.success(f"Lokasi tersebut masuk ke dalam Cluster: {cluster_pred[0]}")
