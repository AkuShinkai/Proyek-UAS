import streamlit as st
import joblib
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Memuat model dan encoder yang telah dilatih
pipeline = joblib.load('stress_predictor_pipeline.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Memuat dataset untuk mendapatkan kolom fitur
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# 1. Mengonversi kolom 'Blood Pressure' menjadi dua kolom terpisah
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)

# 2. Ubah tipe data kolom 'Systolic' dan 'Diastolic' menjadi numerik
data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')
data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

# 3. Hapus kolom 'Blood Pressure' yang sudah tidak diperlukan lagi
data = data.drop(columns=['Blood Pressure'])

# Mendapatkan nama kolom fitur yang digunakan saat pelatihan
feature_columns = data.drop(['Person ID', 'Stress Level'], axis=1).columns

# Judul aplikasi
st.title("Prediksi Level Stres Berdasarkan Pola Tidur")

# Sidebar untuk memilih menu
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Penjelasan Dataset", "Analisis Data Eksplorasi (EDA)", "Prediksi Level Stres"]
)

if menu == "Penjelasan Dataset":
    st.header("Penjelasan Dataset")
    st.write("""
        Dataset ini berisi informasi tentang pola tidur, aktivitas fisik, dan faktor kesehatan lainnya,
        beserta tingkat stres yang dilaporkan oleh individu. Berikut adalah kolom-kolom dalam dataset:
        - **Person ID**: Identifikasi untuk setiap individu.
        - **Gender**: Jenis kelamin orang (Laki-laki/Perempuan).
        - **Age**: Usia orang dalam tahun.
        - **Occupation**: Pekerjaan atau profesi orang tersebut.
        - **Sleep Duration**: Durasi tidur orang per hari dalam jam.
        - **Quality of Sleep**: Penilaian subjektif tentang kualitas tidur, dengan rentang 1 hingga 10.
        - **Physical Activity Level**: Jumlah menit aktivitas fisik yang dilakukan orang per hari.
        - **Stress Level**: Penilaian subjektif tentang tingkat stres yang dialami orang, dengan rentang 1 hingga 10.
        - **BMI Category**: Kategori BMI orang tersebut (misalnya: Underweight, Normal, Overweight).
        - **Blood Pressure**: Pengukuran tekanan darah orang tersebut (sistolik/diastolik).
        - **Heart Rate**: Denyut jantung orang tersebut dalam bpm (detak per menit).
        - **Daily Steps**: Jumlah langkah yang ditempuh orang per hari.
        - **Sleep Disorder**: Gangguan tidur yang dialami orang tersebut (None, Insomnia, Sleep Apnea).
    """)

elif menu == "Analisis Data Eksplorasi (EDA)":
    st.header("Analisis Data Eksplorasi (EDA)")

    # Mengisi nilai NaN dengan kategori yang sudah ada (menggunakan mode atau kategori yang paling sering muncul)
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for column in categorical_columns:
        if column in data.columns:
            # Mengisi NaN dengan kategori yang paling sering muncul
            most_frequent = data[column].mode()[0]
            data[column] = data[column].fillna(most_frequent)

            # Melakukan encoding untuk kolom kategorikal
            data[column] = label_encoders[column].transform(data[column])

    # Menampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(data.describe())
    st.write("""
    Statistik deskriptif ini memberikan gambaran umum tentang distribusi data numerik, seperti rentang nilai, rata-rata, standar deviasi, dan kuartil untuk kolom-kolom seperti `Age`, `Stress Level`, dan `Sleep Duration`.
    """)

    # Menampilkan heatmap korelasi hanya untuk kolom numerik
    st.subheader("Heatmap Korelasi")
    correlation = data.select_dtypes(include=['float64', 'int64']).corr()  # Hanya untuk kolom numerik
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()
    st.write("""
    Heatmap ini menunjukkan hubungan antar fitur numerik dalam dataset, seperti antara `Sleep Duration` dan `Stress Level`. Warna yang lebih gelap menunjukkan korelasi yang lebih kuat, baik positif maupun negatif.
    """)

    # Menampilkan distribusi tingkat stres
    st.subheader("Distribusi Level Stres")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['Stress Level'], bins=10, kde=True, color='blue')
    st.pyplot()
    st.write("""
    Histogram ini menunjukkan distribusi dari tingkat stres yang dilaporkan oleh individu. Dengan melihat distribusi ini, kita dapat memahami apakah sebagian besar orang memiliki tingkat stres rendah, sedang, atau tinggi.
    """)

    # Menampilkan distribusi durasi tidur
    st.subheader("Distribusi Durasi Tidur")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['Sleep Duration'], bins=20, kde=True, color='green')
    st.pyplot()
    st.write("""
    Histogram ini menunjukkan distribusi durasi tidur yang dilaporkan oleh individu. Ini memberikan gambaran mengenai berapa banyak orang yang tidur lebih banyak atau lebih sedikit dari durasi tidur yang direkomendasikan.
    """)

    # Visualisasi hubungan antara Durasi Tidur dan Tingkat Stres
    st.subheader("Hubungan Antara Durasi Tidur dan Tingkat Stres")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Sleep Duration', y='Stress Level', data=data, color='purple')
    st.pyplot()
    st.write("""
    Grafik ini menggambarkan hubungan antara durasi tidur dan tingkat stres. Dapat dilihat apakah ada pola yang mengindikasikan bahwa kurang tidur berhubungan dengan tingkat stres yang lebih tinggi.
    """)

    # Boxplot untuk membandingkan Stress Level berdasarkan kategori Gender
    st.subheader("Perbandingan Level Stres Berdasarkan Gender")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Gender', y='Stress Level', data=data, palette="Set2")
    st.pyplot()
    st.write("""
    Boxplot ini membandingkan distribusi tingkat stres antara pria dan wanita. Ini memberikan gambaran apakah ada perbedaan signifikan dalam tingkat stres antara gender yang berbeda.
    """)

    # Visualisasi tingkat stres berdasarkan kategori BMI
    st.subheader("Tingkat Stres Berdasarkan Kategori BMI")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='BMI Category', y='Stress Level', data=data, palette="Set1")
    st.pyplot()
    st.write("""
    Boxplot ini menunjukkan perbandingan tingkat stres berdasarkan kategori BMI (misalnya: Kurus, Normal, Gemuk). Ini memberikan wawasan apakah seseorang dengan BMI tertentu lebih rentan terhadap stres.
    """)


elif menu == "Prediksi Level Stres":
    st.header("Masukkan Data Anda:")

    # Input pengguna
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia", min_value=18, max_value=100, value=30)
    occupation = st.selectbox("Pekerjaan", ["Bekerja", "Mahasiswa", "Pengangguran", "Pensiun"])
    sleep_duration = st.number_input("Durasi Tidur (jam)", min_value=0, max_value=24, value=8)
    sleep_quality = st.slider("Kualitas Tidur (1-10)", min_value=1, max_value=10, value=7)
    physical_activity_level = st.number_input("Tingkat Aktivitas Fisik (menit/hari)", min_value=0, value=30)
    bmi_category = st.selectbox("Kategori BMI", ["Kurus", "Normal", "Gemuk"])
    blood_pressure_systolic = st.number_input("Tekanan Darah Sistolik", min_value=50, max_value=200, value=120)
    blood_pressure_diastolic = st.number_input("Tekanan Darah Diastolik", min_value=30, max_value=120, value=80)
    heart_rate = st.number_input("Detak Jantung (bpm)", min_value=40, max_value=200, value=70)
    daily_steps = st.number_input("Langkah Harian", min_value=0, value=10000)
    sleep_disorder = st.selectbox("Gangguan Tidur", ["Tidak Ada", "Insomnia", "Sleep Apnea"])

    # Menyiapkan data input untuk prediksi
    input_data = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": sleep_quality,
        "Physical Activity Level": physical_activity_level,
        "BMI Category": bmi_category,
        "Systolic": blood_pressure_systolic,
        "Diastolic": blood_pressure_diastolic,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Sleep Disorder": sleep_disorder
    }

    # Fungsi untuk melakukan prediksi
    def predict_stress(input_data):
        # Melakukan encoding untuk fitur kategorikal
        for column, le in label_encoders.items():
            if column in input_data:
                try:
                    input_data[column] = le.transform([input_data[column]])[0]
                except ValueError:
                    # Menangani kategori yang tidak ada dalam encoder
                    input_data[column] = le.transform([le.classes_[0]])[0]  # fallback ke kelas pertama
                    print(f"Kategori '{input_data[column]}' tidak ditemukan, fallback ke kelas pertama.")
        
        # Memastikan data input memiliki kolom fitur yang sama dengan data pelatihan
        input_data_df = pd.DataFrame([input_data], columns=feature_columns)
        
        # Memastikan data di-scale sesuai dengan pipeline
        input_data_scaled = pipeline.named_steps['scaler'].transform(input_data_df)

        # Melakukan prediksi menggunakan pipeline model yang sudah dilatih
        prediction = pipeline.named_steps['model'].predict(input_data_scaled)[0]
        return prediction

    # Menampilkan hasil prediksi
    predicted_stress = predict_stress(input_data)
    st.write(f"Prediksi level stres: {predicted_stress:.2f}")
    # Menampilkan hasil prediksi dan memberikan rekomendasi berdasarkan tingkat stres
if predicted_stress <= 3:
    st.write("Tingkat Stres Anda: Rendah")
    st.write("""
        Berdasarkan prediksi, tingkat stres Anda termasuk dalam kategori rendah. Ini berarti Anda cenderung merasa tenang dan tidak tertekan. 
        Teruskan gaya hidup sehat Anda dengan menjaga kualitas tidur dan aktivitas fisik yang cukup.
    """)
elif 3 < predicted_stress <= 6:
    st.write("Tingkat Stres Anda: Sedang")
    st.write("""
        Anda berada pada tingkat stres yang sedang. Ini berarti ada beberapa faktor yang mungkin mempengaruhi kesejahteraan Anda, 
        seperti kurang tidur, tekanan pekerjaan, atau gaya hidup yang sibuk. Disarankan Anda untuk mencoba teknik relaksasi, 
        seperti meditasi atau olahraga ringan, dan pastikan tidur Anda cukup.
    """)
elif 6 < predicted_stress <= 8:
    st.write("Tingkat Stres Anda: Tinggi")
    st.write("""
        Prediksi menunjukkan bahwa tingkat stres Anda cukup tinggi. Stres yang berkelanjutan dapat mempengaruhi kesehatan fisik dan mental Anda.
        Disarankan Anda untuk mencoba teknik manajemen stres seperti yoga, meditasi, atau berbicara dengan seorang profesional 
        untuk mendapatkan dukungan. Jangan ragu untuk mencari bantuan jika diperlukan.
    """)
else:
    st.write("Tingkat Stres Anda: Sangat Tinggi")
    st.write("""
        Tingkat stres Anda sangat tinggi, yang dapat mempengaruhi kualitas hidup Anda. Ini adalah waktu yang baik untuk mengevaluasi 
        penyebab stres dan mencari cara untuk mengurangi beban mental Anda. Pertimbangkan untuk melakukan perubahan dalam rutinitas harian, 
        berfokus pada tidur yang cukup, mengurangi stres di tempat kerja, atau berkonsultasi dengan seorang ahli untuk mendapatkan dukungan lebih lanjut.
    """)

