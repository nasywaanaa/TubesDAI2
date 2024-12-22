# Tugas Besar 2 IF3070 DAI

## Deskripsi Repository

Repository ini berisi implementasi untuk Tugas Besar 2 mata kuliah **IF3070 Dasar Inteligensi Artifisial** untuk **Implementasi Algoritma Pembelajaran Mesin** yang bertujuan untuk melakukan analisis data dan machine learning. Tugas berikut menggunakan dataset terkait URL phishing untuk membangun model prediksi dengan algoritma K-Nearest Neighbors (KNN) dan Naive Bayes dan mencakup proses feature engineering, preprocessing, maupun evaluasi model.

## Cara Setup dan Menjalankan Program

### Prerequisite

- Python terinstall di komputer Anda
- Virtual environment diaktifkan (dapat dipilih sesuai dengan python yang terinstall)

### Langkah Setup

1. Clone repository ini:

   ```bash
   git clone https://github.com/nasywaanaa/TubesDAI2.git
   cd Tubes2DAI
   ```

2. Install dependencies yang diperlukan :

   ```bash
   pip install -r requirements.txt
   ```

3. Pastikan semua file dataset (seperti `train.csv` dan `test.csv`) sudah berada pada direktori yang sesuai

### Cara Menjalankan

1. Untuk menjalankan notebook utama :

   ```bash
   jupyter notebook IF3070_DAI___Tugas_Besar_2_Notebook_Template.ipynb
   ```

2. Jalankan semua sel di dalam notebook sesuai dengan urutan untuk preprocessing, feature engineering, training model (KNN dan Naive Bayes), dan evaluasi

3. Hasil submission dapat dilihat setelah seluruh proses berhasil dijalankan pada file `submission.csv`.

## Pembagian Tugas

| Nama Anggota            | NIM      | Tugas                            |
| ----------------------- | -------- | -------------------------------- |
| Jihan Aurelia           | 18222001 | Implementasi KNN                 |
| Nasywaa Anggun Athiefah | 18222021 | Implementasi Feature Engineering |
| Ricky Wijaya            | 18222043 | Implementasi Naive Bayes         |
| Timotius Vivaldi Gunawan| 18222091 | Implementasi Data Cleaning dan Pre-Processing      |