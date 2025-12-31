# 📊 Big Data Sentiment Analysis: Amazon Customer Reviews (3.6M Rows)

> **Ujian Akhir Semester - Analisis Big Data** > **Universitas Negeri Surabaya (UNESA) - Prodi Sains Data**

Proyek ini melakukan analisis sentimen berskala besar terhadap **3.6 juta ulasan pelanggan** Amazon. Kami membangun pipeline *end-to-end* mulai dari pemrosesan data terdistribusi menggunakan **Apache Spark**, pemodelan Machine Learning, hingga dashboard wawasan bisnis yang didukung oleh **Redis** untuk performa tinggi.

## 👥 Tim Penyusun (Kelompok 2)

* **Faiz Dwi Febriansyah** (22031554023)
* **Riva Dian Ardiansyah** (22031554043)
* **Michael Luwi Pallea** (22031554055)

---

## 📂 Struktur Repository & Penjelasan File

### 1. `Machine Learning modeling Pyspark-Group2` (Big Data Processing & Modeling)
Notebook ini adalah "mesin utama" proyek yang dijalankan di lingkungan Databricks/Spark Cluster.

* **Fokus:** ETL, NLP Preprocessing, dan Model Training.
* **Metodologi (Berdasarkan Laporan):**
    * **Dataset:** 3.6 Juta baris (>300MB).
    * **Preprocessing (8 Tahap):** Case folding, Regex cleaning (a-z), Tokenisasi, Stopwords Removal (NLTK), Stemming (PorterStemmer), hingga TF-IDF (10.000 fitur).
    * **Data Splitting:** 70% Training : 20% Validation : 10% Testing.
    * **Model:** Logistic Regression (dipilih karena performa lebih baik dibanding Random Forest pada data dimensi tinggi).
* **Output:** Model klasifikasi biner (Positif/Negatif) dengan probabilitas prediksi.

### 2. `Data Analisis.ipynb` (Business Intelligence & Redis Integration)
Notebook ini berfokus pada eksplorasi data (EDA) dan simulasi implementasi di dunia nyata menggunakan caching.

* **Fokus:** Analisis pola, visualisasi risiko, dan optimasi performa data.
* **Fitur Utama:**
    * **Analisis Statistik:** Distribusi panjang teks vs sentimen.
    * **Action Priority Matrix:** Visualisasi heatmap untuk menentukan prioritas perbaikan bisnis.
    * **Risk Analysis:** Identifikasi topik dengan rasio negatif >40%.
* **🚀 Implementasi Redis:**
    * **Fungsi:** Menggunakan **Redis** sebagai *In-Memory Key-Value Store*.
    * **Tujuan:** Menyimpan hasil agregasi berat (seperti frekuensi kata atau statistik per kategori) ke dalam cache RAM.
    * **Benefit:** Memungkinkan pengambilan data untuk visualisasi dashboard secara *real-time* tanpa harus melakukan query ulang ke database utama (SQL/Spark) yang lambat. Ini mensimulasikan arsitektur aplikasi analitik modern yang skalabel.

---

## 📊 Ringkasan Hasil (Performance metrics)

Berdasarkan hasil pengujian pada *Test Set* (10% data terpisah):

<img width="1545" height="898" alt="image" src="https://github.com/user-attachments/assets/c15f1327-6821-44e8-ba46-b5871836aae6" />


| Metric | Hasil | Keterangan |
| :--- | :--- | :--- |
| **Akurasi** | **84%** | Model mampu memprediksi sentimen dengan sangat tepat. |
| **AUC Score** | **~0.92** | Kemampuan membedakan kelas positif/negatif sangat baik. |
| **Distribusi Kelas** | **Balanced** | Negatif (49.96%) vs Positif (50.04%). |

---

## 🛠️ Teknologi & Tools

* **Core Processing:** Apache Spark (PySpark MLlib)
* **Database & Caching:** SQLite, **Redis** (untuk High-Performance Data Retrieval)
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **NLP:** NLTK
* **Bahasa:** Python 3.x

---

## 🚀 Cara Menjalankan

1.  **Persiapan Environment:**
    * Pastikan Apache Spark terinstall (atau gunakan Google Colab/Databricks).
    * Pastikan server **Redis** lokal berjalan (untuk file `Data Analisis.ipynb`).
    
2.  **Instalasi Library:**
    ```bash
    pip install pyspark pandas numpy matplotlib seaborn redis nltk wordcloud
    ```

3.  **Eksekusi:**
    * Jalankan `(Clone) Projek ABD 2.ipynb` terlebih dahulu untuk melatih model dan menghasilkan data bersih.
    * Jalankan `Data Analisis.ipynb` untuk melihat insight bisnis dan simulasi caching Redis.

---

## 📝 Kesimpulan Bisnis

Dari analisis data, ditemukan bahwa ulasan negatif cenderung memiliki teks yang lebih panjang (rata-rata 420 karakter) dibandingkan ulasan positif. Isu "Kualitas Produk" menjadi penyumbang terbesar kekecewaan pelanggan, sehingga direkomendasikan sebagai prioritas utama perbaikan.
