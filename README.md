# 📊 Big Data Sentiment Analysis & Business Insights

Proyek ini bertujuan untuk melakukan analisis sentimen berskala besar menggunakan dataset ulasan pelanggan (3.6 Juta baris). Proyek ini menggabungkan kekuatan **PySpark** untuk pemrosesan Machine Learning yang terdistribusi dan **SQL/Pandas** untuk analisis data eksploratif (EDA) mendalam guna menghasilkan rekomendasi bisnis.

## 📂 Struktur Repository

Repository ini terdiri dari dua komponen utama:

1.  **`(Clone) Projek ABD 2.ipynb`**
    * **Fokus:** *Machine Learning Pipeline* (PySpark).
    * **Deskripsi:** Notebook ini menangani pemrosesan data skala besar, *text preprocessing*, ekstraksi fitur (TF-IDF), dan pemodelan klasifikasi sentimen.
2.  **`Data Analisis.ipynb`**
    * **Fokus:** *Exploratory Data Analysis* (EDA) & *Business Intelligence*.
    * **Deskripsi:** Notebook ini menggunakan SQLite dan Pandas untuk menganalisis pola ulasan, panjang teks, topik dominan, dan memberikan rekomendasi strategis berdasarkan data.

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa:** Python 3.x
* **Big Data Processing:** Apache PySpark
* **Data Analysis:** Pandas, SQLite3
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **NLP:** NLTK (Stopwords, Tokenization)

---

## 📝 Deskripsi Dataset

Dataset yang digunakan berjumlah **3,594,576 baris** (setelah pembersihan) dengan fitur utama:
* `label`: Sentimen ulasan (1 = Negatif, 2 = Positif).
* `title`: Judul ulasan.
* `review`: Isi teks ulasan.

> **Distribusi Kelas:** Dataset seimbang (Balanced) dengan proporsi ~50% Negatif dan ~50% Positif.

---

## 🚀 Alur Kerja Proyek (Workflow)

### 1. Machine Learning Pipeline (PySpark)
File: `Sentimen Analyss.ipynb`

* **Data Loading:** Memuat data dari tabel workspace Databricks/Hive.
* **Preprocessing:**
    * Pembersihan teks (Regex untuk menghapus simbol/angka).
    * Penghapusan *Stopwords* menggunakan NLTK.
    * Tokenisasi.
* **Feature Extraction:** Menggunakan `HashingTF` dan `IDF` (TF-IDF) untuk mengubah teks menjadi vektor numerik.
* **Modeling:** Melatih model **Logistic Regression** untuk klasifikasi biner.
* **Visualisasi:** Membuat *Word Cloud* untuk melihat kata-kata yang paling sering muncul pada sentimen positif dan negatif.

### 2. Analisis & Insight Bisnis (SQLite/Pandas)
File: `Data Analisis.ipynb`

* **Data Cleaning:** Menghapus *missing values* dan duplikasi data (~9,000 duplikat dihapus).
* **Feature Engineering:** Menambahkan kolom `text_length` untuk analisis panjang ulasan.
* **Database Storage:** Menyimpan data bersih ke dalam database lokal `uas_sentiment_3.6M.db` untuk kueri SQL yang efisien.
* **Key Insights (SQL Queries):**
    * **Distribusi Sentimen:** Analisis rasio ulasan positif vs negatif.
    * **Analisis Panjang Ulasan:** Menemukan bahwa ulasan negatif cenderung sedikit lebih panjang (Rata-rata 420 karakter) dibandingkan ulasan positif (389 karakter).
    * **Topik Spesifik:** Analisis kata kunci (e.g., "Quality", "Price", "Book/Movie") untuk melihat kategori mana yang paling banyak mendapat keluhan.
    * **Rekomendasi:** Mengidentifikasi prioritas penanganan, seperti ulasan negatif pendek yang mungkin mengindikasikan kekecewaan mendalam (churn risk).

---

## 📊 Hasil Analisis Utama

Berikut adalah ringkasan statistik dari analisis data:

| Kategori | Jumlah Ulasan | Persentase | Rata-rata Panjang Karakter |
| :--- | :--- | :--- | :--- |
| **Positif (Label 2)** | 1,798,547 | 50.04% | 389.5 |
| **Negatif (Label 1)** | 1,796,029 | 49.96% | 420.8 |

**Temuan Menarik:**
* Topik **"Quality Issues"** memiliki persentase sentimen negatif tertinggi (**55.77%**), menandakan pelanggan sangat sensitif terhadap kualitas produk.
* Topik **"Price Concerns"** relatif lebih rendah risiko negatifnya (**40.77%**) dibandingkan isu kualitas.

---

## 📦 Cara Menjalankan

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/username/repository-anda.git](https://github.com/username/repository-anda.git)
    ```
2.  **Install dependencies:**
    Pastikan Anda memiliki library yang dibutuhkan.
    ```bash
    pip install pyspark pandas matplotlib seaborn wordcloud nltk redis
    ```
3.  **Jalankan Notebook:**
    * Buka `(Clone) Projek ABD 2.ipynb` untuk melihat proses training model PySpark.
    * Buka `Data Analisis.ipynb` untuk melihat analisis SQL dan insight bisnis.

---

## 👤 Author

[Faiz Dwi Febriansyah]
[Michael Luwi P]
[Riva Dian Ardiansyah]
