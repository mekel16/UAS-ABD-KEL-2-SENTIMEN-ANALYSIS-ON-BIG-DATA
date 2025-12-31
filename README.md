# Sentiment Analysis on Big Data using PySpark

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi analisis sentimen pada Big Data menggunakan Apache Spark (PySpark) untuk mengelola dan memproses dataset dalam skala besar. Proyek ini bertujuan untuk mengklasifikasikan review produk menjadi sentimen positif atau negatif menggunakan teknik machine learning dengan pendekatan distributed computing.

Dataset yang digunakan berisi sekitar 3.6 juta review produk yang diklasifikasikan menjadi dua kategori:
- **Class 1 (Negatif)**: Review dengan sentimen negatif
- **Class 2 (Positif)**: Review dengan sentimen positif

---

## 👥 Anggota Kelompok

**Kelompok 2 - Analisis Big Data**

*(Silahkan tambahkan nama anggota kelompok di sini)*

---

## 📊 Dataset

### Informasi Dataset
- **Nama Dataset**: Amazon Product Reviews
- **Sumber**: Workspace Databricks (`workspace.default.train_1`)
- **Jumlah Record Awal**: 3,599,999 data
- **Jumlah Record Setelah Cleaning**: 3,594,734 data
- **Fitur Utama**:
  - `class`: Label sentimen (1 = Negatif, 2 = Positif)
  - `title`: Judul review
  - `review`: Teks lengkap review produk

### Distribusi Data
- **Negatif**: 1,796,130 (49.97%)
- **Positif**: 1,798,604 (50.03%)
- Dataset **balanced** antara kelas positif dan negatif

### Karakteristik Teks
- **Review Negatif**: Rata-rata 420.76 karakter, 77.13 kata
- **Review Positif**: Rata-rata 389.47 karakter, 71.20 kata
- Review negatif cenderung lebih panjang dan bervariasi dibanding review positif

---

## 🛠️ Teknologi dan Library yang Digunakan

### Big Data Framework
- **Apache Spark (PySpark)**: Distributed computing framework untuk pemrosesan Big Data
- **Databricks**: Platform untuk menjalankan PySpark notebook

### Machine Learning Libraries
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
```

### Data Processing & NLP
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pyspark.sql.functions import col, when, udf, length, avg, count, regexp_replace
```

### Visualization Libraries
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
```

---

## 📁 Struktur Proyek

```
UAS-ABD-KEL-2-SENTIMEN-ANALYSIS-ON-BIG-DATA/
│
├── Sentimen Analysis on BiG Data (py-Spark)- group 2.ipynb
│   └── Notebook utama dengan implementasi lengkap sentiment analysis
│
├── Data Analisis.ipynb
│   └── Notebook untuk exploratory data analysis
│
├── LAPORAN UJIAN AKHIR SEMESTER ANALISIS BIG DATA.pdf
│   └── Laporan lengkap proyek
│
└── README.md
    └── Dokumentasi proyek
```

---

## 🔬 Langkah-langkah Analisis

### 1. **Benchmarking Spark vs Pandas**
```python
# Membandingkan performa Spark dan Pandas
- Spark Processing Time: 15.08 seconds
- Pandas Collection Time: 17.59 seconds
```

### 2. **Load Data**
```python
df = spark.table("workspace.default.train_1")
```

### 3. **Exploratory Data Analysis (EDA)**

#### a. Pemeriksaan Dataset
- Menampilkan schema data
- Menghitung jumlah total record
- Menampilkan sample data

#### b. Missing Values
- **Sebelum cleaning**: 48 missing values pada kolom `title`
- **Setelah cleaning**: 0 missing values
```python
df = df.na.drop()
```

#### c. Duplicate Data
- **Duplicate ditemukan**: 5,217 review duplikat
- **Setelah deduplikasi**: 0 duplikat
```python
df = df.dropDuplicates(['review'])
```

#### d. Distribusi Kelas
- Visualisasi distribusi menggunakan bar chart dan pie chart
- Dataset terbukti balanced (50-50)

#### e. Analisis Panjang Teks
- Histogram distribusi panjang karakter dan jumlah kata
- Box plot untuk melihat outliers
- Review negatif memiliki variasi panjang yang lebih besar

### 4. **Text Preprocessing**

#### Fungsi Preprocessing
```python
def pra_proses(text):
    text = str(text).lower()                      # Case folding
    text = re.sub(r"[^a-z\s]", "", text)         # Hapus karakter khusus
    tokens = text.split()                         # Tokenization
    tokens = [w for w in tokens if w not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(w) for w in tokens]   # Stemming
    return " ".join(tokens)
```

#### Tahapan:
1. **Case Folding**: Mengubah semua teks menjadi lowercase
2. **Punctuation Removal**: Menghapus tanda baca dan karakter khusus
3. **Tokenization**: Memisahkan teks menjadi kata-kata
4. **Stopwords Removal**: Menghapus kata-kata umum yang tidak bermakna
5. **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan Porter Stemmer

#### Label Encoding
```python
# Mengubah label: 1 → 0 (Negatif), 2 → 1 (Positif)
df = df.withColumn("class", when(col("class") == 1, 0).otherwise(1))
```

### 5. **Feature Engineering**

#### TF-IDF (Term Frequency-Inverse Document Frequency)
```python
# Pipeline untuk feature extraction
tokenizer = Tokenizer(inputCol="review", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
```

### 6. **Model Training**

#### Algoritma: Logistic Regression
```python
lr = LogisticRegression(
    featuresCol="features",
    labelCol="class",
    maxIter=100
)

# Pipeline lengkap
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])
```

#### Train-Test Split
- **Training**: 80% data
- **Testing**: 20% data

### 7. **Model Evaluation**

#### Metrik Evaluasi:
- **Accuracy**: Akurasi keseluruhan model
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan mendeteksi semua kasus positif
- **F1-Score**: Harmonic mean dari precision dan recall
- **AUC-ROC**: Area Under the Curve

---

## 📈 Hasil dan Kesimpulan

### Temuan Utama dari EDA:
1. **Dataset Balanced**: Distribusi kelas hampir sempurna (49.97% negatif, 50.03% positif)
2. **Data Quality**: Setelah cleaning, dataset berkurang 5,265 records (0.15%) namun kualitas meningkat
3. **Karakteristik Review**:
   - Review negatif rata-rata lebih panjang (420 vs 389 karakter)
   - Review negatif lebih bervariasi dalam panjangnya
   - Review positif lebih konsisten

### Performance Insights:
- **Spark Processing** efisien untuk dataset besar (3.6M records)
- Processing time: ~15 detik untuk operasi read dan write
- Overhead conversion ke Pandas: ~17 detik

### Model Performance:
*(Hasil akan diisi setelah model training selesai)*

### Kesimpulan:
1. PySpark terbukti efektif untuk menangani dataset besar (3.6M+ records)
2. Text preprocessing berhasil mengurangi noise dalam data
3. TF-IDF sebagai feature extraction menghasilkan representasi teks yang baik
4. Logistic Regression dengan PySpark ML dapat melakukan training pada big data dengan efisien

---

## 🚀 Cara Menjalankan Proyek

### Prerequisites
- **Databricks Account** atau **Apache Spark cluster**
- **Python 3.7+**
- **Java 8 atau 11** (untuk Spark)

### Setup Environment

#### 1. Install Required Libraries
```bash
pip install pyspark
pip install nltk
pip install pandas
pip install matplotlib
pip install seaborn
pip install wordcloud
```

#### 2. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Running on Databricks

1. **Upload Notebook**:
   - Login ke Databricks workspace
   - Import `Sentimen Analysis on BiG Data (py-Spark)- group 2.ipynb`

2. **Attach to Cluster**:
   - Create atau attach ke existing Spark cluster
   - Recommended: Runtime 11.3 LTS atau lebih baru

3. **Load Dataset**:
   ```python
   df = spark.table("workspace.default.train_1")
   ```
   *(Pastikan dataset sudah tersedia di workspace)*

4. **Run All Cells**:
   - Jalankan semua cell secara berurutan
   - Monitor execution time dan resource usage

### Running Locally with PySpark

```python
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load dataset (sesuaikan dengan lokasi file Anda)
df = spark.read.csv("path/to/dataset.csv", header=True, inferSchema=True)

# Lanjutkan dengan code dari notebook
```

### Tips untuk Optimasi:
1. **Memory Management**: Sesuaikan `spark.driver.memory` dan `spark.executor.memory`
2. **Partitioning**: Gunakan `.repartition()` untuk data yang sangat besar
3. **Caching**: Cache dataframe yang sering digunakan dengan `.cache()`
4. **Sampling**: Untuk testing, gunakan `.sample()` untuk subset data

---

## 📝 Catatan Tambahan

### Limitasi Proyek:
- Dataset terbatas pada review berbahasa Inggris
- Preprocessing masih basic (bisa ditingkatkan dengan lemmatization)
- Model terbatas pada Logistic Regression (bisa eksplorasi model lain)

### Future Improvements:
1. **Model Enhancement**:
   - Implementasi Random Forest, Gradient Boosting
   - Deep Learning dengan Spark NLP
   - Ensemble methods

2. **Feature Engineering**:
   - N-grams (bigrams, trigrams)
   - Word embeddings (Word2Vec, GloVe)
   - Sentiment lexicon features

3. **Deployment**:
   - Model serving dengan MLflow
   - Real-time prediction dengan Spark Streaming
   - REST API untuk inference

---

## 📚 Referensi

1. Apache Spark Documentation: https://spark.apache.org/docs/latest/
2. PySpark ML Guide: https://spark.apache.org/docs/latest/ml-guide.html
3. NLTK Documentation: https://www.nltk.org/
4. Databricks Documentation: https://docs.databricks.com/

---

## 📧 Kontak

Untuk pertanyaan atau feedback, silakan hubungi:
- Email: *(tambahkan email contact)*
- Repository: [GitHub](https://github.com/mekel16/UAS-ABD-KEL-2-SENTIMEN-ANALYSIS-ON-BIG-DATA)

---

**Dibuat sebagai bagian dari Ujian Akhir Semester - Mata Kuliah Analisis Big Data**

*Last Updated: January 2025*
