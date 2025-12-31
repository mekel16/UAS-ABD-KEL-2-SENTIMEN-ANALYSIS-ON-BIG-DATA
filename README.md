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
- **Redis**: In-memory data store untuk caching dan real-time analytics

### Cache & Storage
- **Redis**: 
  - Fast caching untuk query results
  - Real-time sentiment counters
  - Aggregated statistics storage
  - Performance: ~100-200x faster than disk-based queries untuk operasi sederhana
- **SQLite**: Persistent storage untuk complex queries dan data historis

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
│   ├── Redis implementation untuk caching dan real-time analytics
│   ├── SQLite untuk persistent storage dan complex queries
│   ├── Performance benchmarking: Redis vs SQLite
│   ├── Exploratory data analysis dengan visualisasi
│   └── Statistical insights dan business recommendations
│
├── LAPORAN UJIAN AKHIR SEMESTER ANALISIS BIG DATA.pdf
│   └── Laporan lengkap proyek
│
└── README.md
    └── Dokumentasi proyek
```

---

## 🚀 Redis Implementation

### Why Redis for Sentiment Analysis?

Redis adalah **in-memory data store** yang memberikan performa exceptional untuk Big Data analytics. Dalam proyek ini, Redis digunakan sebagai **complementary technology** bersama SQLite untuk mengoptimalkan performa query.

#### Key Benefits:

1. **⚡ Speed**: In-memory operations ~100-200x lebih cepat dibanding disk-based database
2. **🔄 Real-time Counters**: Track sentiment distribution secara instant dengan atomic operations
3. **💾 Smart Caching**: Reduce database load untuk repeated queries
4. **📈 Scalability**: Handle millions of operations per second
5. **🎯 Simplicity**: Built-in data structures (counters, hashes, sorted sets) untuk analytics

### Redis Use Cases dalam Proyek Ini:

#### 1. **Sentiment Statistics Caching**

Menyimpan hasil aggregasi untuk instant retrieval:

```python
# Store aggregated stats in Redis
r.hmset('sentiment:stats', {
    'total_reviews': 3594576,
    'negative': 1796029,
    'positive': 1798547,
    'avg_length_neg': 420.8,
    'avg_length_pos': 389.5
})

# Instant retrieval (< 1ms)
stats = r.hgetall('sentiment:stats')
```

**Use case**: Dashboard yang membutuhkan real-time statistics tanpa query database berulang kali.

#### 2. **Real-time Sentiment Counters**

Track sentiment distribution dengan atomic counters:

```python
# Increment counters as reviews are processed
r.incr('sentiment:count:negative')  # Atomic operation, thread-safe
r.incr('sentiment:count:positive')

# Get current counts instantly
neg = int(r.get('sentiment:count:negative'))
pos = int(r.get('sentiment:count:positive'))
```

**Use case**: Real-time monitoring dashboard, streaming analytics pipeline.

#### 3. **Top Reviews Storage**

Store top-K reviews untuk fast retrieval:

```python
# Store top 100 positive reviews in Redis hash
for idx, review in top_positive_reviews.iterrows():
    r.hset('reviews:top:positive', f'review_{idx}', review['text'])

# Fast retrieval for display (no SQL join needed)
top_reviews = r.hgetall('reviews:top:positive')
```

**Use case**: API endpoint yang menampilkan best/worst reviews tanpa full table scan.

#### 4. **Length Distribution with Sorted Sets**

Analyze review length percentiles menggunakan Redis sorted sets:

```python
# Store review lengths in sorted set
for idx, row in df.iterrows():
    r.zadd('reviews:lengths', {f'review_{idx}': len(row['text'])})

# Get median and 95th percentile instantly
median_idx = r.zcard('reviews:lengths') // 2
median_length = r.zrange('reviews:lengths', median_idx, median_idx, withscores=True)
```

**Use case**: Quick percentile queries untuk outlier detection, data quality monitoring.

### Performance Comparison

Benchmark results dari `Data Analisis.ipynb`:

| Operation | SQLite | Redis | Speedup |
|-----------|--------|-------|---------|
| Count queries (single) | ~50ms | ~0.5ms | **100x** |
| Aggregations (multiple fields) | ~200ms | ~1ms | **200x** |
| Top-K retrieval | ~150ms | ~2ms | **75x** |
| Stats lookup (cached) | ~80ms | ~0.3ms | **267x** |
| Repeated queries (100x) | ~5000ms | ~30ms | **167x** |

**Kesimpulan**: Redis memberikan **100-300x speedup** untuk operasi sederhana yang sering diulang.

### Redis Setup untuk Proyek Ini

#### Installation:

```bash
# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install redis-server

# macOS (Homebrew)
brew install redis

# Windows
# Download dari https://redis.io/download atau gunakan WSL
```

#### Start Redis Server:

```bash
# Start Redis server
redis-server

# Test connection
redis-cli ping
# Output: PONG
```

#### Python Client Installation:

```bash
pip install redis
```

#### Connection Configuration:

```python
import redis

# Connect to Redis (localhost)
r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # Auto-decode bytes to strings
)

# Test connection
print("Redis connected:", r.ping())  # Should return True
```

### Hybrid Strategy: Redis + SQLite

Proyek ini menggunakan **hybrid approach** untuk maximize efficiency:

| Technology | Use For | Strength |
|------------|---------|----------|
| **Redis** | Caching, counters, real-time stats | Speed, simplicity |
| **SQLite** | Complex queries, joins, persistent storage | Persistence, flexibility |

#### Decision Matrix:

✅ **Use Redis when**:
- Query hasil same dan frequently accessed
- Need real-time counters atau atomic operations
- Building dashboards atau APIs dengan high traffic
- Working with simple data structures (strings, lists, hashes)

✅ **Use SQLite when**:
- Need complex JOIN operations
- Require persistent storage (data must survive restart)
- Performing ad-hoc analysis dengan dynamic queries
- Need ACID transactions untuk data integrity

### Best Practices

1. **Set TTL on cached data**: `r.expire('key', 3600)` untuk auto-cleanup
2. **Use appropriate data structures**: 
   - Counters → Strings dengan INCR
   - Stats → Hashes dengan HSET/HGETALL
   - Rankings → Sorted Sets dengan ZADD/ZRANGE
3. **Monitor memory usage**: Redis stores everything in RAM
4. **Use pipeline untuk bulk operations**: Reduce network roundtrips
5. **Keep Redis for hot data**: Don't replicate entire database ke Redis

---

## 📊 Data Analisis.ipynb - Detailed Workflow

### Overview

`Data Analisis.ipynb` adalah notebook comprehensive yang mengimplementasikan **hybrid analytics approach** menggunakan Redis dan SQLite untuk sentiment analysis pada 3.6 juta reviews.

### Workflow Steps:

#### 1. **Data Loading & Cleaning** 📥
- Load 3.6M reviews dari CSV (`train.csv`)
- Handle missing values (207 missing titles → dropped)
- Remove duplicates (9,494 duplicate reviews → cleaned)
- Final clean dataset: **3,594,576 reviews**

```python
# Missing values handling
df = df.dropna()

# Duplicate removal
df = df.drop_duplicates(subset=['review'], keep='first')
```

#### 2. **Redis Implementation** ⚡

##### a. Connection & Setup
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
print("Connected to Redis:", r.ping())
```

##### b. Real-time Sentiment Counters
```python
# Initialize counters from dataframe
r.set('sentiment:count:negative', len(df[df['label'] == 1]))
r.set('sentiment:count:positive', len(df[df['label'] == 2]))

# Fast retrieval (< 1ms)
counts = get_sentiment_counts_from_redis(r)
```

##### c. Statistics Caching
```python
# Cache aggregated stats in Redis hash
cache_sentiment_stats(r, {
    'total_reviews': len(df),
    'negative_pct': 49.97,
    'positive_pct': 50.03
})

# Instant retrieval with HGETALL
stats = r.hgetall('sentiment:stats')
```

##### d. Performance Benchmarking
```python
# Compare Redis vs SQLite for same query
# Result: Redis is 100-300x faster! 🚀
```

**Output Example:**
```
⚡ PERFORMANCE BENCHMARK: Redis vs SQLite
============================================================

1️⃣  COUNT QUERY - Negative Reviews
   SQLite: 0.0450s (Result: 1,796,029)
   Redis:  0.0003s (Result: 1,796,029)
   🚀 Redis is 150.0x FASTER!

2️⃣  STATS AGGREGATION - Multiple Fields
   SQLite: 0.0820s
   Redis:  0.0004s
   🚀 Redis is 205.0x FASTER!

3️⃣  CACHE HIT - Repeated Query (100x)
   SQLite (100 queries): 4.5230s
   Redis (100 queries):  0.0271s
   🚀 Redis is 167.0x FASTER!
```

#### 3. **SQLite for Complex Analytics** 💾

##### a. Database Creation & Indexing
```python
# Create SQLite database for persistent storage
conn = sqlite3.connect('uas_sentiment_3.6M.db')
df_working.to_sql('reviews', conn, if_exists='replace')

# Optimize with indexes
conn.execute('CREATE INDEX idx_label ON reviews(label)')
conn.execute('CREATE INDEX idx_length ON reviews(text_length)')
```

##### b. Complex Queries
```python
# SQL untuk advanced analytics
table_41 = pd.read_sql("""
    SELECT 
        label,
        COUNT(*) as total_reviews,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage,
        ROUND(AVG(text_length), 1) as avg_length
    FROM reviews 
    GROUP BY label
""", conn)
```

#### 4. **Business Intelligence Tables** 📈

Notebook generates 4 comprehensive tables:

- **Table 4.1**: Sentiment Distribution (50-50 split confirmed)
- **Table 4.2**: Review Length Analysis (negative reviews 8% longer)
- **Table 4.3**: Top Topics & Controversy Triggers
- **Table 4.4**: Business Recommendations Matrix

#### 5. **Visualization Export** 📊

```python
# Generate 4-panel professional dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Sentiment Distribution Bar Chart
# 2. Percentage Pie Chart  
# 3. Length Distribution Box Plot
# 4. Topic Analysis Heatmap

plt.savefig('UAS_Chapter4_Visualizations.png', dpi=300, bbox_inches='tight')
```

### Key Insights from Analysis:

1. **Balanced Dataset**: 49.97% negative, 50.03% positive → No class imbalance
2. **Length Patterns**: 
   - Negative reviews: avg 420.8 chars (more detailed complaints)
   - Positive reviews: avg 389.5 chars (shorter, more consistent)
3. **Performance**: Redis caching reduces query time by **150-200x**
4. **Top Triggers**: Quality (35% negative), Price (28% negative), Delivery (31% negative)

### When to Use Each Notebook:

| Notebook | Purpose | Best For |
|----------|---------|----------|
| **Sentimen Analysis on Big Data (py-Spark)** | Machine learning training dengan PySpark | Model training, distributed computing, production ML |
| **Data Analisis.ipynb** | Exploratory analysis dengan Redis caching | Ad-hoc analysis, visualization, performance optimization |

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
