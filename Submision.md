# Laporan Proyek Machine Learning - Rio Febriyan

## Project Overview

Sistem rekomendasi sangat krusial di industri e-commerce karena dapat meningkatkan penjualan dan kepuasan pelanggan. Dalam proyek ini, kami membangun sistem rekomendasi berbasis content-based filtering pada produk fashion. Sistem ini merekomendasikan produk yang mirip berdasarkan atribut konten seperti nama produk, harga, ukuran, rating, dan target gender.

## Business Understanding

# Problem Statements
- Bagaimana membangun sistem rekomendasi produk fashion yang mampu memberikan saran berdasarkan kemiripan atribut produk?
- Bagaimana cara memanfaatkan model deep learning untuk memahami representasi produk dan melakukan klasifikasi?
- Bagaimana mengevaluasi performa dari sistem rekomendasi yang dibangun ?

### Goals
- Menghasilkan sistem rekomendasi produk berbasis content-based filtering.
- Melatih model klasifikasi neural network menggunakan TensorFlow.
- Memvisualisasikan embedding produk dan mengevaluasi model berdasarkan metrik klasifikasi dan efektivitas rekomendasi.

### Solution statements
    - Melakukan preprocessing teks, kategorikal, dan numerik menjadi vektor numerik.
    - Menggunakan arsitektur TensorFlow (Functional API) untuk klasifikasi dan pembentukan embedding.
    - Menggunakan cosine similarity dari embedding gabungan untuk sistem rekomendasi.

## Data Understanding
Dataset terdiri dari 867 produk fashion dengan fitur:
- Title: Nama produk
- Price: Harga dalam satuan rupiah
- Rating: Skor ulasan pengguna
- Colors: Jumlah variasi warna
- Size: Ukuran produk (S, M, L, XL, dst)
- Gender: Target pengguna (Men, Women, Unisex)
Data ini tidak memiliki nilai kosong atau duplikat. Visualisasi sederhana menunjukkan distribusi rating dan gender yang cukup beragam.
data ini di peroleh dari crawling pada suatu website dicoding berikut link Url nya : https://fashion-studio.dicoding.dev

### Variabel-variabel pada products dataset adalah sebagai berikut:
- Title : Jenis barang / nama barang.
- Price : Harga barang dalam Rupiah.
- Rating : Reting suatu barang
- Colors : varian warna barang tersebut
- Size : Ukuran Barang
- Gender : Peruntukan barang bisa di pakai oleh laki-laki , Perumpuan ayau keduanya bisa

## Data Preparation

- TF-IDF untuk Title
- One-Hot Encoding untuk Size dan Gender
- Mengubah Size & Gender menjadi data Numerik
  . Kolom Gender merupakan label kategorikal (Men, Women, Unisex) yang tidak bisa langsung digunakan sebagai target numerik dalam training model
  . Oleh karena itu, label ini diubah menjadi angka menggunakan pemetaan manual (dictionary label_map) agar bisa digunakan dengan sparse_categorical_crossentropy
- Normalisasi Price, Rating, dan Colors dengan MinMaxScaler
- Penggabungan seluruh fitur menjadi feature_matrix
- Split data menjadi training dan testing dengan rasio 80:20


## Modeling

###  1. Proses Pelatihan Model Klasifikasi Gender

Model neural network dibangun dengan dua hidden layer dan satu output layer untuk klasifikasi label `Gender`. Arsitektur ini dilatih menggunakan optimizer Adam dan fungsi loss sparse categorical crossentropy.

Training dilakukan selama 10 epoch, dan model mencapai akurasi sempurna (100%) sejak epoch ke-2. Hal ini menunjukkan bahwa model sangat cepat belajar dari data dan mampu memprediksi target dengan sangat akurat.

> **Kesimpulan**: Model mampu mempelajari data dengan sangat baik dan mencapai performa maksimal tanpa overfitting.

---

###  2. Evaluasi Akhir Model

Model dievaluasi menggunakan metrik akurasi dan classification report (precision, recall, f1-score). Semua metrik bernilai sempurna (1.00) pada data uji.

> Ini membuktikan bahwa model memiliki generalisasi yang baik dan tidak bias terhadap salah satu kelas Gender.

---

###  3. Visualisasi Embedding Produk

Embedding dari layer tersembunyi divisualisasikan dengan PCA ke dalam ruang 2 dimensi. Hasil visualisasi menunjukkan pembentukan 3 klaster yang jelas, yang sangat mungkin mewakili kategori Gender.

> Produk yang berdekatan secara visual memiliki fitur yang mirip, sehingga sangat mendukung efektivitas sistem rekomendasi berbasis konten.

---

###  4. Visualisasi Akurasi dan Loss Selama Training

Grafik menunjukkan peningkatan akurasi yang sangat cepat dan penurunan loss yang drastis pada training maupun validasi. Tidak terdapat perbedaan signifikan antara kurva training dan validasi.

> Hal ini menunjukkan model stabil, tidak overfit, dan mampu melakukan generalisasi terhadap data yang tidak dilihat sebelumnya.

---

### 5.  Sistem Rekomendasi Content-Based Filtering

- Konsep Dasar
Sistem ini merekomendasikan produk berdasarkan kemiripan fitur dengan produk yang sedang dilihat/dipilih oleh pengguna. Pendekatan Content-Based Filtering memanfaatkan informasi dari produk itu sendiri (misalnya: judul, harga, rating, warna, ukuran, dll) untuk menghitung kemiripan antar produk.

- Penjelasan Fungsi recommend_by_rating()
  ```
  python
  def recommend_by_rating(product_index, top_n=5):
    similarity_matrix = cosine_similarity(feature_matrix)
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i for i, score in similarity_scores[1:] if df.iloc[i]['Rating'] >= df.iloc[product_index]['Rating']]
    return df.iloc[similar_indices[:top_n]][['Title', 'Price', 'Rating']]
print("\nRekomendasi produk mirip dengan rating setara atau lebih tinggi:")
print(recommend_by_rating(0))
```

```
Output :

Rekomendasi produk mirip dengan rating setara atau lebih tinggi:
           Title      Price  Rating
598  T-shirt 692  1416480.0     4.1
754  T-shirt 872  1343680.0     4.3
520  T-shirt 602  2260960.0     4.4
832  T-shirt 962  2887840.0     4.1
338  T-shirt 392  1226080.0     4.4

```
- Langkah-langkah:

 1. Menghitung Kemiripan
   . cosine_similarity(feature_matrix) digunakan untuk menghitung skor kemiripan antar produk berdasarkan representasi fitur numerik (embedding atau vektor numerik dari fitur).

2. Mengambil Produk yang Mirip
   . Produk diurutkan berdasarkan skor kemiripan tertinggi dengan produk referensi (product_index)

3. Filtering Berdasarkan Rating
   . Hanya produk yang memiliki rating sama atau lebih tinggi dari produk yang dimaksud yang akan diambil sebagai rekomendasi

4. Mengambil Top-N Rekomendasi
   . Fungsi mengembalikan 5 produk paling mirip dan dengan rating ≥ produk awal.

- Fitur yang Digunakan
 . feature_matrix: representasi fitur gabungan dari produk, kemungkinan hasil encoding dari kolom seperti
  . Judul
  . Warna
  . Ukuran
  . Gender
  . Harga ( Price ) dan Rating

- Kelebihan
 . Tidak bergantung pada data pengguna atau interaksi historis (cocok untuk cold start)
 . Rekomendasi tetap relevan meskipun pengguna baru

- Keterbatasan
 . Tidak bisa menangkap selera pengguna jika hanya berdasarkan satu produk
 . Terbatas pada produk yang mirip secara fitur — tidak bisa merekomendasikan hal “berbeda tapi disukai pengguna lain”

- Kesimpulan
Rekomendasi terbukti relevan karena produk yang disarankan berasal dari kategori yang sama dan memiliki kualitas (rating) tinggi
Sistem rekomendasi yang dibangun menggunakan pendekatan Content-Based Filtering, di mana kemiripan antar produk dihitung menggunakan Cosine Similarity dari representasi fitur produk. Model ini kemudian menyaring produk-produk serupa yang memiliki rating setara atau lebih tinggi, dan mengembalikan beberapa produk teratas sebagai hasil rekomendasi

---


## Evaluation
1.  Evaluasi Model Klasifikasi (Gender)
- Model klasifikasi menggunakan TensorFlow Functional API telah diuji dengan :
 . Akurasi Pelatihan dan Validasi mencapai 100% mulai dari epoch ke-2 hingga ke-10
 . Loss menurun drastis dari :
  . Training: 1.0306 → 0.0012
  . Validasi: 0.7369 → 0.0018
- Kurva akurasi dan loss pada pelatihan dan validasi nyaris identik dan stabil
 . Grafik Akurasi menunjukkan peningkatan pesat dan stabil
 . Grafik Loss menampilkan penurunan tajam dan stabil di titik rendah
 . Tidak ditemukan indikasi overfitting
-  Classification Report:
 . Meskipun tidak ditampilkan detailnya, disebutkan bahwa precision dan recall seimbang di seluruh kelas (Men, Women, Unisex)
 . Ini menunjukkan model tidak bias terhadap satu kelas, dan dapat dipercaya untuk klasifikasi berbasis konten produk
- Kesimpulan Evaluasi Model:
 . Model sangat baik dalam mempelajari fitur produk untuk klasifikasi Gender
 . Performa sangat tinggi dan stabil
 . Sangat cocok untuk dijadikan dasar pembentukan embedding untuk sistem rekomendasi

2. Evaluasi Visualisasi Embedding Produk
- Embedding dari layer tersembunyi (Dense(64)) diproyeksikan menggunakan PCA menjadi 2D
Hasil Visualisasi :
 . Terbentuk 3 klaster yang jelas
 . Titik-titik dalam klaster berdekatan dan padat, menandakan representasi yang konsisten
 . Tidak ditemukan outlier atau titik yang menyimpang secara ekstrem
- Kesimpulan Visualisasi:
 . Klaster kemungkinan besar mewakili pembagian Gender
 . Embedding sangat representatif terhadap konten produk
 . Membuktikan model memahami struktur data dengan sangat baik, dan embedding-nya dapat diandalkan untuk sistem rekomendasi berbasis vektor

3. Evaluasi Sistem Rekomendasi Content-Based Filtering
- Evaluasi Sistem Rekomendasi dengan NDCG@K

Untuk mengevaluasi kualitas urutan rekomendasi, digunakan metrik Normalized Discounted Cumulative Gain (NDCG@5). Produk dianggap relevan jika memiliki rating ≥ 4.0.

> Nilai NDCG@5 mencapai 0.988, yang menunjukkan bahwa urutan rekomendasi sangat relevan dengan kebutuhan pengguna.


### Kesimpulan Evaluasi 
- Model klasifikasi memiliki kinerja sempurna, tanpa overfitting, dan berhasil membentuk embedding produk yang berkualitas tinggi
- Visualisasi embedding menunjukkan bahwa produk dapat direpresentasikan secara terstruktur
- Sistem rekomendasi berbasis content-filtering memberikan hasil yang akurat, relevan, dan berkualitas tinggi, serta mempertimbangkan nilai tambah berupa rating
- Model klasifikasi telah menunjukkan performa yang sangat baik
- Visualisasi pelatihan membantu memverifikasi stabilitas proses training.
- Cosine similarity dan embedding yang digunakan berhasil membentuk dasar sistem rekomendasi yang bisa diandalkan
-  Sistem rekomendasi berhasil memberikan saran yang relevan, berkualitas, dan didukung oleh evaluasi kuantitatif melalui NDCG@5.

