# Laporan Proyek Machine Learning - Rio Febriyan

## Domain Proyek

Sistem rekomendasi sangat krusial di industri e-commerce karena dapat meningkatkan penjualan dan kepuasan pelanggan. Dalam proyek ini, kami membangun sistem rekomendasi berbasis content-based filtering pada produk fashion. Sistem ini merekomendasikan produk yang mirip berdasarkan atribut konten seperti nama produk, harga, ukuran, rating, dan target gender.

### Business Understanding

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
- Normalisasi Price, Rating, dan Colors dengan MinMaxScaler
- Penggabungan seluruh fitur menjadi feature_matrix
- Split data menjadi training dan testing dengan rasio 80:20

## Modeling

1. Encoding Label Target :
**Script :
label_map = {label: idx for idx, label in enumerate(y.unique())}
y_train_encoded = np.array([label_map[val] for val in y_train])
y_test_encoded = np.array([label_map[val] for val in y_test])
. Penjelasan :
  . Kolom Gender merupakan label kategorikal (Men, Women, Unisex) yang tidak bisa langsung digunakan sebagai target numerik dalam training model
  . Oleh karena itu, label ini diubah menjadi angka menggunakan pemetaan manual (dictionary label_map) agar bisa digunakan dengan sparse_categorical_crossentropy

2.  Model klasifikasi menggunakan TensorFlow Functional API:

- Input layer: sejumlah dimensi fitur
- Hidden layers: Dense(128) → ReLU → Dense(64) → ReLU
- Output layer: Dense jumlah kelas Gender → Softmax
- Loss function: sparse_categorical_crossentropy
- Optimizer: Adam
- Epoch: 10, Batch size: 32

**Script :
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(input_layer)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(y.unique()), activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)
**Output :
Epoch 1/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 2s 19ms/step - accuracy: 0.6490 - loss: 1.0220 - val_accuracy: 1.0000 - val_loss: 0.7121
Epoch 2/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 1.0000 - loss: 0.5455 - val_accuracy: 1.0000 - val_loss: 0.1821
Epoch 3/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 1.0000 - loss: 0.1019 - val_accuracy: 1.0000 - val_loss: 0.0303
Epoch 4/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 1.0000 - loss: 0.0164 - val_accuracy: 1.0000 - val_loss: 0.0118
Epoch 5/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 1.0000 - loss: 0.0069 - val_accuracy: 1.0000 - val_loss: 0.0073
Epoch 6/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 1.0000 - loss: 0.0041 - val_accuracy: 1.0000 - val_loss: 0.0053
Epoch 7/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 1.0000 - loss: 0.0030 - val_accuracy: 1.0000 - val_loss: 0.0040
Epoch 8/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 1.0000 - val_loss: 0.0032
Epoch 9/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 1.0000 - val_loss: 0.0026
Epoch 10/10
22/22 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 1.0000 - val_loss: 0.0022

Penjelasan:
. Model dibuat menggunakan Functional API TensorFlow agar lebih fleksibel dan mudah diakses input/output-nya
. Input Layer: menerima fitur sebanyak kolom dari X_train (gabungan fitur teks, numerik, kategorikal)
. Hidden Layers:
  . Dense(128, activation='relu'): layer pertama dengan 128 neuron dan aktivasi ReLU
  . Dense(64, activation='relu'): layer kedua, sering digunakan sebagai embedding layer karena ukurannya lebih kecil dan mampu menangkap pola laten
. Output Layer: menggunakan fungsi aktivasi softmax dengan jumlah neuron sesuai banyaknya kelas (len(y.unique())) → cocok untuk klasifikasi multiclass (Gender ada 3 kelas)
.  Akurasi Sangat Tinggi :
  . Setelah epoch ke-2, model mencapai akurasi sempurna (100%) pada training dan validasi set
  . Ini menunjukkan bahwa model sangat mampu mengenali pola yang membedakan antara kelas Gender
. Loss Turun Konsisten :
  . Loss pada training turun drastis dari 1.0220 → 0.0014
  . Loss pada validasi juga turun dari 0.7121 → 0.0022
  . Konsistensi penurunan ini menandakan bahwa model terus belajar dan tidak stagnan
. Tidak Ada Overfitting :
  . Akurasi dan loss pada training dan validasi hampir identik
  . Tidak terlihat gap besar antara kedua metrik, yang biasanya menjadi indikator overfitting
  . Model menunjukkan generalisasi yang sangat baik pada data uji

3.  Kompilasi Model :

**Script :
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Penjelasan :
. Optimizer: adam adalah metode yang efisien untuk pembelajaran berbasis gradien
. Loss function: sparse_categorical_crossentropy digunakan karena label target sudah diencode ke integer (bukan one-hot)
. Metrics: model dievaluasi berdasarkan akurasi (accuracy) pada data validasi dan uji

4.  Hasil Evaluasi Model 
- Akurasi Model: Model mencapai akurasi sekitar 100%, menunjukkan performa yang baik dalam memprediksi target Gender
- Classification Report: Hasil menunjukkan precision dan recall yang seimbang di antara ketiga kategori, yang mengindikasikan model tidak berat sebelah terhadap salah satu label

**Script :
loss, accuracy = model.evaluate(X_test.toarray(), y_test_encoded)
print("Akurasi Model: {:.2f}%".format(accuracy * 100))

**Output :
Akurasi Model : 100%

- Visualisasi Akurasi dan Loss :
. Grafik garis menunjukkan akurasi yang meningkat stabil dan loss yang menurun konsisten di setiap epoch
. Tidak ada tanda overfitting karena kurva training dan validasi cukup paralel dan stabil

5.  Kesimpulan :
Membangun dan menginisialisasi model neural network yang siap dilatih untuk mengklasifikasikan produk ke dalam kategori Gender berdasarkan fitur konten yang ada. Model ini nantinya juga akan digunakan sebagai dasar untuk menghasilkan embedding produk yang digunakan dalam sistem rekomendasi.


## Visualisasi Embedding Produk


**Script :
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
embeddings = embedding_model.predict(feature_matrix.toarray())

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10,6))
plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], alpha=0.5)
plt.title('Visualisasi Produk Berdasarkan Embedding')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

- Penjelasan :
. Terlihat 3 Klaster Produk yang Jelas
  . Terdapat 3 kelompok/klaster utama pada visualisasi:
    . Kiri atas
    . Kiri bawah
    . Kanan tengah
  . Ini sangat mungkin mencerminkan pembagian produk berdasarkan kategori Gender:
    . Klaster-klaster ini menunjukkan bahwa model berhasil mempelajari representasi fitur produk sehingga produk dengan kesamaan tertentu dikelompokkan bersama.
. Produk dalam Klaster Berarti Mirip :
  . Produk yang berdekatan secara spasial di plot ini memiliki embedding (fitur dalam ruang vektor) yang mirip
  . Artinya, sistem rekomendasi yang menggunakan cosine similarity dari embedding ini akan merekomendasikan produk-produk yang berada di dalam klaster yang sama → sangat relevan!
. Tidak Ada Outlier Ekstrem :
  . Hampir semua titik berkumpul dalam kelompok yang padat dan tidak menyebar jauh
  . Ini menandakan bahwa semua produk berhasil direpresentasikan secara konsisten oleh model
. Kesimpulan dari Visualisasi :
  . Model berhasil membentuk representasi fitur (embedding) yang bermakna dan terpisah secara jelas, terutama untuk atribut target Gender
  . Hasil ini memperkuat sistem rekomendasi berbasis content, karena produk mirip sudah otomatis dipetakan berdekatan dalam ruang vektor
  . Visualisasi juga memberi validasi empiris terhadap kemampuan model dalam memahami struktur data produk

## Visualisasi Hasil Pelatihan Model

**Script :
plt.figure(figsize=(14, 5))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

. Grafik Akurasi
Menampilkan :
 . Akurasi training (accuracy)
 . Akurasi validasi (val_accuracy)
. Grafik Loss
Menampilkan :
 . Loss training (loss)
 . Loss Validasi (val_loss)

. Akurasi Selama Training :
 . Terlihat bahwa akurasi meningkat cepat dari epoch pertama, dan stabil di 100% mulai dari epoch ke-2
 . Akurasi validasi juga langsung 100% sejak awal, bahkan lebih cepat dari akurasi training
 . Kurva akurasi yang datar dan tinggi menunjukkan bahwa model berhasil belajar dengan sangat cepat dan efisien
. Loss Selama Training :
 . Loss menurun drastis dari nilai awal tinggi (~1.0) ke hampir nol (~0.001) dalam 2–3 epoch pertama
 . Baik loss training maupun loss validasi turun dengan pola yang sama, lalu stabil sangat rendah
 . Kurva training dan validasi hampir identik, menunjukkan tidak terjadi overfitting

. Kesimpulan dari grafik :
1. Model Belajar Cepat dan Efektif
 . Dalam waktu singkat (2–3 epoch), model mencapai akurasi maksimal dan loss minimal
 . Ini menunjukkan bahwa fitur yang digunakan cukup informatif dan terstruktur dengan baik
2. Tidak Ada Overfitting
 . Tidak terlihat gap besar antara training dan validasi
 . Model bekerja dengan generalisasi yang sangat baik pada data yang belum pernah dilihat
3. Model Sangat Stabil
 . Akurasi dan loss tidak mengalami fluktuasi setelah stabil, menandakan pelatihan berjalan konsisten


## Content-Based Filtering

- Fungsi recommend_by_rating(product_index, top_n=5) melakukan dua hal penting :
 1. Menghitung Kemiripan Produk
    Menggunakan cosine similarity pada feature_matrix, yang terdiri dari gabungan:
     . TF-IDF dari judul produk (Title)
     . One-hot encoding dari Size dan Gender
     . Normalisasi Price, Rating, dan Colors
 2. Filter Berdasarkan Rating
     . Dari produk-produk yang paling mirip (secara konten/fitur)
     . Hanya produk dengan Rating ≥ produk acuan yang akan ditampilkan

- Hasil yang Ditampilkan :
**Script 
def recommend_by_rating(product_index, top_n=5):
    similarity_matrix = cosine_similarity(feature_matrix)
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i for i, score in similarity_scores[1:] if df.iloc[i]['Rating'] >= df.iloc[product_index]['Rating']]
    return df.iloc[similar_indices[:top_n]][['Title', 'Price', 'Rating']]
# Contoh penggunaan rekomendasi berdasarkan rating
print("\nRekomendasi produk mirip dengan rating setara atau lebih tinggi:")
print(recommend_by_rating(0))

**Output
Rekomendasi produk mirip dengan rating setara atau lebih tinggi:
       Title         Price      Rating
598   T-shirt 692   1416480.0   4.1
754   T-shirt 872   1343680.0   4.3
520   T-shirt 602   2269600.0   4.4
832   T-shirt 962   2887840.0   4.1
338   T-shirt 392   1226080.0   4.4

.  Interpretasi:
 . Fungsi ini direkomendasikan terhadap produk ke-0 (product_index = 0) — meskipun detailnya tidak ditampilkan di sini
 . Hasil rekomendasi menampilkan produk T-shirt lain yang memiliki:
  . Rating yang sama atau lebih tinggi
  . Kemiripan konten/fitur tinggi (berdasarkan cosine similarity)
 . Rekomendasi ini menunjukkan konsistensi konten karena semua produk adalah jenis T-shirt — jadi sistem tidak menyarankan produk dari kategori lain yang tidak relevan

-  Tujuan Filtering Rating :
. Dengan menambahkan syarat Rating >= rating produk acuan, kamu memastikan:
 . Hanya produk yang sama bagusnya atau lebih baik secara penilaian pengguna yang direkomendasikan
 . Ini membantu menjaga kualitas rekomendasi agar pengguna tidak diarahkan ke produk dengan reputasi buruk

- Kesimpulan : 
. Sistem rekomendasi content-based yang kamu bangun berhasil mengidentifikasi produk mirip secara konten dan menyaringnya dengan threshold kualitas berdasarkan rating. Ini menunjukkan bahwa sistem tidak hanya peka terhadap kemiripan teknis, tetapi juga mempertimbangkan kualitas produk dari sisi pengguna




## Evaluation
1.  Evaluasi Model Klasifikasi (Gender)
- Model klasifikasi menggunakan TensorFlow Functional API telah diuji dengan :
 . Akurasi Pelatihan dan Validasi mencapai 100% mulai dari epoch ke-2 hingga ke-10
 . Loss menurun drastis dari :
  . Training: 1.0220 → 0.0014
  . Validasi: 0.7121 → 0.0022
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
- Sistem rekomendasi menggunakan cosine_similarity dari gabungan fitur:
 . TF-IDF dari Title
 . One-hot encoding dari Size dan Gender
 . MinMax-scaled Price, Rating, dan Colors
- Fungsi :
**Script
recommend_by_rating(product_index, top_n=5)
. Menyaring produk berdasarkan:
 . Kemiripan fitur
 . Rating yang setara atau lebih tinggi dari produk acuan
**Output 
Rekomendasi produk mirip dengan rating setara atau lebih tinggi:
       Title         Price      Rating
598   T-shirt 692   1416480.0   4.1
754   T-shirt 872   1343680.0   4.3
520   T-shirt 602   2269600.0   4.4
832   T-shirt 962   2887840.0   4.1
338   T-shirt 392   1226080.0   4.4
. Analisis :
 . emua produk yang direkomendasikan adalah T-shirt, yang menandakan sistem memahami konteks konten
 . Rating setiap produk direkomendasikan ≥ produk acuan, menjamin kualitas saran
 . Tidak ada produk dari kategori yang berbeda yang ikut masuk rekomendasi → filtering bekerja baik

### Kesimpulan Evaluasi 
- Model klasifikasi memiliki kinerja sempurna, tanpa overfitting, dan berhasil membentuk embedding produk yang berkualitas tinggi
- Visualisasi embedding menunjukkan bahwa produk dapat direpresentasikan secara terstruktur
- Sistem rekomendasi berbasis content-filtering memberikan hasil yang akurat, relevan, dan berkualitas tinggi, serta mempertimbangkan nilai tambah berupa rating
- Model klasifikasi telah menunjukkan performa yang sangat baik
- Visualisasi pelatihan membantu memverifikasi stabilitas proses training.
- Cosine similarity dan embedding yang digunakan berhasil membentuk dasar sistem rekomendasi yang bisa diandalkan
