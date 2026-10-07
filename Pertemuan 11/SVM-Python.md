<h1>Mata kuliah : Modern Prediksi dan Machine Learning</h1>
<h3>Pertemuan 11: Support Vector Machine</h3>

<tt>Author: Muhammad Fadil</tt><br>
<tt>Update: 2026-10-07</tt>

# Klasifikasi Kanker Payudara dengan Support Vector Machine (Python)

Program ini menerapkan **Support Vector Machine (SVM)** untuk mengklasifikasikan tumor payudara sebagai **ganas (malignant)** atau **jinak (benign)** berdasarkan 30 ukuran sel dari citra biopsi. Notebook `Pertemuan 11 SVM.ipynb` membahas pentingnya standarisasi, perbandingan kernel, tuning `C` dan `gamma` dengan _grid search_, evaluasi (confusion matrix, ROC-AUC), perbandingan dengan model lain, visualisasi batas keputusan, dan prediksi data baru.

> Semua angka di dokumen ini diambil dari output yang tersimpan di notebook (`random_state=42`). Pada versi scikit-learn yang berbeda, angka bisa sedikit berbeda.

---

## Daftar Isi

1. [Tujuan](#1-tujuan)
2. [Dataset](#2-dataset)
3. [Persyaratan & Cara Menjalankan](#3-persyaratan--cara-menjalankan)
4. [Konsep Singkat](#4-konsep-singkat)
5. [Penjelasan Kode & Interpretasi Hasil](#5-penjelasan-kode--interpretasi-hasil)
6. [Ringkasan Hasil](#6-ringkasan-hasil)
7. [Kesimpulan](#7-kesimpulan)

---

## 1. Tujuan

- Membangun model SVM untuk mendiagnosis tumor payudara (ganas vs jinak).
- Menunjukkan bahwa **standarisasi fitur** sangat penting bagi SVM.
- Membandingkan empat kernel (`linear`, `poly`, `rbf`, `sigmoid`).
- Menentukan nilai `C` dan `gamma` terbaik dengan grid search dan validasi silang.
- Mengevaluasi dengan akurasi, confusion matrix, precision/recall, dan ROC-AUC.
- Membandingkan SVM dengan Regresi Logistik dan Random Forest.

## 2. Dataset

**Wisconsin Diagnostic Breast Cancer** dari `sklearn.datasets.load_breast_cancer`: **569 pasien** dan **30 fitur numerik**. Fitur merupakan 10 ukuran inti sel (radius, tekstur, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), masing-masing dalam tiga versi: rata-rata (_mean_), galat baku (_error_), dan nilai terburuk (_worst_).

| Kelas             | Jumlah | Proporsi |
| ----------------- | ------ | -------- |
| benign (jinak)    | 357    | 62,7%    |
| malignant (ganas) | 212    | 37,3%    |

**Penting:** di scikit-learn kode asli target adalah `0 = malignant`, `1 = benign`. Notebook ini **membalik** pengodean menjadi `y = (d.target == 0).astype(int)` sehingga **`1 = malignant`** menjadi kelas positif. Akibatnya recall, precision, dan AUC dibaca untuk kelas **ganas**.

## 3. Persyaratan & Cara Menjalankan

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Jalankan seluruh sel dari atas ke bawah (**Run All**). Notebook menghasilkan berkas:

| Berkas              | Isi                                                                             |
| ------------------- | ------------------------------------------------------------------------------- |
| `breast_cancer.csv` | Data lengkap (30 fitur + `diagnosis`), diekspor agar dapat dipakai di program R |
| `heat.png`          | Heatmap akurasi CV untuk kombinasi `C` dan `gamma`                              |
| `roc.png`           | Kurva ROC pada data uji                                                         |
| `boundary.png`      | Ilustrasi batas keputusan SVM dengan 2 variabel                                 |

## 4. Konsep Singkat

**SVM** mencari _hyperplane_ pemisah dua kelas dengan **margin** (jarak ke titik terdekat) sebesar mungkin. Titik-titik yang menentukan batas disebut **_support vector_**; titik lain tidak memengaruhi model. Untuk data yang tidak terpisah linear, SVM memakai **kernel** yang memetakan data ke ruang berdimensi lebih tinggi tanpa menghitung pemetaan itu secara eksplisit.

Dua hiperparameter utama (kernel RBF):

| Hiperparameter | Fungsi                                                     | Jika terlalu kecil                                     | Jika terlalu besar                                                 |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| `C`            | Penalti kesalahan klasifikasi (kekakuan margin)            | Margin lebar, model terlalu sederhana (_underfitting_) | Margin sempit, model menghafal data (_overfitting_)                |
| `gamma`        | Jangkauan pengaruh satu titik: K(x, x′) = exp(−γ‖x − x′‖²) | Batas sangat halus, hampir linear                      | Batas sangat berliku, tiap titik berpengaruh lokal (_overfitting_) |

SVM berbasis **jarak**, sehingga fitur dengan skala besar mendominasi perhitungan. Karena itu standarisasi (rata-rata 0, simpangan baku 1) hampir selalu wajib. `gamma="scale"` setara dengan 1 / (jumlah fitur × varians data); setelah standarisasi nilainya ≈ 1/30 ≈ 0,033.

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Import library (Sel 1)

Memuat `numpy`, `pandas`, `matplotlib`, dataset `load_breast_cancer`, alat pembagian data dan validasi (`train_test_split`, `cross_val_score`, `GridSearchCV`, `StratifiedKFold`), `StandardScaler` dan `make_pipeline`, tiga model (`SVC`, `LogisticRegression`, `RandomForestClassifier`), serta metrik (`accuracy_score`, `confusion_matrix`, `classification_report`, `roc_auc_score`, `roc_curve`).

### 5.2 Memuat dan mengeksplorasi data (Sel 2)

```python
d = load_breast_cancer(as_frame=True)
X = d.data
y = (d.target == 0).astype(int)    # 1 = malignant, 0 = benign
print(X.shape)
print(X[["mean radius", "mean texture", "mean area", "mean smoothness"]].describe().round(2).T)
```

**Hasil:** `(569, 30)`, serta statistik empat fitur contoh:

| Fitur           | Rata-rata | Std    | Min    | Median | Maks    |
| --------------- | --------- | ------ | ------ | ------ | ------- |
| mean radius     | 14,13     | 3,52   | 6,98   | 13,37  | 28,11   |
| mean texture    | 19,29     | 4,30   | 9,71   | 18,84  | 39,28   |
| mean area       | 654,89    | 351,91 | 143,50 | 551,10 | 2501,00 |
| mean smoothness | 0,10      | 0,01   | 0,05   | 0,10   | 0,16    |

**Interpretasi:**

- Data lengkap (569 baris, tanpa nilai hilang) dan seluruhnya numerik.
- Kelas tidak seimbang ringan (63% jinak : 37% ganas). Akurasi masih layak dipakai, tetapi **recall kelas ganas** perlu diperhatikan.
- **Skala fitur sangat berbeda:** `mean area` (rata-rata 655, std 352) berbeda sekitar 5 digit dari `mean smoothness` (0,10, std 0,01). Tanpa standarisasi, `mean area` akan mendominasi jarak antar titik. Inilah alasan standarisasi diuji pada 5.4.
- Sel ini juga mengekspor data ke `breast_cancer.csv` (nama kolom diganti spasi → garis bawah dan label `diagnosis` berupa teks).

### 5.3 Pembagian data latih dan uji (Sel 3)

```python
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

**Hasil:** `455 114`, yaitu 455 data latih dan 114 data uji.

**Interpretasi:** pembagian 80/20 **berstrata** (`stratify=y`) menjaga proporsi ganas : jinak sama di kedua bagian. Data uji berisi 72 jinak dan 42 ganas (terlihat dari confusion matrix pada 5.7).

### 5.4 Pentingnya standarisasi (Sel 4)

```python
svm_raw = SVC(kernel="rbf", C=1, gamma="scale").fit(Xtr, ytr)
svm_std = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale")).fit(Xtr, ytr)
```

`make_pipeline` menggabungkan `StandardScaler` dan `SVC` sehingga skala dipelajari **hanya dari data latih** lalu diterapkan ke data uji (mencegah kebocoran data).

**Hasil:**

| Model                  | Akurasi uji          |
| ---------------------- | -------------------- |
| SVM RBF tanpa scaling  | 0,9035 (103/114)     |
| SVM RBF dengan scaling | **0,9737** (111/114) |

**Interpretasi:** standarisasi saja menaikkan akurasi **7 poin persentase** (dari 90,4% menjadi 97,4%), tanpa mengubah model atau parameter lain. Ini menegaskan bahwa pada SVM, **pra-pemrosesan sama pentingnya dengan pemilihan algoritma**. Seluruh langkah berikutnya selalu memakai scaling.

### 5.5 Perbandingan kernel (Sel 5)

```python
for k in ["linear", "poly", "rbf", "sigmoid"]:
    m = make_pipeline(StandardScaler(), SVC(kernel=k, C=1, gamma="scale")).fit(Xtr, ytr)
    n_sv = m.named_steps["svc"].n_support_.sum()
```

**Hasil:**

| Kernel  | Akurasi uji          | Jumlah support vector |
| ------- | -------------------- | --------------------- |
| linear  | 0,9649 (110/114)     | 38                    |
| poly    | 0,8860 (101/114)     | 149                   |
| **rbf** | **0,9737** (111/114) | 107                   |
| sigmoid | 0,9474 (108/114)     | 70                    |

**Interpretasi:**

- **RBF terbaik** (97,4%), tetapi **linear hampir sama baik** (96,5%) dengan hanya **38 support vector**. Hal ini menandakan data hampir terpisah secara linear, sehingga model sederhana pun sudah memadai.
- **Poly (derajat 3 bawaan) paling buruk** (88,6%) dengan SV terbanyak (149). Kernel polinomial pada pengaturan bawaan terlalu fleksibel dan tidak cocok dengan data ini.
- **Jumlah support vector** adalah ukuran kompleksitas model: semakin sedikit SV, semakin sederhana batas keputusan. Linear (38) jauh lebih ringkas daripada poly (149).
- Selisih akurasi RBF vs linear hanya 1 sampel dari 114, jadi belum tentu bermakna.

### 5.6 Tuning `C` dan `gamma` (Sel 6)

```python
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.0001, 0.001, 0.01, 0.1, 1]}
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
gs.fit(Xtr, ytr)
```

Grid search menguji 4 × 5 = **20 kombinasi** dengan CV 5-fold berstrata (100 pelatihan) pada **data latih saja**. Prefiks `svc__` menunjuk parameter di dalam _pipeline_. Karena scaling berada di dalam pipeline, ia dihitung ulang di tiap lipatan sehingga tidak ada kebocoran.

**Hasil:** `{'svc__C': 10, 'svc__gamma': 0.01}` dengan akurasi CV **0,9758**.

Akurasi CV tiap kombinasi (baris = `C`, kolom = `gamma`):

| C \ gamma | 0,0001 | 0,001  | 0,01       | 0,1    | 1      |
| --------- | ------ | ------ | ---------- | ------ | ------ |
| 0,1       | 0,6264 | 0,7165 | 0,9451     | 0,9319 | 0,6264 |
| 1         | 0,7363 | 0,9495 | 0,9648     | 0,9582 | 0,6308 |
| **10**    | 0,9495 | 0,9648 | **0,9758** | 0,9582 | 0,6374 |
| 100       | 0,9692 | 0,9736 | 0,9692     | 0,9560 | 0,6374 |

**Interpretasi:**

- **Nilai 0,6264 adalah akurasi "tebak kelas mayoritas":** 285 dari 455 data latih adalah jinak (285/455 = 0,6264). Dua pojok tabel mencapai angka ini, tetapi dengan penyebab berlawanan:
  - **Pojok kiri atas** (`C` = 0,1, `gamma` = 0,0001): model **terlalu sederhana (underfitting)**, penalti kecil dan batas sangat halus sehingga semua data ditebak jinak.
  - **Kolom `gamma` = 1** (semua `C`, 0,63–0,64): model **overfitting parah**, tiap titik hanya memengaruhi tetangga sangat dekat sehingga model menghafal data latih dan gagal pada data lipatan lain.
- **Ada punggung diagonal** dari kiri bawah ke kanan atas: `C` besar cocok dengan `gamma` kecil, dan `C` kecil cocok dengan `gamma` lebih besar. Keduanya saling mengimbangi, sehingga banyak kombinasi di punggung ini bagus (0,96–0,976).
- Kombinasi terbaik (`C` = 10, `gamma` = 0,01) berada di **bagian dalam** grid, bukan di tepi. Jadi rentang pencarian sudah tepat.
- Beberapa kombinasi sangat dekat dengan yang terbaik (mis. `C` = 100, `gamma` = 0,001: 0,9736). Selisihnya hanya **1 sampel** dari 455 (0,0022), jadi tidak ada satu kombinasi tunggal yang pasti "terbaik"; sekelompok parameter pada punggung diagonal sama-sama baik.
- Peta panasnya divisualisasikan di `heat.png` (Sel 9a).

### 5.7 Model final dan evaluasi pada data uji (Sel 7)

```python
final = make_pipeline(StandardScaler(),
        SVC(kernel="rbf", C=best["svc__C"], gamma=best["svc__gamma"],
            probability=True, random_state=42)).fit(Xtr, ytr)
pred  = final.predict(Xte)
proba = final.predict_proba(Xte)[:, 1]
```

`probability=True` mengaktifkan estimasi probabilitas (skala Platt) yang diperlukan untuk kurva ROC dan AUC.

**Hasil:** akurasi uji **0,9825** (112/114).

**Confusion matrix (baris = aktual, kolom = prediksi):**

| Aktual \ Prediksi | benign | malignant |
| ----------------- | ------ | --------- |
| **benign**        | 72     | 0         |
| **malignant**     | 2      | 40        |

**Classification report:**

| Kelas     | Precision | Recall | F1        | Support |
| --------- | --------- | ------ | --------- | ------- |
| benign    | 0,973     | 1,000  | 0,986     | 72      |
| malignant | 1,000     | 0,952  | 0,976     | 42      |
| accuracy  |           |        | **0,982** | 114     |
| macro avg | 0,986     | 0,976  | 0,981     | 114     |

**AUC = 0,996.** Jumlah support vector: **[29, 27]** (total 56) dari 455 data latih.

**Interpretasi:**

- Hanya **2 dari 114** pasien yang salah diklasifikasikan, dan keduanya **tumor ganas yang diprediksi jinak (false negative)**. Tidak ada tumor jinak yang salah dicap ganas.
- Precision ganas = 1,000 (setiap prediksi "ganas" benar), tetapi **recall ganas = 0,952**: 2 dari 42 kasus ganas terlewat. Dalam konteks diagnosis, _false negative_ (kanker tidak terdeteksi) umumnya jauh lebih berbahaya daripada _false positive_, jadi recall ganas adalah metrik yang paling perlu diperhatikan.
- **AUC 0,996** (maks 1,0) menunjukkan model hampir sempurna dalam **memeringkat** pasien: probabilitas kasus ganas hampir selalu lebih tinggi daripada kasus jinak. Karena AUC tidak bergantung pada ambang, ini mengindikasikan bahwa dua kasus yang terlewat mungkin dapat ditangkap dengan **menurunkan ambang keputusan** (mis. di bawah 0,5), dengan konsekuensi lebih banyak _false positive_.
- Model sangat **ringkas**: hanya 56 support vector (12,3% dari data latih) yang menentukan batas keputusan. Titik lainnya tidak berpengaruh.
- Akurasi uji (98,25%) sedikit di atas akurasi CV (97,58%), selisih yang setara 1–2 sampel pada data uji sekecil 114. Selisih dengan SVM RBF awal (97,37%) pun hanya **1 pasien**, jadi manfaat tuning di sini kecil.
- Sel ini menampilkan `FutureWarning`: parameter `probability` di `SVC` ditandai usang (_deprecated_) mulai scikit-learn 1.9 dan akan dihapus di 1.11; penggantinya adalah `CalibratedClassifierCV(SVC(), ensemble=False)`. Kode masih berjalan normal, tetapi perlu disesuaikan di versi mendatang.

### 5.8 Perbandingan dengan model lain, CV 10-fold (Sel 8)

```python
skf = StratifiedKFold(10, shuffle=True, random_state=42)
cross_val_score(m, X, y, cv=skf)    # empat model, seluruh 569 data
```

**Hasil (akurasi rata-rata ± simpangan baku antar lipatan):**

| Model                     | Akurasi CV          |
| ------------------------- | ------------------- |
| **SVM RBF (tuned)**       | **0,9772 ± 0,0193** |
| Regresi Logistik          | 0,9754 ± 0,0195     |
| SVM Linear                | 0,9719 ± 0,0251     |
| Random Forest (500 pohon) | 0,9526 ± 0,0368     |

**Interpretasi:**

- Ketiga model berbasis batas linear/halus (SVM RBF, Regresi Logistik, SVM Linear) hampir **identik** (97,2–97,7%), dengan selisih jauh lebih kecil dari simpangan baku (±0,02). Perbedaannya **tidak bermakna** secara statistik.
- **Regresi Logistik, model sederhana, setara dengan SVM RBF**. Ini selaras dengan temuan kernel (linear hampir sama dengan RBF): data ini hampir terpisah linear, jadi model kompleks tidak memberi keuntungan besar.
- **Random Forest paling rendah** (95,3%) dan paling tidak stabil (±0,037). Pohon keputusan kurang efisien dibanding model berbasis jarak/linear pada 30 fitur numerik yang saling berkorelasi tinggi dan telah terstandarisasi.
- **Catatan keadilan:** `C` dan `gamma` SVM RBF dipilih dari data latih, yang merupakan bagian dari 569 data pada CV ini. Akibatnya SVM RBF sedikit diuntungkan (kebocoran ringan); keunggulan 0,0018 atas Regresi Logistik sebaiknya tidak dianggap nyata.

### 5.9 Visualisasi (Sel 9, 10, 11)

**(a) Heatmap akurasi CV (`heat.png`):** menggambar tabel 5.6 sebagai peta warna (biru tua = akurasi tinggi, `vmin = 0,6`, `vmax = 1,0`) beserta angkanya. Wilayah gelap membentuk pita diagonal, sedangkan kolom `gamma` = 1 dan pojok kiri atas tampak pucat.

**(b) Kurva ROC (`roc.png`):** memplot _True Positive Rate_ terhadap _False Positive Rate_ pada berbagai ambang untuk data uji, dengan garis diagonal putus-putus sebagai tebakan acak. Kurva SVM menempel ke pojok kiri atas dengan **AUC = 0,996**.

**(c) Batas keputusan 2 variabel (`boundary.png`):**

```python
cols = ["mean radius", "mean texture"]
m2 = SVC(kernel="rbf", C=1, gamma="scale").fit(Z, y)
```

Ilustrasi dua dimensi menggunakan hanya `mean radius` dan `mean texture` (distandarisasi). Garis hitam penuh adalah batas keputusan (_decision function_ = 0), garis putus-putus adalah margin (±1), dan lingkaran hitam menandai _support vector_.

**Hasil:** `SV 2 variabel: 157 dari 569 | akurasi latih: 0.9069`

**Interpretasi:**

- Dengan hanya 2 variabel, model memerlukan **157 support vector (27,6%)**, jauh lebih banyak daripada 56 SV (12,3%) pada model 30 variabel. Banyak titik berada di dalam atau dekat margin karena kelas tidak terpisah rapi pada 2 dimensi.
- Akurasi hanya **90,7%** (dan ini akurasi _latih_ pada seluruh 569 data), jauh di bawah 98% dari model 30 variabel. Artinya dua variabel saja **tidak cukup**; informasi pembeda tersebar pada banyak fitur.
- Grafik ini murni **ilustrasi** untuk memahami margin dan support vector. Model sebenarnya tidak dapat divisualisasikan langsung karena berada di ruang 30 dimensi.

### 5.10 Prediksi pasien (Sel 12)

```python
baru = Xte.iloc[:3]
final.predict(baru); final.predict_proba(baru)[:, 1]
```

**Hasil:**

| Pasien | Aktual    | Prediksi  | P(malignant) |
| ------ | --------- | --------- | ------------ |
| 1      | benign    | benign    | 0,001        |
| 2      | malignant | malignant | 1,000        |
| 3      | benign    | benign    | 0,045        |

**Interpretasi:**

- Ketiga prediksi **benar**. Pasien 1 dan 2 memiliki probabilitas sangat ekstrem (0,001 dan 1,000), artinya model sangat yakin. Pasien 3 (0,045) tetap jinak, tetapi sedikit lebih dekat ke batas dibandingkan pasien 1.
- Probabilitas SVM berasal dari **kalibrasi Platt** (regresi logistik di atas skor SVM), bukan probabilitas bawaan model, sehingga tidak boleh dianggap sebagai risiko klinis yang pasti.
- "Pasien baru" di sini sebenarnya tiga baris pertama data uji, bukan data yang benar-benar baru.

## 6. Ringkasan Hasil

| Ukuran                                 | Nilai                                                                 |
| -------------------------------------- | --------------------------------------------------------------------- |
| Data latih / uji                       | 455 / 114 (berstrata)                                                 |
| SVM RBF tanpa scaling → dengan scaling | 90,35% → 97,37%                                                       |
| Kernel terbaik (bawaan, scaling)       | RBF 97,37% (linear 96,49%, sigmoid 94,74%, poly 88,60%)               |
| Parameter terbaik (grid search)        | `C = 10`, `gamma = 0,01`                                              |
| Akurasi CV 5-fold (data latih)         | 97,58%                                                                |
| **Akurasi data uji (model final)**     | **98,25%** (112/114)                                                  |
| Recall ganas / precision ganas         | 0,952 / 1,000                                                         |
| AUC data uji                           | 0,996                                                                 |
| Support vector                         | 56 dari 455 (12,3%)                                                   |
| CV 10-fold seluruh data                | SVM RBF 97,72% · Reg. Logistik 97,54% · SVM Linear 97,19% · RF 95,26% |

## 7. Kesimpulan

1. **Standarisasi fitur wajib untuk SVM:** tanpa scaling akurasi 90,4%, dengan scaling 97,4%.
2. Model final **SVM RBF (`C`=10, `gamma`=0,01)** mencapai akurasi uji **98,25%**, AUC **0,996**, dan hanya 2 kesalahan, keduanya _false negative_ (kasus ganas terlewat).
3. Data ini **hampir terpisah secara linear**: SVM linear dan Regresi Logistik sama baiknya dengan SVM RBF (selisih < 1 poin, di bawah ketidakpastian CV). Model yang lebih sederhana sudah memadai.
4. Hubungan `C` dan `gamma` membentuk punggung diagonal: `gamma` terlalu besar menyebabkan _overfitting_ dan akurasi runtuh ke tebakan mayoritas (62,6%), sedangkan `C` dan `gamma` terlalu kecil menyebabkan _underfitting_.
5. Untuk diagnosis, perhatikan **recall kelas ganas**. Menurunkan ambang keputusan layak dipertimbangkan, dengan konsekuensi lebih banyak _false positive_.

## Referensi

- Fadil, M., Islamiyati, A., & Thamrin, S. A. (2025). Classification of nutritional status in toddlers using the support vector machine method. _Communications in Mathematical Biology and Neuroscience_, 2025, Article ID 57. https://doi.org/10.28919/cmbn/9126
- Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. _Advances in Large Margin Classifiers_, 61–74.
- Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. _ACM Transactions on Intelligent Systems and Technology_, 2(3), 27.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning (2nd ed.)_. Springer.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). _An Introduction to Statistical Learning (2nd ed.)_. Springer.
- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. _Proceedings of IS&T/SPIE_, 1905, 861–870.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_, 12, 2825–2830.
- Meyer, D., et al. _e1071: Misc Functions of the Department of Statistics, Probability Theory Group (TU Wien)_. R package.
