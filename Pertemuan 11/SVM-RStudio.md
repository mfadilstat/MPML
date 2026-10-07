<h1>Mata kuliah : Modern Prediksi dan Machine Learning</h1>
<h3>Pertemuan 11: Support Vector Machine</h3>

<tt>Author: Muhammad Fadil</tt><br>
<tt>Update: 2026-10-07</tt>

# Klasifikasi Kanker Payudara dengan Support Vector Machine (R & RStudio)

Program ini menerapkan **Support Vector Machine (SVM)** di R menggunakan paket `e1071` untuk mengklasifikasikan tumor payudara sebagai **ganas (malignant)** atau **jinak (benign)** dari 30 ukuran sel. Skrip `SVM_R.R` membahas pentingnya _scaling_, perbandingan kernel, tuning `cost` dan `gamma`, evaluasi (confusion matrix, ROC-AUC), validasi silang 10-fold dengan `caret`, ilustrasi batas keputusan 2 variabel, dan prediksi data baru.

Skrip ini adalah padanan versi R dari notebook Python `Pertemuan11_SVM.ipynb` dan memakai berkas `breast_cancer.csv` yang diekspor notebook tersebut.

> Semua angka di dokumen ini diperoleh dari menjalankan skrip apa adanya dengan `set.seed(42)`. Pada versi R atau paket yang berbeda, pembagian data dan angka bisa sedikit bergeser.

---

## Daftar Isi

1. [Tujuan](#1-tujuan)
2. [Dataset](#2-dataset)
3. [Persyaratan & Cara Menjalankan](#3-persyaratan--cara-menjalankan)
4. [Konsep Singkat](#4-konsep-singkat)
5. [Penjelasan Kode & Interpretasi Hasil](#5-penjelasan-kode--interpretasi-hasil)
6. [Ringkasan Hasil](#6-ringkasan-hasil)
7. [Perbandingan dengan Versi Python](#7-perbandingan-dengan-versi-python)
8. [Kesimpulan](#8-kesimpulan)

---

## 1. Tujuan

- Membangun model SVM untuk mendiagnosis tumor payudara (ganas vs jinak).
- Menunjukkan pentingnya _scaling_ fitur pada SVM.
- Membandingkan empat kernel: `linear`, `polynomial`, `radial`, `sigmoid`.
- Menentukan `cost` (C) dan `gamma` terbaik dengan grid search dan CV 5-fold.
- Mengevaluasi dengan confusion matrix, sensitivitas/spesifisitas, dan ROC-AUC.
- Memvalidasi model dengan CV 10-fold memakai `caret`.

## 2. Dataset

**Wisconsin Diagnostic Breast Cancer**: **569 pasien**, **30 fitur numerik** (10 ukuran inti sel × tiga versi: _mean_, _error_, _worst_), dan satu kolom `diagnosis`. Berkas `breast_cancer.csv` dibuat oleh notebook Python (nama kolom memakai garis bawah, mis. `mean_radius`).

| Kelas             | Jumlah | Proporsi |
| ----------------- | ------ | -------- |
| benign (jinak)    | 357    | 62,7%    |
| malignant (ganas) | 212    | 37,3%    |

Skrip mengatur `factor(df$diagnosis, levels = c("benign", "malignant"))`, sehingga **`malignant` adalah kelas positif** (level kedua) dalam ROC dan pada `confusionMatrix(..., positive = "malignant")`.

## 3. Persyaratan & Cara Menjalankan

```r
install.packages(c("e1071", "caret", "pROC", "kernlab"))
```

`kernlab` tidak dipanggil langsung oleh skrip, tetapi **dibutuhkan** oleh `caret` agar `method = "svmRadial"` dapat berjalan.

1. Letakkan `breast_cancer.csv` di _working directory_ R.
2. Jalankan:

```r
source("SVM_R.R")
# atau di terminal:
# Rscript SVM_R.R
```

Grafik (`plot(tuned)`, kurva ROC, `plot(svm_cv)`, dan batas keputusan) tampil di jendela _Plots_ RStudio.

## 4. Konsep Singkat

**SVM** mencari _hyperplane_ pemisah dengan **margin** terbesar; titik yang menentukan batas disebut **_support vector_**. Untuk data yang tidak terpisah linear, **kernel** (di sini RBF/_radial_) memetakan data ke ruang berdimensi lebih tinggi: K(x, x′) = exp(−γ‖x − x′‖²).

| Parameter (`e1071`) | Fungsi                        | Terlalu kecil                | Terlalu besar                |
| ------------------- | ----------------------------- | ---------------------------- | ---------------------------- |
| `cost` (C)          | Penalti kesalahan klasifikasi | Margin lebar, _underfitting_ | Margin sempit, _overfitting_ |
| `gamma`             | Jangkauan pengaruh satu titik | Batas sangat halus           | Batas berliku, _overfitting_ |

Karena SVM berbasis **jarak**, fitur berskala besar mendominasi. Pada `e1071`, argumen `scale = TRUE` (bawaan) menstandarisasi tiap fitur memakai rata-rata dan simpangan baku **data latih**, lalu menerapkan hal yang sama ke data uji secara otomatis. Perhatikan bahwa nilai bawaan `gamma` di `e1071` adalah **1 / jumlah fitur** (= 1/30 ≈ 0,033), tanpa memperhitungkan varians data.

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Persiapan: workspace, seed, dan library

```r
rm(list=ls())
set.seed(42)
library('e1071'); library('caret'); library('pROC')
```

Membersihkan _workspace_, mengunci bilangan acak agar hasil dapat direproduksi, lalu memuat `e1071` (SVM), `caret` (partisi data, `confusionMatrix`, validasi silang), dan `pROC` (kurva ROC dan AUC).

### 5.2 Memuat dan mengeksplorasi data

```r
df <- read.csv("breast_cancer.csv")
df$diagnosis <- factor(df$diagnosis, levels = c("benign", "malignant"))
dim(df); table(df$diagnosis)
summary(df[, c("mean_radius", "mean_texture", "mean_area", "mean_smoothness")])
```

**Hasil:** `569 31` (569 baris, 30 fitur + 1 target), dan statistik empat fitur contoh:

| Fitur           | Min    | Median | Rata-rata | Maks   |
| --------------- | ------ | ------ | --------- | ------ |
| mean_radius     | 6,981  | 13,37  | 14,127    | 28,11  |
| mean_texture    | 9,71   | 18,84  | 19,29     | 39,28  |
| mean_area       | 143,5  | 551,1  | 654,9     | 2501,0 |
| mean_smoothness | 0,0526 | 0,0959 | 0,0964    | 0,1634 |

**Interpretasi:**

- Data lengkap tanpa nilai hilang, dan seluruh fitur numerik.
- Kelas tidak seimbang ringan (63% jinak : 37% ganas), jadi **sensitivitas kelas ganas** perlu dipantau selain akurasi.
- **Skala fitur sangat berbeda:** `mean_area` bernilai ratusan sampai ribuan, sedangkan `mean_smoothness` hanya sekitar 0,1. Tanpa standarisasi, `mean_area` akan mendominasi perhitungan jarak (lihat 5.4).

### 5.3 Pembagian data latih dan uji (80/20, berstrata)

```r
idx   <- createDataPartition(df$diagnosis, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]
```

`createDataPartition()` dari `caret` melakukan pembagian **berstrata**, menjaga proporsi kelas di kedua bagian.

| Set   | Total | benign | malignant |
| ----- | ----- | ------ | --------- |
| Latih | 456   | 286    | 170       |
| Uji   | 113   | 71     | 42        |

**Interpretasi:** proporsi ganas pada data latih (37,3%) dan uji (37,2%) hampir sama dengan data asli. Dengan 113 data uji, satu kesalahan setara ± 0,9 poin akurasi.

### 5.4 Data biasa vs. dengan scaling

```r
m_raw <- svm(diagnosis ~ ., data = train, kernel = "radial", cost = 1, scale = FALSE)
m_std <- svm(diagnosis ~ ., data = train, kernel = "radial", cost = 1, scale = TRUE)
akurasi <- function(m) mean(predict(m, test) == test$diagnosis)
```

**Hasil:**

| Model                           | Akurasi uji          |
| ------------------------------- | -------------------- |
| Tanpa scaling (`scale = FALSE`) | **0,6283**           |
| Dengan scaling (`scale = TRUE`) | **0,9823** (111/113) |

**Interpretasi:**

- Scaling menaikkan akurasi dari 62,8% menjadi **98,2%**, selisih **35 poin persentase**.
- Angka 0,6283 sama persis dengan proporsi kelas jinak pada data uji (71/113). Pada pengecekan saya, model tanpa scaling **memprediksi semua data uji sebagai benign**, dan seluruh 456 data latih menjadi _support vector_. Model itu runtuh menjadi "tebak kelas mayoritas".
- **Penyebabnya:** jarak kuadrat antar titik pada fitur mentah sangat besar (karena `mean_area`, `worst_area`, dan sejenisnya), sehingga dengan `gamma` bawaan 1/30, nilai kernel exp(−γd²) mendekati 0 untuk hampir semua pasangan titik. Tiap titik hanya "mirip" dengan dirinya sendiri, dan model tidak dapat menggeneralisasi.
- Ini lebih parah dibanding versi Python (90,35% tanpa scaling), karena `SVC(gamma="scale")` di Python menyesuaikan `gamma` dengan varians data, sedangkan `e1071` memakai 1/jumlah fitur tetap. Pelajarannya sama: **standarisasi wajib untuk SVM**.

### 5.5 Perbandingan kernel

```r
for (k in c("linear", "polynomial", "radial", "sigmoid")) {
  m <- svm(diagnosis ~ ., data = train, kernel = k, cost = 1, scale = TRUE)
  cat(sprintf("%-11s akurasi uji: %.4f | jumlah SV: %d\n", k, akurasi(m), m$tot.nSV))
}
```

**Hasil:**

| Kernel     | Akurasi uji      | Jumlah SV |
| ---------- | ---------------- | --------- |
| linear     | 0,9823 (111/113) | 38        |
| polynomial | 0,8938 (101/113) | 137       |
| radial     | 0,9823 (111/113) | 111       |
| sigmoid    | 0,9823 (111/113) | 72        |

**Interpretasi:**

- **Linear, radial, dan sigmoid seri** di 98,23%. Perbedaan kernel tidak terlihat pada akurasi uji.
- **Polynomial** (derajat 3 bawaan) paling buruk (89,4%), dengan SV banyak (137), serupa dengan temuan di Python.
- Dari sisi **kompleksitas**, kernel linear paling ringkas (hanya **38 SV**) dengan akurasi yang sama, tanda bahwa data ini **hampir terpisah secara linear**. Kernel RBF memakai 111 SV (kurang efisien) tanpa keuntungan akurasi pada `cost = 1`.
- Sigmoid mencapai akurasi yang sama di sini, tetapi kernel ini kurang stabil secara umum (tidak selalu valid sebagai kernel), jadi bukan pilihan utama.

### 5.6 Tuning `cost` dan `gamma` (grid search, CV 5-fold)

```r
tuned <- tune.svm(diagnosis ~ ., data = train, kernel = "radial",
                  cost  = c(0.1, 1, 10, 100),
                  gamma = c(0.0001, 0.001, 0.01, 0.1, 1),
                  tunecontrol = tune.control(cross = 5))
print(tuned)
plot(tuned)
```

Menguji 4 × 5 = **20 kombinasi** dengan CV 5-fold pada data latih. `tune.svm` mengukur **error** (bukan akurasi), dan `plot(tuned)` menampilkan peta kontur error CV (warna lebih gelap/rendah = error lebih kecil).

**Hasil:**

```
- best parameters:  gamma 0.01   cost 10
- best performance: 0.02415194
```

Error CV terbaik 0,02415 setara akurasi CV **97,58%**.

Saya juga memeriksa seluruh grid (skrip hanya mencetak yang terbaik). Akurasi CV (= 1 − error), baris = `cost`, kolom = `gamma`:

| cost \ gamma | 0,0001 | 0,001  | 0,01       | 0,1    | 1      |
| ------------ | ------ | ------ | ---------- | ------ | ------ |
| 0,1          | 0,6273 | 0,7457 | 0,9408     | 0,9056 | 0,6273 |
| 1            | 0,7500 | 0,9452 | 0,9539     | 0,9495 | 0,6317 |
| **10**       | 0,9452 | 0,9649 | **0,9758** | 0,9496 | 0,6317 |
| 100          | 0,9671 | 0,9693 | 0,9649     | 0,9495 | 0,6317 |

**Interpretasi:**

- **Nilai ≈ 0,627–0,632 adalah akurasi "tebak kelas mayoritas"** (286/456 = 0,6272). Muncul di pojok kiri atas (_underfitting_: `cost` kecil dan `gamma` sangat kecil) dan pada kolom `gamma` = 1 (_overfitting_ parah: tiap titik hanya berpengaruh pada tetangga terdekatnya).
- Seperti pada versi Python, terdapat **punggung diagonal** akurasi tinggi: `cost` besar cocok dengan `gamma` kecil, dan sebaliknya. Banyak kombinasi di punggung ini bagus (0,96–0,976).
- Parameter terbaik (**`cost` = 10, `gamma` = 0,01**) berada di **bagian dalam** grid dan sama persis dengan hasil di Python.
- Beberapa kombinasi hampir sama baiknya (mis. `cost` = 100, `gamma` = 0,001: 0,9693). Selisih dengan yang terbaik hanya ≈ 2 sampel dari 456, jadi bukan perbedaan yang tegas.

### 5.7 Model final

```r
final <- svm(diagnosis ~ ., data = train, kernel = "radial", scale = TRUE,
             cost  = tuned$best.parameters$cost,
             gamma = tuned$best.parameters$gamma,
             probability = TRUE)
summary(final); final$nSV; final$tot.nSV
```

`probability = TRUE` mengaktifkan estimasi probabilitas yang dibutuhkan untuk ROC.

**Hasil:**

```
SVM-Type: C-classification | SVM-Kernel: radial | cost: 10
Number of Support Vectors: 56   ( 25 31 )
```

**Interpretasi:**

- Hanya **56 dari 456** data latih (12,3%) yang menjadi _support vector_: 25 dari kelas jinak dan 31 dari kelas ganas. Hanya titik-titik di sekitar batas inilah yang menentukan model.
- Dibanding model awal (`cost = 1`, tanpa tuning) dengan 111 SV, model tuned **jauh lebih ringkas** (56 SV) sambil mempertahankan akurasi tinggi.
- Jumlah SV (56) sama dengan versi Python, hanya dengan komposisi per kelas berbeda karena pembagian data berbeda.

### 5.8 Evaluasi pada data uji

```r
pred <- predict(final, test)
confusionMatrix(pred, test$diagnosis, positive = "malignant")
```

**Confusion matrix (baris = prediksi, kolom = aktual):**

| Prediksi \ Aktual | benign | malignant |
| ----------------- | ------ | --------- |
| **benign**        | 70     | 0         |
| **malignant**     | 1      | 42        |

**Statistik:**

| Metrik                           | Nilai                |
| -------------------------------- | -------------------- |
| Accuracy                         | **0,9912** (112/113) |
| 95% CI                           | 0,9517 – 0,9998      |
| No Information Rate              | 0,6283               |
| P-Value [Acc > NIR]              | < 2e-16              |
| Kappa                            | 0,9811               |
| Sensitivity (recall ganas)       | **1,0000**           |
| Specificity                      | 0,9859               |
| Pos Pred Value (precision ganas) | 0,9767               |
| Neg Pred Value                   | 1,0000               |
| Balanced Accuracy                | 0,9930               |

**Interpretasi:**

- Dari 113 data uji, **112 benar** dan hanya **1 salah**: satu tumor **jinak yang diprediksi ganas** (_false positive_).
- **Tidak ada kasus ganas yang terlewat** (sensitivitas 100%, _false negative_ = 0). Ini hasil yang diutamakan dalam diagnosis, karena kanker yang tidak terdeteksi jauh lebih berbahaya daripada alarm palsu.
- Konsekuensinya: precision ganas 0,9767, artinya dari 43 pasien yang diprediksi ganas, 42 benar-benar ganas.
- Akurasi (99,12%) jauh di atas NIR (62,83%, akurasi bila selalu menebak jinak) dan p-value < 2e-16 menunjukkan perbedaannya sangat signifikan. **Kappa 0,98** termasuk kesepakatan hampir sempurna.
- **Selang kepercayaan 95% lebar** (95,2% – 99,98%) karena data uji hanya 113. Akurasi 99,1% adalah perkiraan, bukan nilai pasti.
- Akurasi uji (99,12%) sedikit lebih tinggi dari akurasi CV (97,58%); selisihnya setara 1–2 sampel dan wajar pada data uji sekecil ini. Tidak ada tanda overfitting.

### 5.9 Kurva ROC dan AUC

```r
p    <- predict(final, test, probability = TRUE)
prob <- attr(p, "probabilities")[, "malignant"]
roc_obj <- roc(test$diagnosis, prob, levels = c("benign", "malignant"))
auc(roc_obj)
plot(roc_obj, main = "Kurva ROC SVM RBF")
```

**Hasil:** `Area under the curve: 1` (pesan _Setting direction: controls < cases_ berarti probabilitas lebih tinggi menunjukkan kelas ganas).

**Interpretasi:**

- **AUC = 1,0**: pada data uji ini, setiap kasus ganas memiliki probabilitas lebih tinggi daripada setiap kasus jinak. Kurva ROC menempel ke pojok kiri atas.
- Dengan AUC = 1 berarti **ada suatu ambang probabilitas** yang memisahkan kedua kelas dengan sempurna pada data uji, meski satu prediksi `predict()` default tetap keliru. Namun memilih ambang itu berdasarkan data uji berarti mengoptimalkan pada data yang sama (hasilnya menjadi optimistis).
- **Waspadai AUC = 1:** pada data uji kecil (113), nilai ini bukan jaminan kinerja sempurna pada data baru. Angka yang jujur di populasi hampir pasti sedikit di bawah 1. Probabilitas dari `e1071` juga berasal dari kalibrasi Platt (estimasi internal), bukan probabilitas bawaan SVM.

### 5.10 Validasi silang 10-fold dengan `caret`

```r
ctrl <- trainControl(method = "cv", number = 10)
svm_cv <- train(diagnosis ~ ., data = train, method = "svmRadial",
                preProcess = c("center", "scale"),
                trControl = ctrl, tuneLength = 5)
```

`caret` melakukan standarisasi (`center` + `scale`) di dalam tiap lipatan CV, lalu mencoba 5 nilai `C`. Pada `svmRadial`, parameter `sigma` **tidak dituning** tetapi dipatok dari estimasi data (`sigma` setara `gamma`).

**Hasil (456 sampel, 30 prediktor, 10-fold):**

| C        | Accuracy   | Kappa      |
| -------- | ---------- | ---------- |
| 0,25     | 0,9452     | 0,8813     |
| 0,50     | 0,9650     | 0,9242     |
| 1,00     | 0,9693     | 0,9341     |
| 2,00     | 0,9694     | 0,9339     |
| **4,00** | **0,9715** | **0,9381** |

`sigma` tetap 0,043436; model terpilih: **`C` = 4**.

**Interpretasi:**

- Akurasi CV **97,15%** sejalan dengan hasil `tune.svm` (97,58%) dan akurasi uji (99,12%): tiga evaluasi yang berbeda memberi kesimpulan konsisten (≈ 97–99%).
- Akurasi naik **monoton** seiring `C` dan `C` terbaik (4) berada di **tepi atas** grid. Artinya `C` yang lebih besar mungkin masih memperbaiki hasil; perluas grid (mis. `tuneGrid` dengan `C` hingga 32 atau 100) bila diperlukan. Kenaikan dari C = 1 ke 4 sangat kecil (≈ 0,2 poin) sehingga praktis tidak bermakna.
- Hasil ini tidak persis sama dengan `tune.svm` karena pendekatannya berbeda: lipatan CV berbeda (10 vs 5), `sigma` tetap 0,0434 (vs `gamma` = 0,01 hasil tuning), dan hanya `C` yang dituning.
- `plot(svm_cv)` menampilkan kurva akurasi CV terhadap `C`.

### 5.11 Ilustrasi batas keputusan 2 variabel

```r
m2 <- svm(diagnosis ~ mean_radius + mean_texture, data = df, kernel = "radial", cost = 1, scale = TRUE)
plot(m2, df, mean_texture ~ mean_radius)
```

Model hanya memakai `mean_radius` dan `mean_texture` dan dilatih pada **seluruh** data, supaya batas keputusan dapat digambar dalam 2 dimensi. Plot `e1071` menampilkan kelas sebagai warna latar dan _support vector_ sebagai tanda silang.

**Interpretasi:** skrip tidak mencetak ukuran model ini; saat saya memeriksanya, model memakai **156 support vector dari 569 (27,4%)** dengan akurasi latih **90,7%**. Jauh lebih buruk dibanding model 30 variabel (56 SV, akurasi ≈ 99%). Dua variabel saja **tidak cukup** untuk memisahkan kelas, karena banyak titik berada di dalam atau dekat margin. Gambar ini murni ilustrasi untuk memahami margin dan _support vector_, bukan model yang dipakai.

### 5.12 Prediksi pasien

```r
baru <- test[1:3, ]
pb   <- predict(final, baru, probability = TRUE)
data.frame(aktual = baru$diagnosis, prediksi = pb,
           P_malignant = round(attr(pb, "probabilities")[, "malignant"], 3))
```

**Hasil:**

| Baris data | Aktual    | Prediksi  | P(malignant) |
| ---------- | --------- | --------- | ------------ |
| 4          | malignant | malignant | 0,972        |
| 5          | malignant | malignant | 1,000        |
| 9          | malignant | malignant | 0,998        |

**Interpretasi:**

- Ketiga prediksi **benar**, dengan keyakinan tinggi (97,2% – 100%).
- Ketiganya kebetulan **kasus ganas**, jadi contoh ini tidak menguji kasus jinak.
- Probabilitas berasal dari kalibrasi Platt, bukan risiko klinis yang pasti.
- "Pasien baru" di sini sebenarnya tiga baris pertama data uji, bukan data yang benar-benar baru.

## 6. Ringkasan Hasil

| Ukuran                                | Nilai                                                              |
| ------------------------------------- | ------------------------------------------------------------------ |
| Data latih / uji                      | 456 / 113 (berstrata)                                              |
| RBF tanpa scaling → dengan scaling    | 62,83% → 98,23%                                                    |
| Kernel (cost = 1, scaling)            | linear 98,23% · radial 98,23% · sigmoid 98,23% · polynomial 89,38% |
| Parameter terbaik (`tune.svm`)        | `cost = 10`, `gamma = 0,01`                                        |
| Akurasi CV 5-fold (`tune.svm`)        | 97,58%                                                             |
| Akurasi CV 10-fold (`caret`, `C` = 4) | 97,15%                                                             |
| **Akurasi data uji (model final)**    | **99,12%** (112/113)                                               |
| Sensitivitas / spesifisitas           | 1,000 / 0,9859                                                     |
| Kappa / AUC                           | 0,9811 / 1,0                                                       |
| Support vector                        | 56 dari 456 (12,3%)                                                |

## 7. Perbandingan dengan Versi Python

| Aspek                           | Python (`Pertemuan11_SVM.ipynb`) | R (`SVM_R.R`)               |
| ------------------------------- | -------------------------------- | --------------------------- |
| Data latih / uji                | 455 / 114                        | 456 / 113                   |
| RBF tanpa scaling               | 90,35%                           | 62,83%                      |
| RBF dengan scaling (`cost = 1`) | 97,37%                           | 98,23%                      |
| Parameter terbaik               | `C = 10`, `gamma = 0,01`         | `cost = 10`, `gamma = 0,01` |
| Akurasi CV 5-fold terbaik       | 97,58%                           | 97,58%                      |
| Akurasi uji model final         | 98,25%                           | 99,12%                      |
| Jenis kesalahan di data uji     | 2 _false negative_               | 1 _false positive_          |
| Support vector                  | 56                               | 56                          |
| AUC                             | 0,996                            | 1,0                         |

**Catatan:**

- Parameter terbaik, akurasi CV, dan jumlah SV **konsisten** di kedua bahasa. Itu adalah temuan yang kokoh.
- Angka data uji tidak boleh dibandingkan langsung karena pembagian data acak berbeda (R vs. NumPy menghasilkan data uji yang berbeda). Selisih 1 pasien (≈ 0,9 poin) masih dalam batas wajar.
- Hasil **tanpa scaling** sangat berbeda (90% vs 63%) karena bawaan `gamma` berbeda: `gamma = "scale"` di scikit-learn menyesuaikan dengan varians data, sedangkan `e1071` memakai 1/jumlah fitur.
- Perbedaan jenis kesalahan (2 _false negative_ di Python vs 1 _false positive_ di R) hanya mencerminkan data uji yang berbeda, bukan kualitas model yang berbeda.

## 8. Kesimpulan

1. **Scaling wajib untuk SVM:** tanpa scaling model runtuh ke tebakan mayoritas (62,8%); dengan scaling 98,2%.
2. Parameter terbaik **`cost` = 10, `gamma` = 0,01** (CV 5-fold 97,58%) menghasilkan model final dengan akurasi uji **99,12%**, sensitivitas **100%**, dan hanya 56 support vector.
3. Data ini **hampir terpisah linear**: kernel linear sama baiknya dengan RBF pada data uji dan jauh lebih ringkas (38 SV).
4. Hubungan `cost` dan `gamma` membentuk punggung diagonal; `gamma = 1` menyebabkan _overfitting_ parah, dan kombinasi terlalu kecil menyebabkan _underfitting_.
5. Tiga evaluasi (CV 5-fold, CV 10-fold, data uji) konsisten di kisaran 97–99%, tanpa tanda overfitting.

## Referensi

- Fadil, M., Islamiyati, A., & Thamrin, S. A. (2025). Classification of nutritional status in toddlers using the support vector machine method. _Communications in Mathematical Biology and Neuroscience_, 2025, Article ID 57. https://doi.org/10.28919/cmbn/9126
- Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. _Advances in Large Margin Classifiers_, 61–74.
- Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. _ACM Transactions on Intelligent Systems and Technology_, 2(3), 27.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The Elements of Statistical Learning (2nd ed.)_. Springer.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). _An Introduction to Statistical Learning (2nd ed.)_. Springer.
- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. _Proceedings of IS&T/SPIE_, 1905, 861–870.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_, 12, 2825–2830.
- Meyer, D., et al. _e1071: Misc Functions of the Department of Statistics, Probability Theory Group (TU Wien)_. R package.
