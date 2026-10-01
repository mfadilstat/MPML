# Mata kuliah: Modern Prediksi dan Machine Learning

Author: Muhammad Fadil

Update: 2026-10-01

# Klasifikasi Bunga Iris dengan Random Forest (R)

Program ini menerapkan algoritma **Random Forest** untuk mengklasifikasikan spesies bunga _Iris_ (`setosa`, `versicolor`, `virginica`) berdasarkan empat ukuran morfologi bunga. Seluruh alur kerja ada di satu skrip, `Program R`: eksplorasi data, pembagian data latih/uji, pelatihan model, evaluasi, analisis pentingnya variabel, _tuning_ parameter `mtry`, validasi silang 10-fold, hingga prediksi data baru.

> Semua angka hasil di dokumen ini diperoleh dari menjalankan `Program R` apa adanya dengan `set.seed(42)`. Jika Anda memakai versi R atau package yang berbeda, angka bisa sedikit berbeda.

---

## Daftar Isi

1. [Tujuan](#1-tujuan)
2. [Dataset](#2-dataset)
3. [Persyaratan & Cara Menjalankan](#3-persyaratan--cara-menjalankan)
4. [Penjelasan Konsep Singkat](#4-penjelasan-konsep-singkat)
5. [Penjelasan Kode & Interpretasi Hasil](#5-penjelasan-kode--interpretasi-hasil)
   - [5.1 Persiapan](#51-persiapan-environment-package-dan-seed)
   - [5.2 Eksplorasi Data](#52-memuat-dan-mengeksplorasi-data)
   - [5.3 Split Data](#53-membagi-data-latih-dan-uji-7030)
   - [5.4 Model Random Forest](#54-membangun-model-random-forest)
   - [5.5 Evaluasi pada Data Uji](#55-evaluasi-model-pada-data-uji)
   - [5.6 Pentingnya Variabel](#56-pentingnya-variabel-variable-importance)
   - [5.7 Mencari mtry Optimal (tuneRF)](#57-mencari-mtry-optimal-dengan-oob-tunerf)
   - [5.8 Cross-Validation dengan caret](#58-validasi-silang-10-fold-dan-tuning-dengan-caret)
   - [5.9 Prediksi Data Baru](#59-prediksi-data-baru)
6. [Ringkasan Hasil](#6-ringkasan-hasil)
7. [Kesimpulan](#7-kesimpulan)

---

## 1. Tujuan

- Membangun model klasifikasi spesies Iris menggunakan Random Forest.
- Mengukur kinerja model dengan **OOB error**, **confusion matrix** pada data uji, dan **10-fold cross-validation**.
- Mengetahui variabel yang paling berpengaruh dalam membedakan spesies.
- Menentukan nilai `mtry` yang sesuai.
- Memakai model untuk memprediksi data pengamatan baru.

## 2. Dataset

Dataset `iris` bawaan R (Fisher, 1936): **150 observasi**, **4 variabel prediktor numerik**, dan **1 variabel target kategorik** dengan 3 kelas yang seimbang (masing-masing 50 sampel).

| Variabel       | Tipe             | Keterangan                                   |
| -------------- | ---------------- | -------------------------------------------- |
| `Sepal.Length` | numerik          | Panjang kelopak luar (cm)                    |
| `Sepal.Width`  | numerik          | Lebar kelopak luar (cm)                      |
| `Petal.Length` | numerik          | Panjang mahkota (cm)                         |
| `Petal.Width`  | numerik          | Lebar mahkota (cm)                           |
| `Species`      | faktor (3 level) | `setosa`, `versicolor`, `virginica` (target) |

## 3. Persyaratan & Cara Menjalankan

**Kebutuhan:** R (disarankan ≥ 4.0) dan dua package berikut.

```r
install.packages(c("randomForest", "caret"))
```

Skrip memakai `set.seed(42)` agar hasil dapat direproduksi. Grafik (`plot`, `varImpPlot`, `tuneRF`, `plot(rf_cv)`) akan tampil di jendela _Plots_ pada RStudio.

## 4. Penjelasan Konsep Singkat

**Random Forest** adalah metode _ensemble_ yang menggabungkan banyak pohon keputusan. Ada dua sumber keacakan:

1. **Bagging (bootstrap aggregating):** tiap pohon dilatih pada sampel bootstrap (pengambilan acak dengan pengembalian) dari data latih.
2. **Pemilihan variabel acak:** di setiap percabangan (_split_), hanya `mtry` variabel yang dipilih acak untuk dipertimbangkan.

Untuk klasifikasi, prediksi akhir ditentukan dengan **suara terbanyak (majority vote)** dari seluruh pohon. Karena tiap pohon hanya memakai ± 63% data asli, sisanya (± 37%) yang disebut **_out-of-bag_ (OOB)** dipakai untuk menguji pohon tersebut. Hasilnya adalah **OOB error**, estimasi galat yang jujur tanpa perlu data uji terpisah.

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Persiapan: environment, package, dan seed

```r
rm(list = ls())

library('randomForest')
library('caret')
set.seed(42)
```

- `rm(list = ls())` membersihkan semua objek di _workspace_ agar skrip berjalan dari kondisi bersih.
- `randomForest` menyediakan algoritma Random Forest; `caret` menyediakan `confusionMatrix()` dan `train()` untuk validasi silang.
- `set.seed(42)` mengunci bilangan acak sehingga pembagian data dan pembentukan pohon dapat **direproduksi**.

### 5.2 Memuat dan mengeksplorasi data

```r
data(iris)
str(iris)
summary(iris)
table(iris$Species)
```

- `str()` menampilkan struktur: 150 baris, 4 kolom numerik, dan `Species` bertipe _factor_ dengan 3 level.
- `summary()` memberi ringkasan statistik tiap variabel.
- `table()` menghitung jumlah sampel per kelas.

**Hasil:**

| Variabel     | Min | Median | Mean  | Max |
| ------------ | --- | ------ | ----- | --- |
| Sepal.Length | 4.3 | 5.80   | 5.843 | 7.9 |
| Sepal.Width  | 2.0 | 3.00   | 3.057 | 4.4 |
| Petal.Length | 1.0 | 4.35   | 3.758 | 6.9 |
| Petal.Width  | 0.1 | 1.30   | 1.199 | 2.5 |

```
    setosa versicolor  virginica
        50         50         50
```

**Interpretasi:**

- Tidak ada nilai hilang, dan semua prediktor sudah numerik, sehingga tidak perlu pra-pemrosesan.
- Kelas **seimbang sempurna** (50/50/50), jadi _accuracy_ adalah metrik yang adil dan tidak perlu penyeimbangan kelas.
- Pada `Petal.Length`, median (4.35) jauh di atas rata-rata (3.76). Ini mengindikasikan distribusi yang tidak seragam. Hal ini wajar karena data berasal dari tiga kelompok spesies dengan ukuran mahkota yang berbeda jauh. Pola inilah yang nanti dimanfaatkan model.

### 5.3 Membagi data latih dan uji (70/30)

```r
n     <- nrow(iris)
idx   <- sample(n, size = n*0.7)
train <- iris[idx, ]
test  <- iris[-idx, ]
```

- `sample(n, size = n*0.7)` mengambil acak 105 indeks baris (70%) untuk data latih.
- Sisa 45 baris (30%) menjadi data uji yang **tidak pernah dilihat** model saat pelatihan.

**Komposisi hasil pembagian (dengan seed 42):**

| Set   | Total | setosa | versicolor | virginica |
| ----- | ----- | ------ | ---------- | --------- |
| Latih | 105   | 38     | 35         | 32        |
| Uji   | 45    | 12     | 15         | 18        |

**Interpretasi:** pembagian ini acak biasa (_simple random sampling_), bukan berstrata, sehingga proporsi kelas di data latih/uji tidak persis sama (mis. virginica 40% di data uji, tetapi ± 30% di data latih). Untuk dataset kecil seperti ini, perbedaan itu dapat memengaruhi hasil evaluasi (lihat [Catatan](#8-catatan--saran-pengembangan)).

### 5.4 Membangun model Random Forest

```r
rf <- randomForest(Species ~ ., data = train,
                   ntree      = 500,
                   mtry       = 2,
                   importance = TRUE)

print(rf)
plot(rf)
```

| Parameter     | Nilai | Arti                                                                                               |
| ------------- | ----- | -------------------------------------------------------------------------------------------------- |
| `Species ~ .` |       | Target `Species`, prediktor = semua kolom lain                                                     |
| `ntree`       | 500   | Jumlah pohon dalam hutan                                                                           |
| `mtry`        | 2     | Jumlah variabel acak yang dipertimbangkan di tiap split (≈ √4 = 2, nilai bawaan untuk klasifikasi) |
| `importance`  | TRUE  | Hitung pentingnya variabel                                                                         |

**Hasil `print(rf)`:**

```
Type of random forest: classification
Number of trees: 500
No. of variables tried at each split: 2

OOB estimate of  error rate: 5.71%
Confusion matrix:
           setosa versicolor virginica class.error
setosa         38          0         0  0.00000000
versicolor      0         33         2  0.05714286
virginica       0          4        28  0.12500000
```

**Interpretasi:**

- **OOB error 5,71%** berarti ± 94,3% data latih diprediksi benar oleh pohon-pohon yang tidak memakai data tersebut saat dilatih. Total 6 dari 105 sampel salah (2 + 4).
- **setosa** diklasifikasikan **sempurna** (galat 0%). Spesies ini sangat mudah dipisahkan dari dua lainnya.
- **versicolor** salah 2 kali (keduanya ditebak sebagai virginica), galat 5,71%.
- **virginica** salah 4 kali (semuanya ditebak sebagai versicolor), galat 12,5%, yang paling tinggi.
- Seluruh kesalahan terjadi antara **versicolor ↔ virginica**; tidak ada yang tertukar dengan setosa. Artinya dua spesies tersebut memiliki ukuran yang tumpang tindih.

**Grafik `plot(rf)`** menampilkan error (OOB dan per kelas) terhadap jumlah pohon. Biasanya error turun tajam pada puluhan pohon pertama lalu **mendatar (stabil)**. Jika kurva sudah datar sebelum 500 pohon, menambah pohon tidak memperbaiki akurasi, hanya menambah waktu komputasi. Random Forest tidak mengalami _overfitting_ hanya karena jumlah pohon bertambah.

### 5.5 Evaluasi model pada data uji

```r
pred <- predict(rf, newdata = test)
confusionMatrix(pred, test$Species)
```

`predict()` menghasilkan prediksi kelas untuk 45 data uji, lalu `confusionMatrix()` (dari `caret`) membandingkannya dengan kelas sebenarnya.

**Confusion matrix (baris = prediksi, kolom = aktual):**

| Prediksi \ Aktual | setosa | versicolor | virginica |
| ----------------- | ------ | ---------- | --------- |
| **setosa**        | 12     | 0          | 0         |
| **versicolor**    | 0      | 14         | 1         |
| **virginica**     | 0      | 1          | 17        |

**Statistik keseluruhan:**

| Metrik              | Nilai               |
| ------------------- | ------------------- |
| Accuracy            | **0,9556** (95,56%) |
| 95% CI              | 0,8485 – 0,9946     |
| No Information Rate | 0,40                |
| P-Value [Acc > NIR] | 2,842 × 10⁻¹⁵       |
| Kappa               | 0,9324              |

**Statistik per kelas:**

| Metrik                     | setosa | versicolor | virginica |
| -------------------------- | ------ | ---------- | --------- |
| Sensitivity (Recall)       | 1,0000 | 0,9333     | 0,9444    |
| Specificity                | 1,0000 | 0,9667     | 0,9630    |
| Pos Pred Value (Precision) | 1,0000 | 0,9333     | 0,9444    |
| Balanced Accuracy          | 1,0000 | 0,9500     | 0,9537    |

**Interpretasi:**

- Dari 45 data uji, **43 diprediksi benar** dan hanya **2 salah** (1 versicolor ditebak virginica, 1 virginica ditebak versicolor). Akurasi 95,56%.
- **Accuracy vs. NIR:** _No Information Rate_ 0,40 adalah akurasi jika selalu menebak kelas mayoritas (virginica). Model (0,9556) jauh melampaui itu, dan p-value ≈ 10⁻¹⁵ menunjukkan perbedaannya **sangat signifikan secara statistik**.
- **Kappa 0,93** (kesepakatan setelah dikoreksi peluang tebakan acak) termasuk kategori _almost perfect agreement_ (> 0,80).
- **Selang kepercayaan 95%** akurasi cukup lebar (84,9% – 99,5%) karena data uji hanya 45. Satu kesalahan tambahan saja mengubah akurasi sekitar 2,2 poin persentase. Jadi angka 95,56% sebaiknya dibaca sebagai perkiraan, bukan nilai pasti.
- Pola kesalahan konsisten dengan OOB: setosa sempurna, kebingungan hanya antara versicolor dan virginica.
- Akurasi uji (95,56%) dan OOB (94,29%) berdekatan, sehingga **tidak ada tanda overfitting**.

### 5.6 Pentingnya variabel (variable importance)

```r
importance(rf)
varImpPlot(rf, main = "Pentingnya Variabel")
```

**Hasil:**

| Variabel         | setosa | versicolor | virginica | MeanDecreaseAccuracy | MeanDecreaseGini |
| ---------------- | ------ | ---------- | --------- | -------------------- | ---------------- |
| Sepal.Length     | 7,50   | 6,20       | 7,69      | 10,43                | 7,08             |
| Sepal.Width      | 5,13   | −0,56      | 1,66      | 3,21                 | 2,12             |
| **Petal.Length** | 22,07  | 28,02      | 28,36     | **31,14**            | **30,52**        |
| **Petal.Width**  | 21,08  | 27,33      | 29,26     | 29,80                | 29,39            |

**Cara membaca:**

- **MeanDecreaseAccuracy:** rata-rata penurunan akurasi OOB jika nilai variabel tersebut diacak (_permutasi_). Semakin besar, semakin penting.
- **MeanDecreaseGini:** total penurunan impuritas Gini yang disumbangkan variabel di seluruh split. Semakin besar, semakin penting.
- Kolom per kelas menunjukkan pentingnya variabel untuk mengenali kelas tertentu.

**Interpretasi:**

- **`Petal.Length` dan `Petal.Width` adalah dua variabel paling penting**, dengan skor jauh di atas dua variabel lain pada kedua ukuran (≈ 30 vs ≤ 10). Ukuran **mahkota** (petal) jauh lebih informatif untuk membedakan spesies daripada ukuran kelopak luar (sepal).
- `Sepal.Length` berkontribusi sedang, sedangkan **`Sepal.Width` paling lemah**. Nilainya untuk versicolor bahkan negatif (−0,56), artinya mengacak variabel ini hampir tidak menurunkan (bahkan sedikit menaikkan) akurasi kelas tersebut; dalam praktiknya dianggap tidak berkontribusi.
- Selisih antara `Petal.Length` dan `Petal.Width` sangat kecil, dan keduanya berkorelasi tinggi. Peringkat di antara keduanya bisa bertukar bila seed atau data berubah, jadi jangan menafsirkannya sebagai keunggulan yang pasti.
- `varImpPlot` menampilkan temuan yang sama dalam bentuk grafik dua panel (kiri: MeanDecreaseAccuracy, kanan: MeanDecreaseGini).

### 5.7 Mencari `mtry` optimal dengan OOB (`tuneRF`)

```r
tuneRF(train[, -5], train$Species, ntreeTry = 500,
       stepFactor = 1.5, improve = 0.01, trace = TRUE, plot = TRUE)
```

| Argumen            | Arti                                                         |
| ------------------ | ------------------------------------------------------------ |
| `train[, -5]`      | Prediktor (semua kolom kecuali kolom ke-5, `Species`)        |
| `train$Species`    | Target                                                       |
| `ntreeTry = 500`   | Pohon yang dibangun untuk tiap kandidat `mtry`               |
| `stepFactor = 1.5` | `mtry` dikali/dibagi 1,5 di tiap langkah pencarian           |
| `improve = 0.01`   | Pencarian lanjut hanya bila OOB error membaik ≥ 1% (relatif) |

**Hasil:**

| mtry  | OOB error |
| ----- | --------- |
| 2     | 5,71%     |
| **3** | **4,76%** |
| 4     | 6,67%     |

**Interpretasi:**

- Pencarian dimulai dari `mtry` = 2 lalu bergerak ke kiri dan kanan. `mtry` = **3** memberi OOB error terendah (4,76%).
- Perbedaan antara 5,71% dan 4,76% setara dengan **1 sampel** dari 105. Selisih sekecil itu **tidak cukup kuat** untuk menyimpulkan bahwa `mtry` = 3 benar-benar lebih baik. Pada dataset Iris, hasil untuk `mtry` 2–4 secara praktis setara.
- Catatan teknis: `tuneRF` hanya mengembalikan tabel dan grafik, hasilnya **tidak disimpan** ke objek. Model `rf` di atas tetap memakai `mtry = 2`.

### 5.8 Validasi silang 10-fold dan tuning dengan `caret`

```r
ctrl <- trainControl(method = "cv", number = 10)
rf_cv <- train(Species ~ ., data = train, method = "rf",
               trControl = ctrl, tuneGrid = expand.grid(mtry = 1:4),
               ntree = 500)
print(rf_cv)
plot(rf_cv)
```

- `trainControl(method = "cv", number = 10)` mengatur **10-fold cross-validation**: data latih dibagi 10 bagian, 9 untuk melatih dan 1 untuk menguji, diulang 10 kali bergantian.
- `tuneGrid = expand.grid(mtry = 1:4)` mencoba semua nilai `mtry` dari 1 hingga 4 (seluruh kemungkinan, karena hanya ada 4 prediktor).
- `plot(rf_cv)` menampilkan akurasi CV untuk tiap `mtry`.

**Hasil:**

| mtry | Accuracy | Kappa  |
| ---- | -------- | ------ |
| 1    | 0,9436   | 0,9143 |
| 2    | 0,9436   | 0,9143 |
| 3    | 0,9345   | 0,9004 |
| 4    | 0,9436   | 0,9143 |

```
Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 1.
```

**Interpretasi:**

- Akurasi CV rata-rata sekitar **94%**, selaras dengan OOB error (± 94,3%) dan akurasi uji (95,6%). Tiga metode evaluasi yang berbeda memberikan kesimpulan yang konsisten, yang menandakan model **stabil dan andal**.
- `mtry` = 1, 2, dan 4 menghasilkan akurasi **identik** (0,9436), sedangkan `mtry` = 3 sedikit lebih rendah (0,9345). `caret` memilih `mtry` = 1 hanya karena seri, lalu mengambil nilai pertama.
- Perhatikan bahwa `tuneRF` (OOB) menyarankan `mtry` = 3, sedangkan CV menunjukkan `mtry` = 3 sebagai yang _terendah_. Kontradiksi ini wajar dan justru menegaskan bahwa **perbedaan antar nilai `mtry` pada data ini hanya "noise"** (selisih ≈ 1 sampel), bukan perbedaan kinerja yang nyata. Nilai bawaan `mtry` = 2 sudah memadai.

### 5.9 Prediksi data baru

```r
baru <- data.frame(Sepal.Length = c(5.1, 6.0, 6.9),
                   Sepal.Width  = c(3.5, 2.9, 3.1),
                   Petal.Length = c(1.4, 4.5, 5.4),
                   Petal.Width  = c(0.2, 1.5, 2.1))
predict(rf, baru)                 # kelas prediksi
predict(rf, baru, type = "prob")  # probabilitas tiap kelas
```

Dibuat tiga bunga hipotetis, lalu diprediksi memakai model `rf`. Nama kolom harus **persis sama** dengan data latih.

**Hasil:**

| Bunga | Sepal (P × L) | Petal (P × L) | Prediksi       | P(setosa) | P(versicolor) | P(virginica) |
| ----- | ------------- | ------------- | -------------- | --------- | ------------- | ------------ |
| 1     | 5,1 × 3,5     | 1,4 × 0,2     | **setosa**     | 1,000     | 0,000         | 0,000        |
| 2     | 6,0 × 2,9     | 4,5 × 1,5     | **versicolor** | 0,000     | 0,984         | 0,016        |
| 3     | 6,9 × 3,1     | 5,4 × 2,1     | **virginica**  | 0,000     | 0,000         | 1,000        |

**Interpretasi:**

- `type = "prob"` mengembalikan **proporsi pohon** (dari 500) yang memilih tiap kelas, bukan probabilitas statistik yang terkalibrasi.
- Bunga 1 dan 3 diprediksi dengan keyakinan penuh (100% pohon sepakat). Ukuran petal-nya jelas khas setosa (kecil) dan virginica (besar).
- Bunga 2 diprediksi versicolor dengan keyakinan tinggi (98,4%); sekitar 1,6% pohon (± 8 pohon) memilih virginica. Ini sesuai dengan temuan sebelumnya bahwa versicolor dan virginica memiliki area tumpang tindih.

## 6. Ringkasan Hasil

| Ukuran                          | Nilai                         |
| ------------------------------- | ----------------------------- |
| Data latih / uji                | 105 / 45                      |
| OOB error (mtry = 2, 500 pohon) | 5,71%                         |
| Akurasi data uji                | 95,56%                        |
| Kappa data uji                  | 0,9324                        |
| Akurasi 10-fold CV              | ≈ 94,4% (mtry 1, 2, 4)        |
| `mtry` terbaik menurut `tuneRF` | 3 (OOB 4,76%)                 |
| `mtry` terpilih `caret`         | 1 (seri dengan 2 dan 4)       |
| Variabel terpenting             | `Petal.Length`, `Petal.Width` |
| Variabel paling lemah           | `Sepal.Width`                 |

## 7. Kesimpulan

1. Random Forest mampu mengklasifikasikan spesies Iris dengan akurasi **± 94–96%** berdasarkan tiga cara evaluasi independen (OOB, data uji, dan CV 10-fold) yang hasilnya konsisten.
2. **Setosa** dapat dipisahkan sempurna, sedangkan seluruh kesalahan terjadi antara **versicolor dan virginica**.
3. Ukuran **petal (mahkota)**, terutama `Petal.Length` dan `Petal.Width`, adalah penentu utama klasifikasi; `Sepal.Width` hampir tidak berkontribusi.
4. Pemilihan `mtry` antara 1 sampai 4 tidak mengubah kinerja secara berarti pada dataset ini; `mtry` = 2 (bawaan) sudah cukup baik.
5. Model dapat langsung dipakai untuk memprediksi bunga baru, lengkap dengan tingkat keyakinan berbasis proporsi suara pohon.

## Referensi

- Breiman, L. (2001). _Random Forests_. Machine Learning, 45(1), 5–32.
- Fisher, R. A. (1936). _The use of multiple measurements in taxonomic problems_. Annals of Eugenics, 7(2), 179–188.
- Liaw, A. & Wiener, M. (2002). _Classification and Regression by randomForest_. R News, 2(3), 18–22.
- Kuhn, M. (2008). _Building Predictive Models in R Using the caret Package_. Journal of Statistical Software, 28(5).
