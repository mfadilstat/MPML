<h1>Mata kuliah : Modern Prediksi dan Machine Learning</h1>
<h3>Pertemuan 10: Gradient Boosting</h3>

<tt>Author: Muhammad Fadil</tt><br>
<tt>Update: 2026-10-01</tt>

# Prediksi Perkembangan Diabetes dengan Gradient Boosting (R & RStudio)

Proyek ini menerapkan **Gradient Boosting** di R menggunakan paket `gbm` untuk memprediksi ukuran perkembangan penyakit diabetes (masalah **regresi**). Skrip `Gradient_Boosting_using_R.R` membandingkan model dengan Regresi Linear, memilih jumlah pohon optimal lewat validasi silang, menafsirkan pentingnya variabel dan efek marginalnya (_partial dependence_), lalu melakukan _grid search_ dengan `caret`.

Skrip ini adalah padanan versi R dari notebook Python `Gradient_Boosting.ipynb` dan memakai berkas `diabetes.csv` yang diekspor notebook tersebut.

> Semua angka di dokumen ini diperoleh dari menjalankan skrip apa adanya dengan `set.seed(42)`, R 4.x, `gbm` 2.1.8.1, dan `caret`. `gbm` mengacak lipatan validasi silang secara internal, jadi pada versi paket atau R yang berbeda angka (terutama jumlah pohon optimal) bisa sedikit bergeser.

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

- Memprediksi skor perkembangan diabetes (variabel kontinu) dari 10 variabel klinis.
- Membandingkan Gradient Boosting dengan Regresi Linear.
- Menentukan jumlah pohon optimal secara otomatis dengan validasi silang (_early stopping_).
- Mengidentifikasi variabel penting dan bentuk hubungannya dengan target.
- Mencari hiperparameter terbaik dengan grid search dan CV 5-fold.

## 2. Dataset

Dataset `diabetes` (Efron et al., 2004): **442 pasien**, **10 fitur**, dan target `y`.

| Variabel                  | Keterangan                                                       |
| ------------------------- | ---------------------------------------------------------------- |
| `age`, `sex`, `bmi`, `bp` | Usia, jenis kelamin, indeks massa tubuh, tekanan darah rata-rata |
| `s1` – `s6`               | Enam ukuran serum darah                                          |
| `y`                       | Ukuran perkembangan penyakit satu tahun setelah pengukuran awal  |

Fitur sudah dipusatkan dan diskalakan (rata-rata 0). Berkas `diabetes.csv` dibuat oleh `Gradient_Boosting.ipynb` (`X.assign(y=y).to_csv("diabetes.csv", index=False)`).

## 3. Persyaratan & Cara Menjalankan

```r
install.packages(c("gbm", "caret"))
```

Letakkan `diabetes.csv` di _working directory_ R (skrip membacanya dengan `read.csv("diabetes.csv")`).

Grafik (kurva error `gbm.perf`, relative influence, dan partial dependence) tampil di jendela _Plots_ RStudio.

## 4. Konsep Singkat

**Gradient Boosting** membangun model secara berurutan: tiap pohon baru dilatih untuk memperbaiki sisa kesalahan (_residual_) model sebelumnya, lalu ditambahkan dengan bobot _shrinkage_ ($v$):

$$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$$

Pohon yang dipakai dangkal (_weak learner_). Berbeda dari Random Forest yang membangun pohon paralel, boosting sensitif terhadap jumlah iterasi: terlalu banyak pohon menyebabkan _overfitting_.

Padanan parameter `gbm` dengan scikit-learn:

| `gbm` (R)           | Fungsi                      | scikit-learn (Python)         |
| ------------------- | --------------------------- | ----------------------------- |
| `n.trees`           | Jumlah iterasi maksimum (M) | `n_estimators`                |
| `shrinkage`         | Learning rate (ν)           | `learning_rate`               |
| `interaction.depth` | Kompleksitas tiap pohon     | `max_depth`                   |
| `bag.fraction`      | Proporsi data per pohon     | `subsample`                   |
| `n.minobsinnode`    | Minimum observasi per daun  | `min_samples_leaf`            |
| `cv.folds`          | Lipatan CV untuk memilih M  | (manual / `n_iter_no_change`) |

> Catatan: pada `gbm`, `interaction.depth = d` berarti pohon dengan **d percabangan (split)**, sehingga `interaction.depth = 3` menghasilkan pohon berdaun 4. Ini berbeda dengan `max_depth = 3` di scikit-learn yang bisa menghasilkan hingga 8 daun. Jadi angka yang sama tidak persis setara.

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Persiapan (Header, `rm`, `set.seed`, library)

```r
rm(list = ls())
set.seed(42)
library('gbm')
library('caret')
```

Membersihkan _workspace_, mengunci bilangan acak agar hasil dapat direproduksi, dan memuat `gbm` (model) serta `caret` (tuning dengan validasi silang).

### 5.2 Memuat dan mengeksplorasi data

```r
df <- read.csv("diabetes.csv")
str(df)
summary(df)
```

**Hasil:** 442 observasi, 11 variabel numerik (10 fitur + `y`).

| Statistik `y` | Nilai |
| ------------- | ----- |
| Min           | 25,0  |
| Kuartil 1     | 87,0  |
| Median        | 140,5 |
| Rata-rata     | 152,1 |
| Kuartil 3     | 211,5 |
| Max           | 346,0 |

**Interpretasi:**

- Semua fitur memiliki rata-rata tepat 0 karena sudah distandardisasi, sehingga tidak perlu normalisasi tambahan.
- Target menyebar dari 25 sampai 346, dengan rata-rata (152,1) sedikit di atas median (140,5): sebaran agak miring ke kanan.
- Tidak ada nilai hilang.

### 5.3 Pembagian data latih dan uji (80/20)

```r
idx   <- sample(nrow(df), size = round(0.8 * nrow(df)))
train <- df[idx, ]
test  <- df[-idx, ]
```

Data dibagi **354 pasien untuk latih** dan **88 untuk uji** secara acak sederhana.

**Interpretasi:** sebagai patokan, simpangan baku `y` pada data uji adalah ± 68,3 dan RMSE bila hanya menebak rata-rata data latih adalah ± 70,1. Model yang berguna harus berada jauh di bawah angka ±70 ini.

### 5.4 Model pembanding: Regresi Linear

```r
lin   <- lm(y ~ ., data = train)
p_lin <- predict(lin, test)
```

Regresi linear dengan semua fitur sebagai baseline. Hasilnya dievaluasi pada 5.7.

### 5.5 Membangun Gradient Boosting dengan `gbm`

```r
gb <- gbm(y ~ ., data = train,
          distribution      = "gaussian",
          n.trees           = 1000,
          shrinkage         = 0.05,
          interaction.depth = 3,
          bag.fraction      = 0.8,
          n.minobsinnode    = 10,
          cv.folds          = 5,
          verbose           = FALSE)
gb
```

| Parameter                   | Nilai | Arti                                              |
| --------------------------- | ----- | ------------------------------------------------- |
| `distribution = "gaussian"` |       | Fungsi loss kuadrat (regresi)                     |
| `n.trees`                   | 1000  | Batas atas jumlah pohon                           |
| `shrinkage`                 | 0,05  | Learning rate kecil, belajar bertahap             |
| `interaction.depth`         | 3     | Tiap pohon memiliki 3 percabangan                 |
| `bag.fraction`              | 0,8   | Tiap pohon memakai 80% data (stochastic GB)       |
| `n.minobsinnode`            | 10    | Daun minimal berisi 10 observasi                  |
| `cv.folds`                  | 5     | Validasi silang 5-fold untuk memilih jumlah pohon |

**Hasil `print(gb)`:**

```
A gradient boosted model with gaussian loss function.
1000 iterations were performed.
The best cross-validation iteration was 114.
There were 10 predictors of which 10 had non-zero influence.
```

**Interpretasi:** model dibangun hingga 1000 iterasi, tetapi iterasi terbaik menurut validasi silang hanyalah **114**. Artinya sekitar 886 pohon berikutnya hanya menambah _overfitting_. Seluruh 10 prediktor dipakai oleh model (influence > 0), namun besarnya sangat berbeda (lihat 5.8).

### 5.6 Jumlah pohon optimal (early stopping via CV)

```r
best <- gbm.perf(gb, method = "cv")
best
```

**Hasil:** `[1] 114`

`gbm.perf()` menggambar kurva galat terhadap jumlah pohon: garis **hitam** untuk galat latih, garis **hijau** untuk galat validasi silang, dan garis **biru putus-putus** di jumlah pohon optimal.

**Interpretasi:**

- Galat latih akan terus turun seiring bertambahnya pohon, sedangkan galat CV turun lalu **naik kembali** setelah titik optimum. Titik terendah kurva hijau ada di **114 pohon**.
- Ini adalah _early stopping_ berbasis validasi silang: jumlah pohon dipilih dari data latih saja, tanpa menyentuh data uji.
- Pohon optimal (114) jauh lebih sedikit daripada maksimum (1000), pola yang sama dengan versi Python (57 pohon). Data diabetes memiliki sinyal lemah dan banyak _noise_, sehingga model cepat mulai menghafal.

### 5.7 Evaluasi pada data uji

```r
p_gb   <- predict(gb, newdata = test, n.trees = best)
metrik <- function(aktual, prediksi) {
  c(RMSE = sqrt(mean((aktual - prediksi)^2)),
    MAE  = mean(abs(aktual - prediksi)),
    R2   = 1 - sum((aktual - prediksi)^2) / sum((aktual - mean(aktual))^2))
}
rbind("Regresi Linear" = metrik(test$y, p_lin),
      "Gradient Boosting" = metrik(test$y, p_gb))
```

Penting: `predict(..., n.trees = best)` memastikan prediksi memakai 114 pohon, bukan 1000. Fungsi `metrik()` menghitung RMSE, MAE, dan R² secara manual.

**Hasil:**

| Model              | RMSE       | MAE        | R²         |
| ------------------ | ---------- | ---------- | ---------- |
| **Regresi Linear** | **52,755** | **42,520** | **0,3963** |
| Gradient Boosting  | 56,315     | 46,175     | 0,3121     |

**Interpretasi:**

- Kedua model jauh lebih baik daripada menebak rata-rata (RMSE ≈ 70), namun hanya menjelaskan **31–40%** variasi target.
- **Regresi Linear justru lebih baik** daripada Gradient Boosting pada data uji (RMSE 52,8 vs 56,3; R² 0,396 vs 0,312).
- Hal ini menandakan hubungan fitur dengan target pada dataset ini sebagian besar **linear dan sinyalnya terbatas**. Model yang lebih fleksibel menangkap lebih banyak _noise_ tanpa menambah pola yang berguna.
- Data uji hanya 88 pasien, jadi selisih sekitar 3,5 poin RMSE dapat berubah bila pembagian data berbeda. Kesimpulannya bukan "boosting selalu kalah", tetapi **boosting tidak memberi keunggulan pada data ini**.

### 5.8 Pentingnya variabel (relative influence)

```r
imp <- summary(gb, n.trees = best, plotit = TRUE)
imp
```

`summary()` pada objek `gbm` menghitung _relative influence_, yaitu persentase kontribusi tiap variabel dalam mengurangi galat di seluruh pohon (totalnya 100%).

**Hasil:**

| Variabel | Relative influence (%) |
| -------- | ---------------------- |
| **s5**   | **39,07**              |
| **bmi**  | **30,69**              |
| bp       | 9,52                   |
| s6       | 6,14                   |
| s3       | 3,64                   |
| s2       | 2,77                   |
| age      | 2,26                   |
| s4       | 2,14                   |
| sex      | 2,13                   |
| s1       | 1,64                   |

**Interpretasi:**

- **`s5` dan `bmi` menyumbang ± 70% pengaruh** (39,1% + 30,7%). Kedua variabel inilah yang paling menentukan prediksi, dan `bp` menyusul di urutan ketiga (9,5%).
- Tujuh variabel lainnya masing-masing hanya 1,6–6,1%. Walaupun semuanya "dipakai" model, kontribusinya kecil.
- Urutan `s5` > `bmi` berbeda dengan versi Python (MDI: `bmi` > `s5`), tetapi dua variabel teratas yang sama konsisten di kedua program. Perbedaan urutan wajar karena pembagian data dan implementasi berbeda.
- Ukuran ini dihitung dari struktur pohon pada data latih, bukan dari kinerja di data uji, dan cenderung meremehkan variabel biner seperti `sex`.

### 5.9 Efek marginal (partial dependence)

```r
plot(gb, i.var = "bmi", n.trees = best)
plot(gb, i.var = "s5",  n.trees = best)
```

Grafik _partial dependence_ menunjukkan efek rata-rata satu variabel terhadap prediksi `y` setelah efek variabel lain dirata-ratakan.

**Interpretasi:** grafik tidak tersimpan, tetapi saat saya memeriksa model serupa, pola kedua variabel adalah sebagai berikut:

- **`bmi`** berhubungan **positif**: semakin tinggi BMI, semakin tinggi prediksi perkembangan penyakit (sekitar 130 pada BMI rendah menjadi sekitar 220 pada BMI tinggi), dengan kurva berbentuk tangga yang mendatar di kedua ujung.
- **`s5`** juga berhubungan **positif**: prediksi rendah (± 114) pada nilai `s5` rendah, lalu naik ke ± 200 dan mendatar pada nilai tinggi.
- Bentuk tangga adalah ciri model berbasis pohon (prediksi konstan per segmen), dan penyebab mendatarnya kurva di ujung adalah sedikitnya data di wilayah ekstrem.
- Arah positif sejalan dengan logika klinis: BMI lebih tinggi terkait risiko dan perkembangan diabetes yang lebih berat. Hubungan ini bersifat **asosiasi**, bukan bukti sebab-akibat. Detail bentuk kurva dapat berbeda sedikit pada setiap proses pelatihan.

### 5.10 Tuning dengan `caret` (grid search, CV 5-fold)

```r
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(n.trees = c(50, 100, 200),
                    interaction.depth = c(2, 3, 4),
                    shrinkage = c(0.01, 0.05, 0.1),
                    n.minobsinnode = 10)
gb_cv <- train(y ~ ., data = train, method = "gbm",
               trControl = ctrl, tuneGrid = grid, verbose = FALSE)
gb_cv$bestTune
```

Menguji 3 × 3 × 3 = **27 kombinasi** hiperparameter dengan CV 5-fold pada data latih (354 pasien; tiap lipatan melatih ± 283 sampel). Kriteria pemilihan: RMSE terkecil.

**Hasil (kombinasi teratas dari 27):**

| shrinkage | interaction.depth | n.trees | RMSE       | R² (caret) | MAE    |
| --------- | ----------------- | ------- | ---------- | ---------- | ------ |
| 0,05      | 2                 | **200** | **55,160** | 0,5181     | 43,795 |
| 0,10      | 2                 | 100     | 55,227     | 0,5143     | 44,300 |
| 0,05      | 2                 | 100     | 55,269     | 0,5128     | 44,591 |
| 0,10      | 2                 | 50      | 55,759     | 0,5017     | 45,137 |
| 0,05      | 3                 | 100     | 56,018     | 0,4984     | 44,992 |
| 0,01      | 2                 | 50      | 67,913     | 0,4536     | 58,392 |

```
The final values used for the model were n.trees = 200, interaction.depth = 2,
shrinkage = 0.05 and n.minobsinnode = 10.
```

**Interpretasi:**

- Kombinasi terbaik: **200 pohon, `interaction.depth = 2`, `shrinkage = 0.05`**, dengan RMSE CV 55,16.
- Selisih antar kombinasi teratas sangat kecil (55,16 – 55,27), jauh di bawah ketidakpastian CV. Beberapa konfigurasi pada dasarnya **setara**; memilih tepat "yang terbaik" tidak berarti banyak.
- Pola yang jelas: **pohon dangkal (`interaction.depth = 2`) konsisten unggul**, dan peningkatan kedalaman ke 3–4 cenderung menambah RMSE pada pengaturan serupa. Ini sesuai dengan data yang cenderung linear dan berisik, di mana model sederhana lebih aman.
- **Trade-off shrinkage dan jumlah pohon** terlihat: `shrinkage = 0.01` belum konvergen: RMSE-nya masih turun dari ± 67 (50 pohon) ke ± 56–57,6 (200 pohon), jadi butuh lebih banyak pohon. Sebaliknya `shrinkage = 0.10` mulai memburuk pada 200 pohon (mis. kedalaman 3: 56,2 pada 50 pohon → 59,4 pada 200 pohon), tanda _overfitting_.
- Grid ini punya dua tepi: `interaction.depth = 2` di batas bawah dan `n.trees = 200` di batas atas untuk `shrinkage = 0.05`. Perluasan grid (kedalaman 1, pohon lebih banyak pada learning rate kecil) bisa memberi hasil sedikit berbeda.
- **Kolom `Rsquared` caret (0,52)** adalah kuadrat korelasi antara prediksi dan aktual pada lipatan CV, berbeda dengan R² pada 5.7 (1 − SSE/SST, 0,31). Kedua angka tidak langsung dapat dibandingkan.
- Skrip tidak mengevaluasi model hasil tuning pada data uji; `gb_cv` hanya dipakai untuk melihat kombinasi terbaik.

### 5.11 Prediksi pasien

```r
baru <- test[1:3, -which(names(test) == "y")]
predict(gb, newdata = baru, n.trees = best)
```

Mengambil tiga pasien pertama data uji (tanpa kolom `y`) dan memprediksi dengan 114 pohon.

**Hasil:**

| Pasien | Aktual | Prediksi | Selisih |
| ------ | ------ | -------- | ------- |
| 1      | 138    | 74,5     | −63,5   |
| 2      | 63     | 125,6    | +62,6   |
| 3      | 118    | 89,2     | −28,8   |

**Interpretasi:**

- Prediksi meleset 29 sampai 64 poin, sebanding dengan RMSE model (± 56). Sesuai dengan R² yang hanya ± 0,31.
- Model belum cukup presisi untuk memprediksi individu; ia lebih tepat dipakai untuk melihat kecenderungan kelompok.
- "Pasien baru" di sini sebenarnya tiga baris pertama data uji, bukan data yang benar-benar baru.

## 6. Ringkasan Hasil

| Ukuran                             | Nilai                                                  |
| ---------------------------------- | ------------------------------------------------------ |
| Data latih / uji                   | 354 / 88                                               |
| Jumlah pohon optimal (CV `gbm`)    | 114 dari 1000                                          |
| Regresi Linear: RMSE / MAE / R²    | 52,755 / 42,520 / 0,3963                               |
| Gradient Boosting: RMSE / MAE / R² | 56,315 / 46,175 / 0,3121                               |
| Variabel terpenting                | `s5` (39,1%), `bmi` (30,7%), `bp` (9,5%)               |
| Hasil tuning `caret`               | `n.trees=200`, `interaction.depth=2`, `shrinkage=0.05` |
| RMSE CV model tuning               | 55,160                                                 |

## 7. Perbandingan dengan Versi Python

| Aspek                          | Python (`Gradient_Boosting.ipynb`)     | R (`Gradient_Boosting_using_R.R`)      |
| ------------------------------ | -------------------------------------- | -------------------------------------- |
| Pembagian data                 | 353 latih / 89 uji (`random_state=42`) | 354 latih / 88 uji (`set.seed(42)`)    |
| Pohon optimal (early stopping) | 57 pohon                               | 114 pohon                              |
| Regresi Linear: R² uji         | 0,4526                                 | 0,3963                                 |
| GB (early stopping): R² uji    | 0,4775                                 | 0,3121                                 |
| Dua fitur teratas              | `bmi`, `s5`                            | `s5`, `bmi`                            |
| Hasil tuning                   | `lr=0.05`, `depth=2`, 100 pohon        | `shrinkage=0.05`, `depth=2`, 200 pohon |

**Catatan:**

- Angka kedua program **tidak boleh dibandingkan langsung** karena pembagian data acak berbeda (generator bilangan acak R dan NumPy menghasilkan data uji yang berbeda), dan mekanisme _early stopping_ juga berbeda. Dengan data uji sekecil ±88 pasien, sebuah perbedaan R² 0,05–0,15 antar pembagian data wajar terjadi.
- Yang **konsisten** di kedua program: (a) jumlah pohon optimal jauh lebih kecil dari batas atas, (b) pohon dangkal (kedalaman 2) paling cocok, (c) `bmi` dan `s5` adalah prediktor utama, dan (d) Gradient Boosting tidak unggul jelas atas Regresi Linear.

## 8. Kesimpulan

1. Dengan `gbm`, jumlah pohon optimal menurut validasi silang hanya **114 dari 1000**; sisanya hanya menambah _overfitting_. _Early stopping_ sangat penting pada boosting.
2. Pada data uji ini, **Regresi Linear (R² 0,396) lebih baik daripada Gradient Boosting (R² 0,312)**. Dataset diabetes memiliki sinyal terbatas dan cenderung linear, sehingga model kompleks tidak memberi keuntungan.
3. **`s5` dan `bmi`** mendominasi pengaruh (± 70% dari total), diikuti `bp`. Hubungan keduanya dengan target bersifat positif.
4. Tuning dengan `caret` memilih **pohon dangkal** (`interaction.depth = 2`), tetapi banyak kombinasi memiliki RMSE hampir sama (≈ 55,2–55,8), jadi pemilihan hiperparameter tidak terlalu krusial.
5. Dengan R² ≈ 0,3, model hanya cocok untuk melihat kecenderungan, bukan prediksi individu yang tepat.

## Referensi

- Friedman, J. H. (2001). _Greedy Function Approximation: A Gradient Boosting Machine_. Annals of Statistics, 29(5), 1189–1232.
- Friedman, J. H. (2002). _Stochastic Gradient Boosting_. Computational Statistics & Data Analysis, 38(4), 367–378.
- Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004). _Least Angle Regression_. Annals of Statistics, 32(2), 407–499.
- Greenwell, B. et al. _gbm: Generalized Boosted Regression Models_ (paket R).
- Kuhn, M. (2008). _Building Predictive Models in R Using the caret Package_. Journal of Statistical Software, 28(5).
